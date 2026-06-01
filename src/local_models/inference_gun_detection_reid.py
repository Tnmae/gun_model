"""
inference_gun_detection_reid.py

Real-time gun detection pipeline:
  - TensorRT yolo11n-pose  → person/pose detection  (fastest nano model)
  - TensorRT best          → gun detection (YOLOv8n fine-tuned, ~6MB)
  - Pose + gun inference run IN PARALLEL via ThreadPoolExecutor
  - non_weapons.pt removed entirely (no cross-verification step)

Draws:
  RED box    → every tracked gun (G-ID + score)
  PURPLE box → the person holding that gun
  RED banner → alert on first detection
"""

import os
import json
import base64
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _model_path(filename: str) -> str:
    return os.path.join(_ROOT, filename)


import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import torchreid
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception as e:
    raise RuntimeError(
        "pip install ultralytics torch torchreid deep-sort-realtime"
    ) from e


# ─────────────────────────────────────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────────────────────────────────────
GUN_COLOR    = (0,   0,   255)   # RED    — gun box
HOLDER_COLOR = (255, 0,   255)   # PURPLE — holder box
ALERT_COLOR  = (0,   0,   255)   # RED    — alert banner

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS: Dict[str, Any] = {
    # ── Models ──────────────────────────────────────────────────────────────
    # TensorRT engines are used when available; falls back to .pt automatically.
    # Run convert_to_tensorrt.py once to generate the .engine files.
    "GUN_MODELS": [
        {
            "path":            _model_path("best.engine"),   # TRT preferred
            "fallback_path":   _model_path("best.pt"),       # .pt fallback
            "weight":          1.0,
            "conf":            0.35,
            "name":            "gun-primary",
            "target_class_id": 0,
        },
    ],

    # Pose model — nano TRT for maximum speed
    "POSE_MODEL_PATH":          _model_path("yolo11n-pose.engine"),
    "POSE_MODEL_FALLBACK_PATH": _model_path("yolo11n-pose.pt"),
    "CONF_THR_POSE":   0.40,
    "CONF_THR_WRIST":  0.20,

    # Ensemble / scoring
    "WBF_IOU_THR":                0.45,
    "AGREEMENT_BONUS":            0.20,
    "FINAL_CONFIDENCE_THRESHOLD": 0.50,

    # Gun size filter
    "GUN_MIN_AREA":          300,
    "GUN_MAX_FRAME_FRACTION":0.07,
    "GUN_MAX_WIDTH":         320,
    "GUN_MAX_HEIGHT":        260,
    "GUN_MAX_ASPECT":        8.0,

    # Holder association
    "WRIST_HALF":            40,
    "MIN_INTERSECTION_FRAC": 0.005,
    "MAX_HOLDER_DIST":       380,

    # Gun IoU tracker
    "GUN_TRACKER_IOU_THRESHOLD":   0.25,
    "GUN_TRACKER_MAX_LOST_FRAMES": 8,

    # Alert
    "ALERT_THRESHOLD":               0.60,
    "ALERT_COOLDOWN_FRAMES":         90,
    "ALERT_ON_FIRST_DETECTION_ONLY": True,

    # Person filter
    "MIN_PERSON_WIDTH":  20,
    "MIN_PERSON_HEIGHT": 40,
    "MIN_PERSON_AREA":   1000,

    # DeepSort
    "TRACKER_MAX_AGE":            60,
    "TRACKER_N_INIT":              2,
    "TRACKER_MAX_IOU_DISTANCE":    0.80,
    "TRACKER_MAX_COSINE_DISTANCE": 0.25,
    "TRACKER_NN_BUDGET":           200,

    # Parallel inference thread pool size
    # 2 workers: pose + gun run concurrently (OSNet disabled)
    "INFERENCE_WORKERS": 2,

    # Gun detection frame-skip: 1 = disabled (run gun model every frame).
    # Increase to 2 or 3 only if fps is insufficient for your hardware.
    "GUN_SKIP_FRAMES": 1,

    # Input frame resize — letterbox to this size before both models.
    # Eliminates the internal TRT resize overhead on every call.
    # Must match the longest side the engines were built for (640).
    "INFER_IMGSZ": 640,

    # Gun inference input size passed to YOLO.predict(imgsz=...).
    # best.engine was exported with dynamic=True so this can be tuned.
    # 480 gives ~30% speedup over 640 with minimal accuracy loss.
    "GUN_INFER_IMGSZ": 480,

    # OSNet ReID — disabled for ~35ms/frame saving.
    # Set True to re-enable appearance-based person re-identification.
    "USE_OSNET": False,

    "VERBOSE": True,
}

LEFT_WRI, RIGHT_WRI = 9, 10


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def log(msg: str, verbose: bool = True):
    if not verbose:
        return
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def area(box) -> float:
    return max(0.0, box[2]-box[0]) * max(0.0, box[3]-box[1])

def intersect(a, b) -> float:
    return (max(0.0, min(a[2],b[2]) - max(a[0],b[0])) *
            max(0.0, min(a[3],b[3]) - max(a[1],b[1])))

def iou(a, b) -> float:
    i = intersect(a, b)
    u = area(a) + area(b) - i
    return 0.0 if u <= 0 else i / u

def box_center(box) -> Tuple[float, float]:
    return (box[0]+box[2])/2.0, (box[1]+box[3])/2.0


# ─────────────────────────────────────────────────────────────────────────────
# Gun size validator
# ─────────────────────────────────────────────────────────────────────────────
def valid_gun_box(bbox, cfg: Dict, frame_shape: Tuple) -> bool:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    ba = bw * bh
    if ba < cfg.get("GUN_MIN_AREA", 400):
        return False
    fh, fw = frame_shape[:2]
    if ba / (fw * fh) > cfg.get("GUN_MAX_FRAME_FRACTION", 0.06):
        return False
    if bw > cfg.get("GUN_MAX_WIDTH", 280):
        return False
    if bh > cfg.get("GUN_MAX_HEIGHT", 220):
        return False
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    if aspect > cfg.get("GUN_MAX_ASPECT", 7.0):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Sequential ID mapper
# ─────────────────────────────────────────────────────────────────────────────
class SequentialIDMapper:
    def __init__(self):
        self._map: Dict[int, int] = {}
        self.next_id = 1

    def get(self, raw: int) -> int:
        if raw not in self._map:
            self._map[raw] = self.next_id
            self.next_id += 1
        return self._map[raw]

    def reset(self):
        self._map.clear()
        self.next_id = 1


# ─────────────────────────────────────────────────────────────────────────────
# Gun IoU tracker — stable G-IDs across frames
# ─────────────────────────────────────────────────────────────────────────────
class GunTracker:
    def __init__(self, iou_thr: float = 0.30, max_lost: int = 6):
        self.iou_thr  = iou_thr
        self.max_lost = max_lost
        self._tracks: Dict[int, Dict] = {}
        self._nid = 1

    def update(self, dets: List[Tuple[np.ndarray, float]]
               ) -> List[Tuple[int, np.ndarray, float]]:
        for t in self._tracks.values():
            t["lost"] += 1

        result = []
        if dets and self._tracks:
            tids   = list(self._tracks.keys())
            tboxes = [self._tracks[tid]["bbox"] for tid in tids]
            unmat  = list(range(len(dets)))
            mat    = np.zeros((len(dets), len(tids)), dtype=np.float32)
            for di, (db, _) in enumerate(dets):
                for tj, tb in enumerate(tboxes):
                    mat[di, tj] = iou(tuple(db), tuple(tb))
            while True:
                fi = np.unravel_index(np.argmax(mat), mat.shape)
                if mat[fi] < self.iou_thr:
                    break
                di, tj = fi
                tid = tids[tj]
                db, ds = dets[di]
                self._tracks[tid].update(bbox=db, score=ds, lost=0)
                result.append((tid, db, ds))
                mat[di, :] = -1
                mat[:, tj] = -1
                unmat.remove(di)
            for di in unmat:
                db, ds = dets[di]
                self._tracks[self._nid] = dict(bbox=db, score=ds, lost=0)
                result.append((self._nid, db, ds))
                self._nid += 1
        else:
            for db, ds in dets:
                self._tracks[self._nid] = dict(bbox=db, score=ds, lost=0)
                result.append((self._nid, db, ds))
                self._nid += 1

        stale = [t for t, v in self._tracks.items() if v["lost"] > self.max_lost]
        for t in stale:
            del self._tracks[t]
        return result

    def reset(self):
        self._tracks.clear()
        self._nid = 1


# ─────────────────────────────────────────────────────────────────────────────
# Alert manager
# ─────────────────────────────────────────────────────────────────────────────
class AlertManager:
    def __init__(self, cfg: Dict):
        self.cfg     = cfg
        self.history: Dict[int, int] = {}
        self.frame   = 0

    def check(self, pid: int, conf: float) -> bool:
        if conf < self.cfg["ALERT_THRESHOLD"]:
            return False
        pid = int(pid)
        if pid not in self.history:
            self.history[pid] = self.frame
            return True
        if self.cfg["ALERT_ON_FIRST_DETECTION_ONLY"]:
            return False
        if self.frame - self.history[pid] >= self.cfg["ALERT_COOLDOWN_FRAMES"]:
            self.history[pid] = self.frame
            return True
        return False

    def advance(self): self.frame += 1
    def reset(self):   self.history.clear(); self.frame = 0


# ─────────────────────────────────────────────────────────────────────────────
# OSNet ReID
# ─────────────────────────────────────────────────────────────────────────────
def load_osnet(device: str):
    m = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=1000, pretrained=True)
    m.eval().to(device)
    return m

def osnet_prep(img: np.ndarray) -> torch.Tensor:
    img = cv2.resize(img, (128, 256))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

@torch.no_grad()
def osnet_encode(model, crop: np.ndarray, device: str) -> Optional[np.ndarray]:
    if crop is None or crop.size == 0:
        return None
    try:
        t = osnet_prep(crop).to(device)
        f = F.normalize(model(t), p=2, dim=1)
        return f.cpu().numpy().flatten()
    except Exception:
        return None


@torch.no_grad()
def _osnet_encode_batch(model, crops: List[np.ndarray], device: str) -> List[Optional[np.ndarray]]:
    """
    Run OSNet on all crops in a single batched forward pass.
    Falls back to None for any crop that fails preprocessing.
    Significantly faster than N serial calls when multiple persons are in frame.
    """
    if not crops:
        return []
    tensors = []
    valid   = []   # (original_index, tensor_index)
    for i, crop in enumerate(crops):
        if crop is None or crop.size == 0:
            tensors.append(None)
            continue
        try:
            tensors.append(osnet_prep(crop))
            valid.append((i, len([t for t in tensors if t is not None]) - 1))
        except Exception:
            tensors.append(None)

    good = [t for t in tensors if t is not None]
    results: List[Optional[np.ndarray]] = [None] * len(crops)
    if not good:
        return results

    try:
        batch = torch.cat(good, dim=0).to(device)
        feats = F.normalize(model(batch), p=2, dim=1).cpu().numpy()
        feat_idx = 0
        for i, t in enumerate(tensors):
            if t is not None:
                results[i] = feats[feat_idx].flatten()
                feat_idx += 1
    except Exception:
        pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Model loader — resolves TRT engine path with .pt fallback
# ─────────────────────────────────────────────────────────────────────────────
def _letterbox_frame(frame: np.ndarray, target: int = 640) -> Tuple[np.ndarray, float, Tuple[int,int]]:
    """
    Resize frame so the longest side == target, padding the short side with
    grey to keep aspect ratio. Returns (resized_frame, scale, (pad_w, pad_h)).
    Coordinates from model output must be unscaled with _unscale_boxes().
    """
    h, w = frame.shape[:2]
    scale = target / max(h, w)
    if scale != 1.0:
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        new_h, new_w = h, w
    pad_w = (target - new_w) // 2
    pad_h = (target - new_h) // 2
    frame = cv2.copyMakeBorder(frame, pad_h, target - new_h - pad_h,
                                pad_w, target - new_w - pad_w,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return frame, scale, (pad_w, pad_h)


def _unscale_boxes(boxes: np.ndarray, scale: float,
                   pad: Tuple[int,int]) -> np.ndarray:
    """Invert letterbox transform on xyxy boxes."""
    if boxes.shape[0] == 0:
        return boxes
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / scale
    return out


def _unscale_kpts(kpts: np.ndarray, scale: float,
                  pad: Tuple[int,int]) -> np.ndarray:
    """Invert letterbox transform on keypoints array (N, 17, 3)."""
    if kpts.shape[0] == 0:
        return kpts
    out = kpts.copy().astype(np.float32)
    out[:, :, 0] = (out[:, :, 0] - pad[0]) / scale
    out[:, :, 1] = (out[:, :, 1] - pad[1]) / scale
    return out


def _resolve_model_path(primary: str, fallback: str, label: str, verbose: bool) -> str:
    """Return the TRT engine path if it exists, otherwise fall back to .pt.
    For pose/standard YOLO models, if neither file exists locally, return the
    model name so ultralytics can auto-download it from its model hub."""
    if os.path.exists(primary):
        log(f"✓ TensorRT engine: {os.path.basename(primary)} [{label}]", verbose)
        return primary
    if os.path.exists(fallback):
        log(f"⚠ TRT engine not found — using .pt fallback: {os.path.basename(fallback)} [{label}]"
            f"  (run convert_to_tensorrt.py for full speed)", verbose)
        return fallback
    # Neither file found locally — let ultralytics download by model name
    model_name = os.path.basename(fallback)
    log(f"⚠ [{label}] '{os.path.basename(primary)}' not found, "
        f"auto-downloading '{model_name}' from ultralytics hub...", verbose)
    return model_name


def model_fn(overrides: Optional[Dict] = None) -> Dict:
    cfg     = {**DEFAULTS, **(overrides or {})}
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    verbose = cfg.get("VERBOSE", True)
    log(f"Device: {device}", verbose)

    # ── Gun models ────────────────────────────────────────────────────────
    gun_models = []
    for gm in cfg["GUN_MODELS"]:
        try:
            path = _resolve_model_path(
                gm["path"],
                gm.get("fallback_path", gm["path"].replace(".engine", ".pt")),
                gm["name"],
                verbose,
            )
            gc = gm.copy()
            # Pass task="detect" explicitly to suppress the auto-guess warning
            gc["model"] = YOLO(path, task="detect")
            gun_models.append(gc)
        except Exception as e:
            log(f"✗ Gun model {gm['name']}: {e}", verbose)

    if not gun_models:
        raise RuntimeError("No gun models loaded — check model paths.")

    # ── Pose model (nano TRT) ─────────────────────────────────────────────
    pose_path = _resolve_model_path(
        cfg["POSE_MODEL_PATH"],
        cfg.get("POSE_MODEL_FALLBACK_PATH", "yolo11n-pose.pt"),
        "pose",
        verbose,
    )
    pose_model = YOLO(pose_path)
    log("✓ Pose model loaded", verbose)

    # ── OSNet ReID — disabled for performance (~35ms saved per frame)
    # DeepSort falls back to IoU-only matching, which is sufficient for
    # fixed-camera security feeds. Re-enable by setting USE_OSNET: True.
    osnet = None
    if cfg.get("USE_OSNET", False):
        try:
            osnet = load_osnet(device)
            log("✓ OSNet ReID", verbose)
        except Exception as e:
            log(f"⚠ OSNet: {e}", verbose)
    else:
        log("OSNet ReID disabled (USE_OSNET=False) — IoU-only tracking", verbose)

    # ── Trackers ──────────────────────────────────────────────────────────
    tracker = DeepSort(
        max_age=cfg["TRACKER_MAX_AGE"],
        n_init=cfg["TRACKER_N_INIT"],
        max_iou_distance=cfg["TRACKER_MAX_IOU_DISTANCE"],
        # When OSNet is disabled pass embedder=None and set max_cosine_distance
        # to 1.0 so DeepSort never tries to compute cosine similarity on None
        # embeddings — it falls back to pure IoU matching.
        max_cosine_distance=cfg["TRACKER_MAX_COSINE_DISTANCE"] if cfg.get("USE_OSNET", False) else 1.0,
        nn_budget=cfg["TRACKER_NN_BUDGET"] if cfg.get("USE_OSNET", False) else None,
        embedder=None,
    )

    gun_tracker = GunTracker(
        iou_thr=cfg["GUN_TRACKER_IOU_THRESHOLD"],
        max_lost=cfg["GUN_TRACKER_MAX_LOST_FRAMES"],
    )

    # ── Shared thread pool for parallel inference ─────────────────────────
    inference_pool = ThreadPoolExecutor(
        max_workers=cfg.get("INFERENCE_WORKERS", 2),
        thread_name_prefix="infer",
    )

    log("✓ All models loaded — parallel inference enabled", verbose)

    return {
        "gun_models":     gun_models,
        "pose_model":     pose_model,
        "tracker":        tracker,
        "gun_tracker":    gun_tracker,
        "id_mapper":      SequentialIDMapper(),
        "osnet":          osnet,
        "alerts":         AlertManager(cfg),
        "device":         device,
        "config":         cfg,
        "inference_pool": inference_pool,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parallel inference workers
# These run concurrently — pose and gun detection overlap on the same frame.
# ─────────────────────────────────────────────────────────────────────────────
def _run_pose(frame: np.ndarray, pose_model, conf_thr: float):
    """Worker: run pose model and return (boxes, scores, keypoints).
    stream=True keeps the CUDA stream open so the GPU doesn't sync between
    the pose and gun TRT calls when they run in parallel threads."""
    r = pose_model.predict(frame, conf=conf_thr, verbose=False, stream=True)
    r = list(r)[0]
    boxes  = np.zeros((0, 4))
    scores = np.zeros(0)
    kpts   = np.zeros((0, 17, 3))
    if len(getattr(r, "boxes", [])):
        boxes  = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        try:    kpts = r.keypoints.data.cpu().numpy()
        except Exception:
            try: kpts = r.keypoints.cpu().numpy()
            except Exception: kpts = np.zeros((len(boxes), 17, 3))
    return boxes, scores, kpts


def _run_gun_model(frame: np.ndarray, m: Dict, imgsz: int = 640):
    """Worker: run one gun model and return list of (bbox, weighted_score).
    stream=True avoids a CUDA sync stall when pose runs concurrently."""
    results = []
    try:
        r = m["model"].predict(frame, conf=m["conf"], verbose=False,
                               imgsz=imgsz, stream=True)
        r = list(r)[0]
        if not len(getattr(r, "boxes", [])):
            return results
        boxes  = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls    = r.boxes.cls.cpu().numpy().astype(int)
        mask   = cls == m.get("target_class_id", 0)
        for b, s in zip(boxes[mask], scores[mask]):
            results.append((b, float(s) * m.get("weight", 1.0)))
    except Exception:
        pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Holder association
# ─────────────────────────────────────────────────────────────────────────────
def associate_holder(gun_bbox:  np.ndarray,
                     persons:   List,
                     kpts_list: List,
                     cfg:       Dict) -> Optional[int]:
    if not persons:
        return None

    ga       = area(tuple(gun_bbox))
    gcx, gcy = box_center(gun_bbox)
    max_dist = cfg.get("MAX_HOLDER_DIST", 350)

    # Step 1: wrist keypoint overlap
    best_pid, best_overlap = None, -1.0
    for idx, person in enumerate(persons):
        l, t_, r, b_, pid = person
        kp = kpts_list[idx]
        if kp is None:
            continue
        for wi in (LEFT_WRI, RIGHT_WRI):
            if kp.shape[0] <= wi or kp[wi][2] < cfg["CONF_THR_WRIST"]:
                continue
            wx, wy = float(kp[wi][0]), float(kp[wi][1])
            wh = cfg["WRIST_HALF"]
            wb = [wx-wh, wy-wh, wx+wh, wy+wh]
            frac = intersect(tuple(gun_bbox), tuple(wb)) / max(1.0, ga)
            if frac > best_overlap:
                best_overlap = frac
                best_pid     = pid

    if best_pid is not None and best_overlap >= cfg["MIN_INTERSECTION_FRAC"]:
        return best_pid

    # Step 2: gun centroid inside person box
    for person in persons:
        l, t_, r, b_, pid = person
        if l <= gcx <= r and t_ <= gcy <= b_:
            return pid

    # Step 3: nearest person within distance gate
    best_pid, best_dist = None, float("inf")
    for person in persons:
        l, t_, r, b_, pid = person
        dist = math.hypot(gcx - (l+r)/2.0, gcy - (t_+b_)/2.0)
        if dist < best_dist:
            best_dist = dist
            best_pid  = pid

    if best_pid is not None and best_dist <= max_dist:
        return best_pid

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main inference — parallel pose + gun detection
# ─────────────────────────────────────────────────────────────────────────────
def predict_fn(input_data: Dict, model: Dict) -> Dict:
    alerts    = model["alerts"]
    id_mapper = model["id_mapper"]
    tracker   = model["tracker"]
    gun_trk   = model["gun_tracker"]
    pose      = model["pose_model"]
    osnet     = model["osnet"]
    device    = model["device"]
    cfg       = model["config"]
    pool      = model["inference_pool"]

    frame        = input_data["frame"]
    cam_id       = input_data.get("cam_id", -1)
    org_id       = input_data.get("org_id", -1)
    user_id      = input_data.get("user_id", -1)
    frame_number = input_data.get("frame_number", 0)
    timestamp    = datetime.now(timezone.utc).isoformat()

    try:
        alerts.advance()
        h, w = frame.shape[:2]

        # ── Letterbox resize — eliminates internal TRT resize overhead ────
        # Both models receive a pre-resized frame at exactly the target size.
        infer_sz = cfg.get("INFER_IMGSZ", 640)
        if max(h, w) != infer_sz:
            infer_frame, lb_scale, lb_pad = _letterbox_frame(frame, infer_sz)
        else:
            infer_frame, lb_scale, lb_pad = frame, 1.0, (0, 0)

        # ── PARALLEL: submit pose + gun simultaneously ────────────────────
        futures = {}

        # Pose runs every frame (34ms TRT)
        futures["pose"] = pool.submit(_run_pose, infer_frame, pose, cfg["CONF_THR_POSE"])

        # Gun detection: run every GUN_SKIP_FRAMES frames
        gun_skip  = max(1, cfg.get("GUN_SKIP_FRAMES", 1))
        gun_imgsz = cfg.get("GUN_INFER_IMGSZ", 480)
        run_gun   = (frame_number % gun_skip == 0)
        if run_gun:
            for i, m in enumerate(model["gun_models"]):
                futures[f"gun_{i}"] = pool.submit(_run_gun_model, infer_frame, m, gun_imgsz)

        # ── Collect pose results and unscale to original coords ───────────
        p_boxes_lb, p_scores, kpts_lb = futures["pose"].result()
        p_boxes = _unscale_boxes(p_boxes_lb, lb_scale, lb_pad)
        kpts    = _unscale_kpts(kpts_lb, lb_scale, lb_pad)

        # ── Build crop list from pose boxes ──────────────────────────────
        ds_dets = []
        crops   = []
        for b, s in zip(p_boxes, p_scores):
            x1, y1, x2, y2 = map(int, b)
            if (x2-x1)*(y2-y1) < cfg["MIN_PERSON_AREA"]:
                continue
            ds_dets.append(([x1, y1, x2-x1, y2-y1], float(s), "person"))
            if osnet:
                crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                crops.append(crop)

        # ── Submit OSNet as a future — overlaps with gun result collection ─
        # While we wait for the gun model below, OSNet runs concurrently.
        if osnet and crops:
            futures["osnet"] = pool.submit(_osnet_encode_batch, osnet, crops, device)

        # ── Collect gun results (propagate on skipped frames) ────────────
        if run_gun:
            raw_preds: List[Tuple[np.ndarray, float]] = []
            for i in range(len(model["gun_models"])):
                raw_preds.extend(futures[f"gun_{i}"].result())
            # Unscale gun boxes from letterbox coords back to original frame coords
            raw_preds = [(_unscale_boxes(b[np.newaxis], lb_scale, lb_pad)[0], s)
                         for b, s in raw_preds]
        else:
            raw_preds = [
                (np.array(v["bbox"]), v["score"])
                for v in model["gun_tracker"]._tracks.values()
                if v["lost"] == 0
            ]

        # ── Collect OSNet embeddings (likely already done by now) ─────────
        # When OSNet is disabled, pass unit-vectors so DeepSort's cosine
        # metric always receives valid normalised 2D arrays (no NaN/divide).
        # nn_budget=None ensures no gallery is built, so cosine distance
        # is never actually used for matching — IoU dominates.
        _EMBED_DIM = 512
        if osnet:
            embeds = [None] * len(ds_dets)
            if "osnet" in futures:
                embeds = futures["osnet"].result()
        else:
            unit = np.ones(_EMBED_DIM, dtype=np.float32) / np.sqrt(_EMBED_DIM)
            embeds = [unit.copy() for _ in ds_dets]

        tracks = tracker.update_tracks(ds_dets, embeds=embeds, frame=frame)

        # Keypoint alignment map
        pose_boxes = p_boxes if len(p_boxes) > 0 else np.zeros((0, 4))
        kpts_map: Dict[int, Optional[np.ndarray]] = {}
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            cid  = id_mapper.get(int(t.track_id))
            tbox = np.array(list(map(float, t.to_ltrb())))
            bi, bv = -1, 0.0
            for i, pb in enumerate(pose_boxes):
                v = iou(tuple(tbox), tuple(pb))
                if v > bv:
                    bv, bi = v, i
            kpts_map[cid] = (
                kpts[bi] if bi >= 0 and bv >= 0.3 and len(kpts) > bi else None
            )

        persons = []
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            cid = id_mapper.get(int(t.track_id))
            l, t_, r, b_ = map(int, t.to_ltrb())
            if (r-l) < cfg["MIN_PERSON_WIDTH"] or (b_-t_) < cfg["MIN_PERSON_HEIGHT"]:
                continue
            persons.append([l, t_, r, b_, cid])

        kpts_list = [kpts_map.get(p[4]) for p in persons]

        # Agreement bonus across models
        fused: List[Tuple[np.ndarray, float]] = []
        for i, (b, s) in enumerate(raw_preds):
            agreements = sum(
                iou(tuple(b), tuple(raw_preds[j][0])) > cfg["WBF_IOU_THR"]
                for j in range(len(raw_preds)) if j != i
            )
            bonus = cfg["AGREEMENT_BONUS"] if agreements >= 1 else 0
            fused.append((b, min(1.0, s + bonus)))

        # NMS
        if fused:
            all_b = [f[0] for f in fused]
            all_s = [f[1] for f in fused]
            order = np.argsort(all_s)[::-1]
            used  = set()
            kept  = []
            for i in order:
                if i in used:
                    continue
                kept.append(i)
                for j in order:
                    if j != i and j not in used:
                        if iou(tuple(all_b[i]), tuple(all_b[j])) >= 0.40:
                            used.add(j)
                used.add(i)
            fused = [fused[i] for i in kept]

        # ── Size filter + confidence threshold (no non-weapon verifier) ───
        verified: List[Tuple[np.ndarray, float]] = []
        for b, s in fused:
            if s < cfg["FINAL_CONFIDENCE_THRESHOLD"]:
                continue
            if not valid_gun_box(b, cfg, frame.shape):
                continue
            verified.append((b, s))

        # ── Gun tracker ───────────────────────────────────────────────────
        tracked_guns = gun_trk.update(verified)

        # ── Holder association ────────────────────────────────────────────
        gun_dets:       List[Dict]      = []
        active_holders: Dict[int, Dict] = {}
        new_alerts = 0

        for gun_id, gun_bbox, gun_score in tracked_guns:
            if gun_score < cfg["FINAL_CONFIDENCE_THRESHOLD"]:
                continue

            holder_id = associate_holder(gun_bbox, persons, kpts_list, cfg)
            if holder_id is None:
                continue

            gun_dets.append({
                "gun_id":    gun_id,
                "bbox":      gun_bbox.tolist(),
                "score":     round(gun_score, 3),
                "holder_id": holder_id,
            })

            prev = active_holders.get(holder_id)
            if prev is None or gun_score > prev["conf"]:
                for person in persons:
                    l, t_, r, b_, pid = person
                    if pid == holder_id:
                        active_holders[holder_id] = {
                            "bbox": [l, t_, r, b_],
                            "conf": gun_score,
                        }
                        break

            if alerts.check(holder_id, gun_score):
                new_alerts += 1

        # ── Draw ──────────────────────────────────────────────────────────
        out = frame.copy()

        for pid, info in active_holders.items():
            l, t_, r, b_ = info["bbox"]
            cv2.rectangle(out, (int(l), int(t_)), (int(r), int(b_)),
                          HOLDER_COLOR, 3)
            cv2.putText(out,
                        f"ID:{pid} [ARMED] ({info['conf']:.2f})",
                        (min(int(l), w-200), max(0, int(t_)-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, HOLDER_COLOR, 2)

        for g in gun_dets:
            x1, y1, x2, y2 = map(int, g["bbox"])
            label = f"G{g['gun_id']} {g['score']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), GUN_COLOR, 2)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            ly = max(0, y1-6)
            cv2.rectangle(out, (x1, ly-th-2), (x1+tw+2, ly+2), GUN_COLOR, -1)
            cv2.putText(out, label, (x1, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

        if new_alerts > 0:
            cv2.rectangle(out, (0, 0), (w, 44), ALERT_COLOR, -1)
            cv2.putText(out,
                        f"!  {new_alerts} NEW ARMED PERSON"
                        f"{'S' if new_alerts > 1 else ''} DETECTED",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (255, 255, 255), 2)

        ok, jpg = cv2.imencode(".jpg", out)
        b64_out = base64.b64encode(jpg.tobytes()).decode("utf-8") if ok else ""

        return {
            "cam_id":          cam_id,
            "org_id":          org_id,
            "user_id":         user_id,
            "frame_number":    frame_number,
            "timestamp":       timestamp,
            "guns":            gun_dets,
            "active_holders":  list(active_holders.keys()),
            "new_alerts":      new_alerts,
            "annotated_frame": b64_out,
            "stats": {
                "raw_preds":       len(raw_preds),
                "verified_guns":   len(verified),
                "guns_drawn":      len(gun_dets),
                "holders_drawn":   len(active_holders),
                "persons_tracked": len(persons),
            },
            "status": 0,
        }

    except Exception as exc:
        import traceback; traceback.print_exc()
        return {
            "cam_id": cam_id, "org_id": org_id, "user_id": user_id,
            "frame_number": frame_number,
            "status": 1, "error": str(exc),
            "guns": [], "active_holders": [],
            "new_alerts": 0, "annotated_frame": "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Adapter functions — bridge between gun_detection.py and predict_fn
# ─────────────────────────────────────────────────────────────────────────────
def input_frame_fn(payload: Dict, content_type: str = "application/json") -> Dict:
    """Decode a base64 payload into a numpy frame dict."""
    b64 = payload.get("encoding", "")
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    arr   = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image from base64 payload")
    return {
        "frame":        frame,
        "cam_id":       payload.get("cam_id", -1),
        "org_id":       payload.get("org_id", -1),
        "user_id":      payload.get("user_id", -1),
        "frame_number": payload.get("frame_number", 0),
    }


def predict_frame_fn(input_data: Dict, model: Dict) -> Dict:
    """Thin wrapper so gun_detection.py can call predict_fn by name."""
    return predict_fn(input_data, model)


def output_frame_fn(prediction: Dict) -> Dict:
    """
    Reshape predict_fn output into the format expected by the websocket /
    gun_detection.py callers.
    """
    guns = prediction.get("guns", [])

    gun_holders = [
        {"track_id": g["holder_id"], "confidence": g["score"]}
        for g in guns
    ]

    alerts = []
    if prediction.get("new_alerts", 0) > 0:
        for pid in prediction.get("active_holders", []):
            conf = next(
                (g["score"] for g in guns if g["holder_id"] == pid), 0.0
            )
            level = "CRITICAL" if conf >= 0.85 else "HIGH"
            alerts.append({
                "track_id":   pid,
                "confidence": conf,
                "level":      level,
                "timestamp":  prediction.get("timestamp", ""),
            })

    return {
        "cam_id":          prediction.get("cam_id", -1),
        "org_id":          prediction.get("org_id", -1),
        "user_id":         prediction.get("user_id", -1),
        "frame_number":    prediction.get("frame_number", 0),
        "timestamp":       prediction.get("timestamp", ""),
        "guns":            guns,
        "gun_holders":     gun_holders,
        "persons_present": prediction.get("active_holders", []),
        "alerts":          alerts,
        "annotated_frame": prediction.get("annotated_frame", ""),
        "stats":           prediction.get("stats", {}),
        "status":          prediction.get("status", 0),
        "error":           prediction.get("error", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Video processor (for offline testing)
# ─────────────────────────────────────────────────────────────────────────────
def process_video(video_path:  str,
                  output_path: str = "output.mp4",
                  max_frames:  Optional[int] = None,
                  overrides:   Optional[Dict] = None) -> Dict:

    mdl     = model_fn(overrides)
    cfg     = mdl["config"]
    verbose = cfg.get("VERBOSE", True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log(f"Video: {total} frames @ {fps:.1f}fps", verbose)

    writer       = None
    fc           = 0
    total_guns   = 0
    unique_armed = set()
    total_alerts = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fc += 1
        if max_frames and fc > max_frames:
            break

        res = predict_fn({"cam_id": 0, "frame_number": fc, "frame": frame}, mdl)

        if writer is None:
            fh, fw = frame.shape[:2]
            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
            log(f"Writing: {output_path} ({fw}×{fh})", verbose)

        b64 = res.get("annotated_frame", "")
        if b64:
            try:
                arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
                ann = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                writer.write(ann if ann is not None else frame)
            except Exception:
                writer.write(frame)
        else:
            writer.write(frame)

        st = res.get("stats", {})
        total_guns   += st.get("guns_drawn", 0)
        total_alerts += res.get("new_alerts", 0)
        unique_armed.update(res.get("active_holders", []))

        if verbose and fc % 30 == 0:
            log(f"F{fc}/{total or '?'} | "
                f"Raw:{st.get('raw_preds',0)} "
                f"Verified:{st.get('verified_guns',0)} "
                f"Guns:{st.get('guns_drawn',0)} "
                f"Holders:{st.get('holders_drawn',0)} "
                f"Persons:{st.get('persons_tracked',0)} "
                f"Alerts:{res.get('new_alerts',0)}", verbose)

    cap.release()
    if writer:
        writer.release()

    summary = {
        "total_frames":   fc,
        "total_gun_dets": total_guns,
        "unique_armed":   len(unique_armed),
        "total_alerts":   total_alerts,
    }
    log(f"Done — {fc} frames | {len(unique_armed)} armed | "
        f"{total_alerts} alerts", verbose)
    return summary
