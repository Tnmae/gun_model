"""
inference_gun_detection_clean.py

Draws ONLY:
  RED box    → every tracked gun (G-ID + score)
  PURPLE box → the person holding that gun
  RED banner → alert on first detection

Rules:
  - No person nearby = gun not drawn (eliminates cars, empty rooms, objects)
  - Gun box only drawn when holder is confirmed this frame
  - No green boxes, no memory boxes, no unarmed person boxes
"""

import os
import json
import base64
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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
    "GUN_MODELS": [
        {
            "path":            r"E:\All_models\gun_detection\gun_dd_f_best.pt",
            "weight":          1.0,
            "conf":            0.40,
            "name":            "gun-primary",
            "target_class_id": 0,
        },
        {
            "path":            r"E:\All_models\gun_detection\gunn2.pt",
            "weight":          0.9,
            "conf":            0.40,
            "name":            "gun-secondary",
            "target_class_id": 0,
        },
    ],

    "NON_WEAPON_MODEL": {
        "path":               r"E:\All_models\gun_detection\non_weapons.pt",
        "conf":               0.50,    # only run verifier on confident detections
        "rejection_threshold":0.80,    # only reject if VERY confident it's not a gun
        "name":               "non-weapon-verifier",
        "target_class_id":    2,
    },

    "POSE_MODEL_PATH": r"yolo11x-pose.pt",
    "CONF_THR_POSE":   0.45,           # slightly lower — catch partial persons
    "CONF_THR_WRIST":  0.25,           # lower — catch more wrist keypoints

    # Ensemble / scoring
    "WBF_IOU_THR":                0.50,
    "AGREEMENT_BONUS":            0.25,
    "FINAL_CONFIDENCE_THRESHOLD": 0.55,

    # Gun size filter
    # Relative to frame area (0.0–1.0) AND absolute pixel limits
    "GUN_MIN_AREA":          400,      # minimum px² — ignore tiny noise
    "GUN_MAX_FRAME_FRACTION":0.06,     # max 6% of frame area — rejects cars/floors
    "GUN_MAX_WIDTH":         280,      # no held gun is wider than this in px
    "GUN_MAX_HEIGHT":        220,      # no held gun is taller than this in px
    "GUN_MAX_ASPECT":        7.0,      # max width/height ratio

    # Holder association
    "WRIST_HALF":            35,       # wrist patch radius px
    "MIN_INTERSECTION_FRAC": 0.008,    # wrist-gun overlap fraction
    "MAX_HOLDER_DIST":       350,      # px — max distance gun→person center
                                       # if no person within this → gun NOT drawn

    # Gun IoU tracker
    "GUN_TRACKER_IOU_THRESHOLD":   0.30,
    "GUN_TRACKER_MAX_LOST_FRAMES": 6,

    # Alert
    "ALERT_THRESHOLD":               0.68,
    "ALERT_COOLDOWN_FRAMES":         90,
    "ALERT_ON_FIRST_DETECTION_ONLY": True,

    # Person filter
    "MIN_PERSON_WIDTH":  25,
    "MIN_PERSON_HEIGHT": 50,
    "MIN_PERSON_AREA":   1500,

    # DeepSort
    "TRACKER_MAX_AGE":            50,
    "TRACKER_N_INIT":              2,
    "TRACKER_MAX_IOU_DISTANCE":    0.75,
    "TRACKER_MAX_COSINE_DISTANCE": 0.22,
    "TRACKER_NN_BUDGET":           200,

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
# Geometry
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
# Uses frame-relative area so it works at any resolution.
# Rejects: cars, floors, walls, giant dark objects.
# ─────────────────────────────────────────────────────────────────────────────
def valid_gun_box(bbox, cfg: Dict, frame_shape: Tuple) -> bool:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    ba = bw * bh

    # Too small — noise
    if ba < cfg.get("GUN_MIN_AREA", 400):
        return False

    # Too large relative to frame — car, floor, wall
    fh, fw = frame_shape[:2]
    frame_area = fw * fh
    if ba / frame_area > cfg.get("GUN_MAX_FRAME_FRACTION", 0.06):
        return False

    # Absolute pixel size limits — no handheld gun is this large in frame
    if bw > cfg.get("GUN_MAX_WIDTH", 280):
        return False
    if bh > cfg.get("GUN_MAX_HEIGHT", 220):
        return False

    # Aspect ratio — guns are elongated, not square/boxy like cars
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
# Non-weapon verifier
# ─────────────────────────────────────────────────────────────────────────────
class NonWeaponVerifier:
    def __init__(self, cfg: Dict, verbose: bool = True):
        self.cfg   = cfg
        self.model = YOLO(cfg["path"])
        log(f"✓ Non-weapon verifier: {cfg['name']}", verbose)

    def verify(self, frame: np.ndarray, bbox) -> bool:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return True
        try:
            r = self.model.predict(crop, conf=self.cfg["conf"], verbose=False)[0]
            if len(r.boxes) == 0:
                return True
            confs = r.boxes.conf.cpu().numpy()
            cls   = r.boxes.cls.cpu().numpy().astype(int)
            top   = confs.argmax()
            # Only reject if classifier is VERY confident it's a non-weapon
            return not (
                cls[top] == self.cfg.get("target_class_id", 2) and
                confs[top] > self.cfg["rejection_threshold"]
            )
        except Exception:
            return True


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


# ─────────────────────────────────────────────────────────────────────────────
# Person detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_persons(frame, pose_model, conf_thr):
    r = pose_model.predict(frame, conf=conf_thr, verbose=False)[0]
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


# ─────────────────────────────────────────────────────────────────────────────
# Holder association
#
# Step 1 — Wrist keypoint overlap  (most accurate)
# Step 2 — Gun centroid inside person bounding box
# Step 3 — Nearest person within MAX_HOLDER_DIST px
#           Returns None if no person close enough → gun NOT drawn
# ─────────────────────────────────────────────────────────────────────────────
def associate_holder(gun_bbox:  np.ndarray,
                     persons:   List,
                     kpts_list: List,
                     cfg:       Dict) -> Optional[int]:

    # No persons in frame → nothing to associate → gun not drawn
    if not persons:
        return None

    ga       = area(tuple(gun_bbox))
    gcx, gcy = box_center(gun_bbox)
    max_dist = cfg.get("MAX_HOLDER_DIST", 350)

    # ── Step 1: Wrist keypoint overlap ────────────────────────────────────
    best_pid  = None
    best_dist = float("inf")
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
            if ga > 0:
                frac = intersect(tuple(gun_bbox), tuple(wb)) / max(1, ga)
                if frac < cfg["MIN_INTERSECTION_FRAC"]:
                    continue
            dist = math.hypot((l+r)/2 - wx, (t_+b_)/2 - wy)
            if dist < best_dist:
                best_dist = dist
                best_pid  = pid

    if best_pid is not None:
        return best_pid

    # ── Step 2: Gun centroid inside person bounding box ───────────────────
    for person in persons:
        l, t_, r, b_, pid = person
        if l <= gcx <= r and t_ <= gcy <= b_:
            return pid

    # ── Step 3: Nearest person — only within MAX_HOLDER_DIST ─────────────
    # This distance gate is what prevents cars/objects from being drawn:
    # a car in an empty parking lot has no person within 350px → returns None
    best_pid  = None
    best_dist = float("inf")
    for person in persons:
        l, t_, r, b_, pid = person
        dist = math.hypot(gcx - (l+r)/2.0, gcy - (t_+b_)/2.0)
        if dist < best_dist:
            best_dist = dist
            best_pid  = pid

    if best_pid is not None and best_dist <= max_dist:
        return best_pid

    return None   # too far → gun not drawn


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────
def model_fn(overrides: Optional[Dict] = None) -> Dict:
    cfg     = {**DEFAULTS, **(overrides or {})}
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    verbose = cfg.get("VERBOSE", True)
    log(f"Device: {device}", verbose)

    gun_models = []
    for gm in cfg["GUN_MODELS"]:
        try:
            gc = gm.copy()
            gc["model"] = YOLO(gm["path"])
            gun_models.append(gc)
            log(f"✓ YOLO: {gm['name']}", verbose)
        except Exception as e:
            log(f"✗ YOLO {gm['path']}: {e}", verbose)

    pose_model = YOLO(cfg["POSE_MODEL_PATH"])
    log("✓ Pose model", verbose)

    verifier = NonWeaponVerifier(cfg["NON_WEAPON_MODEL"], verbose)

    osnet = None
    try:
        osnet = load_osnet(device)
        log("✓ OSNet ReID", verbose)
    except Exception as e:
        log(f"⚠ OSNet: {e}", verbose)

    tracker = DeepSort(
        max_age=cfg["TRACKER_MAX_AGE"],
        n_init=cfg["TRACKER_N_INIT"],
        max_iou_distance=cfg["TRACKER_MAX_IOU_DISTANCE"],
        max_cosine_distance=cfg["TRACKER_MAX_COSINE_DISTANCE"],
        nn_budget=cfg["TRACKER_NN_BUDGET"],
        embedder=None,
    )

    gun_tracker = GunTracker(
        iou_thr=cfg["GUN_TRACKER_IOU_THRESHOLD"],
        max_lost=cfg["GUN_TRACKER_MAX_LOST_FRAMES"],
    )

    log("✓ All models loaded", verbose)

    return {
        "gun_models":  gun_models,
        "pose_model":  pose_model,
        "verifier":    verifier,
        "tracker":     tracker,
        "gun_tracker": gun_tracker,
        "id_mapper":   SequentialIDMapper(),
        "osnet":       osnet,
        "alerts":      AlertManager(cfg),
        "device":      device,
        "config":      cfg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main inference
# ─────────────────────────────────────────────────────────────────────────────
def predict_fn(input_data: Dict, model: Dict) -> Dict:
    alerts    = model["alerts"]
    id_mapper = model["id_mapper"]
    tracker   = model["tracker"]
    gun_trk   = model["gun_tracker"]
    pose      = model["pose_model"]
    verifier  = model["verifier"]
    osnet     = model["osnet"]
    device    = model["device"]
    cfg       = model["config"]

    frame        = input_data["frame"]
    cam_id       = input_data.get("cam_id", -1)
    frame_number = input_data.get("frame_number", 0)
    timestamp    = datetime.now(timezone.utc).isoformat()

    try:
        alerts.advance()
        h, w = frame.shape[:2]

        # ── Persons ───────────────────────────────────────────────────────
        p_boxes, p_scores, kpts = detect_persons(frame, pose, cfg["CONF_THR_POSE"])

        ds_dets = []; embeds = []
        for b, s in zip(p_boxes, p_scores):
            x1, y1, x2, y2 = map(int, b)
            if (x2-x1)*(y2-y1) < cfg["MIN_PERSON_AREA"]:
                continue
            ds_dets.append(([x1, y1, x2-x1, y2-y1], float(s), "person"))
            crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            embeds.append(osnet_encode(osnet, crop, device) if osnet else None)

        tracks = tracker.update_tracks(ds_dets, embeds=embeds, frame=frame)

        # Build keypoint alignment map (keyed by clean sequential ID)
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

        # ── Gun detection ensemble ────────────────────────────────────────
        raw_preds: List[Tuple[np.ndarray, float]] = []
        for m in model["gun_models"]:
            try:
                r = m["model"].predict(frame, conf=m["conf"], verbose=False)[0]
                if not len(getattr(r, "boxes", [])):
                    continue
                boxes  = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                cls    = r.boxes.cls.cpu().numpy().astype(int)
                mask   = cls == m.get("target_class_id", 0)
                for b, s in zip(boxes[mask], scores[mask]):
                    raw_preds.append((b, float(s) * m.get("weight", 1.0)))
            except Exception:
                continue

        # Agreement bonus when both models agree on same region
        fused: List[Tuple[np.ndarray, float]] = []
        for b, s in raw_preds:
            agreements = sum(
                iou(tuple(b), tuple(p[0])) > cfg["WBF_IOU_THR"]
                for p in raw_preds
            )
            bonus = cfg["AGREEMENT_BONUS"] if agreements > 1 else 0
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

        # ── Verify + size filter ──────────────────────────────────────────
        verified: List[Tuple[np.ndarray, float]] = []
        for b, s in fused:
            if s < cfg["FINAL_CONFIDENCE_THRESHOLD"]:
                continue
            # Pass frame.shape so size filter is resolution-aware
            if not valid_gun_box(b, cfg, frame.shape):
                continue
            if not verifier.verify(frame, tuple(b)):
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

            # ── KEY RULE: no holder = nothing drawn ───────────────────────
            # associate_holder returns None when:
            #   (a) no persons in frame at all, OR
            #   (b) nearest person is > MAX_HOLDER_DIST px away
            # Both cases mean this is not a held gun → skip entirely.
            # This eliminates cars, empty-room detections, floor objects.
            if holder_id is None:
                continue

            gun_dets.append({
                "gun_id":    gun_id,
                "bbox":      gun_bbox.tolist(),
                "score":     round(gun_score, 3),
                "holder_id": holder_id,
            })

            # Keep highest-confidence gun per holder (handles multi-gun)
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

        # ── DRAW ─────────────────────────────────────────────────────────
        out = frame.copy()

        # Purple holder boxes — only when gun active this frame
        for pid, info in active_holders.items():
            l, t_, r, b_ = info["bbox"]
            cv2.rectangle(out, (int(l), int(t_)), (int(r), int(b_)),
                          HOLDER_COLOR, 3)
            cv2.putText(out,
                        f"ID:{pid} [ARMED] ({info['conf']:.2f})",
                        (min(int(l), w-200), max(0, int(t_)-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, HOLDER_COLOR, 2)

        # Red gun boxes
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

        # Alert banner
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
            "cam_id": cam_id, "frame_number": frame_number,
            "status": 1, "error": str(exc),
            "guns": [], "active_holders": [],
            "new_alerts": 0, "annotated_frame": "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Video processor
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
        "total_frames":  fc,
        "total_gun_dets":total_guns,
        "unique_armed":  len(unique_armed),
        "total_alerts":  total_alerts,
    }
    log(f"Done — {fc} frames | {len(unique_armed)} armed | "
        f"{total_alerts} alerts", verbose)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary = process_video(
        video_path  = r"E:\All_models\gun_detection\output\istockphoto-1404365178-640_adpp_is.mp4",
        output_path = r"E:\All_models\gun_detection\output_clean4.mp4",
    )
    print("\n✅ Done!")
    print(f"  Frames    : {summary['total_frames']}")
    print(f"  Armed     : {summary['unique_armed']}")
    print(f"  Alerts    : {summary['total_alerts']}")