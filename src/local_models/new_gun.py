"""
inference_gun_detection_reid.py

Multi-frame pipeline for gun detection with persistent tracking using ReID.

Features:
 - Ensemble gun detection from multiple YOLO models
 - Non-weapon verification to reduce false positives
 - Person detection and tracking with DeepSort
 - OSNet ReID for persistent identity across occlusions
 - Gun holder memory system (remembers armed persons across frames)
 - Pose-based wrist detection for gun-person association
 - Smart alert system (HIGH/CRITICAL alerts only on first detection per person)

Provides:
 - model_fn(model_dir_or_config)
 - input_frame_fn(request_body, content_type="application/json")
 - predict_frame_fn(input_data, model)
 - output_frame_fn(prediction)

Input JSON example (single frame):
{
  "cam_id": 123,
  "org_id": 2,
  "user_id": 2,
  "encoding": "<base64_jpeg_data>",
  "frame_number": 0
}

Output example:
{
  "cam_id": 123,
  "frame_number": 0,
  "guns": [...],
  "gun_holders": [track_ids...],
  "persons_present": [track_ids...],
  "alerts": [{"track_id": 5, "level": "CRITICAL", "confidence": 0.92, "first_detection": true}],
  "annotated_frame": "<base64_jpeg>",
  "stats": {...},
  "status": 0
}
"""

import os
import json
import base64
import math
from datetime import datetime
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
        "Install required packages: pip install ultralytics torch torchreid deep-sort-realtime"
    ) from e

# ---------------- Defaults (editable) ----------------
DEFAULTS = {
    # Gun detection models (ensemble)
    "GUN_MODELS": [
        {
            "path": "E:\All_models\gun_detection\gun_dd_f_best.pt",
            "weight": 1.0,
            "conf": 0.70,
            "name": "gun-70k-detector",
            "classes": ["gun"],
            "target_class_id": 0
        },
        {
            "path": "E:\All_models\gun_detection\gunn2.pt",
            "weight": 0.9,
            "conf": 0.7,
            "name": "gun-knife-detector",
            "classes": ["gun", "knife"],
            "target_class_id": 0
        },
    ],
    
    # Non-weapon verification model
    "NON_WEAPON_MODEL": {
        "path": r"E:\All_models\gun_detection\non_weapons.pt",
        "conf": 0.40,
        "rejection_threshold": 0.40,
        "name": "non-weapon-verifier",
        "classes": ["Gun", "Knife", "non_weapon"]
    },
    
    # Pose detection
    "POSE_MODEL_PATH": "E:\All_models\gun_detection\yolov8x-pose.pt",
    "CONF_THR_POSE": 0.25,
    "CONF_THR_WRIST": 0.20,
    
    # Ensemble configuration
    "ENSEMBLE_METHOD": "weighted_boxes",
    "WBF_IOU_THR": 0.5,
    "AGREEMENT_BONUS": 0.25,
    "FINAL_CONFIDENCE_THRESHOLD": 0.65,
    "CRITICAL_THRESHOLD": 0.85,
    "HIGH_THRESHOLD": 0.65,
    
    # Alert system configuration
    "ALERT_COOLDOWN_FRAMES": 90,  # Don't re-alert for same person for 3 sec @ 30fps
    "ALERT_ON_FIRST_DETECTION_ONLY": True,  # Only alert when person first becomes armed
    "ALERT_ON_CONFIDENCE_INCREASE": False,  # Alert if confidence jumps (e.g., HIGH -> CRITICAL)
    "CONFIDENCE_JUMP_THRESHOLD": 0.15,  # Minimum jump to trigger re-alert
    
    # Person filtering
    "MIN_PERSON_WIDTH": 30,
    "MIN_PERSON_HEIGHT": 60,
    "MIN_PERSON_AREA": 2000,
    "NMS_IOU_POSE": 0.45,
    
    # Wrist detection
    "WRIST_HALF": 25,
    "MIN_INTERSECTION_FRAC": 0.01,
    
    # Persistent tracking configuration
    "GUN_HOLDER_MEMORY_FRAMES": 150,  # 5 sec @ 30fps
    "GUN_HOLDER_DECAY_CONFIDENCE": True,
    "CONFIDENCE_DECAY_RATE": 0.02,
    "MIN_PERSISTENT_CONFIDENCE": 0.40,
    
    # DeepSort tracker config
    "TRACKER_MAX_AGE": 30,
    "TRACKER_N_INIT": 3,
    "TRACKER_MAX_IOU_DISTANCE": 0.7,
    "TRACKER_MAX_COSINE_DISTANCE": 0.2,
    
    # General
    "VERBOSE": True,
}

# COCO keypoint indices
LEFT_WRI, RIGHT_WRI = 9, 10

# ---------------- Logging ----------------
def log(msg: str, verbose: bool = True):
    if not verbose:
        return
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)

# ---------------- Geometry helpers ----------------
def area_of_box(box: Tuple[float, float, float, float]) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def intersect_area(a: Tuple[float, float, float, float], 
                   b: Tuple[float, float, float, float]) -> float:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * \
           max(0, min(a[3], b[3]) - max(a[1], b[1]))

def iou(a: Tuple[float, float, float, float], 
        b: Tuple[float, float, float, float]) -> float:
    inter = intersect_area(a, b)
    union = area_of_box(a) + area_of_box(b) - inter
    return 0.0 if union <= 0 else inter / union

def nms_numpy(boxes: List[Tuple[float, float, float, float]], 
              scores: List[float], 
              iou_thresh: float = 0.5) -> List[int]:
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ious < iou_thresh]
    return keep

# ---------------- OSNet ReID helpers ----------------
def load_osnet(device: str):
    """Load OSNet model for person re-identification"""
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,
        pretrained=True
    )
    model.eval().to(device)
    return model

def osnet_preprocess(img: np.ndarray) -> torch.Tensor:
    """Preprocess image crop for OSNet"""
    img = cv2.resize(img, (128, 256))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

@torch.no_grad()
def osnet_encode(model, crop: np.ndarray, device: str) -> Optional[np.ndarray]:
    """Extract ReID feature vector from person crop"""
    if crop is None or crop.size == 0:
        return None
    try:
        t = osnet_preprocess(crop).to(device)
        f = model(t)
        f = F.normalize(f, p=2, dim=1)
        return f.cpu().numpy().flatten()
    except Exception:
        return None

# ---------------- Non-weapon verifier ----------------
class NonWeaponVerifier:
    """Verifies gun detections aren't actually non-weapon objects"""
    
    def __init__(self, cfg: Dict[str, Any], verbose: bool = True):
        self.cfg = cfg
        self.model = YOLO(cfg["path"])
        self.enabled = True
        if verbose:
            log(f"✓ Non-weapon verifier loaded: {cfg['name']}", verbose)
    
    def verify(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> bool:
        """
        Returns True if detection is likely a gun, False if non-weapon
        """
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
            cls = r.boxes.cls.cpu().numpy().astype(int)
            top = confs.argmax()
            
            # Class 2 = non_weapon
            is_non_weapon = (cls[top] == 2 and 
                           confs[top] > self.cfg["rejection_threshold"])
            return not is_non_weapon
        except Exception:
            return True

# ---------------- Alert Manager ----------------
class AlertManager:
    """
    Manages alert generation for armed persons.
    Prevents duplicate alerts and handles cooldown periods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_history: Dict[int, Dict[str, Any]] = {}
        self.current_frame = 0
    
    def should_alert(self, track_id: int, confidence: float, is_new_armed: bool) -> Tuple[bool, str]:
        """
        Determine if an alert should be generated for this track_id.
        
        Returns:
            (should_alert: bool, alert_level: str)
        """
        # Determine alert level based on confidence
        if confidence >= self.config["CRITICAL_THRESHOLD"]:
            alert_level = "CRITICAL"
        elif confidence >= self.config["HIGH_THRESHOLD"]:
            alert_level = "HIGH"
        else:
            return False, ""
        
        # First time seeing this person armed
        if is_new_armed:
            self.alert_history[track_id] = {
                'last_alert_frame': self.current_frame,
                'last_alert_level': alert_level,
                'last_alert_confidence': confidence,
                'total_alerts': 1
            }
            return True, alert_level
        
        # Person already in alert history
        if track_id in self.alert_history:
            history = self.alert_history[track_id]
            frames_since_alert = self.current_frame - history['last_alert_frame']
            
            # Check if we should only alert once per person
            if self.config["ALERT_ON_FIRST_DETECTION_ONLY"]:
                return False, ""
            
            # Check cooldown period
            if frames_since_alert < self.config["ALERT_COOLDOWN_FRAMES"]:
                return False, ""
            
            # Check for significant confidence increase
            if self.config["ALERT_ON_CONFIDENCE_INCREASE"]:
                conf_jump = confidence - history['last_alert_confidence']
                level_upgrade = (
                    alert_level == "CRITICAL" and 
                    history['last_alert_level'] == "HIGH"
                )
                
                if conf_jump >= self.config["CONFIDENCE_JUMP_THRESHOLD"] or level_upgrade:
                    history['last_alert_frame'] = self.current_frame
                    history['last_alert_level'] = alert_level
                    history['last_alert_confidence'] = confidence
                    history['total_alerts'] += 1
                    return True, alert_level
            
            return False, ""
        
        # Shouldn't reach here, but handle gracefully
        return False, ""
    
    def advance_frame(self):
        """Call at the start of each new frame"""
        self.current_frame += 1
    
    def reset(self):
        """Reset all alert history"""
        self.alert_history.clear()
        self.current_frame = 0
    
    def get_alert_stats(self, track_id: int) -> Dict[str, Any]:
        """Get alert statistics for a track_id"""
        if track_id in self.alert_history:
            history = self.alert_history[track_id]
            return {
                'total_alerts': history['total_alerts'],
                'last_alert_level': history['last_alert_level'],
                'frames_since_alert': self.current_frame - history['last_alert_frame']
            }
        return {'total_alerts': 0}

# ---------------- Gun holder memory ----------------
class GunHolderMemory:
    """
    Maintains persistent memory of which track_ids have been seen holding guns.
    Handles confidence decay and automatic cleanup.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gun_holders: Dict[int, Dict[str, Any]] = {}
        self.current_frame = 0
    
    def update(self, track_id: int, confidence: float, detected_now: bool = True) -> bool:
        """
        Update gun holder status for a track_id.
        
        Returns:
            is_new_armed: True if this is the first time seeing this person armed
        """
        is_new = track_id not in self.gun_holders
        
        if is_new:
            # New gun holder detected
            self.gun_holders[track_id] = {
                'last_seen_frame': self.current_frame,
                'first_seen_frame': self.current_frame,
                'confidence': confidence,
                'max_confidence': confidence,
                'detections': 1
            }
            if self.config.get("VERBOSE"):
                log(f"⚠️  NEW ARMED PERSON: Track ID {track_id} (conf: {confidence:.2f})", 
                    self.config.get("VERBOSE", True))
        else:
            # Update existing gun holder
            holder = self.gun_holders[track_id]
            holder['last_seen_frame'] = self.current_frame
            
            if detected_now:
                holder['confidence'] = max(holder['confidence'], confidence)
                holder['max_confidence'] = max(holder['max_confidence'], confidence)
                holder['detections'] += 1
        
        return is_new
    
    def decay_and_cleanup(self):
        """Decay confidence for holders not recently detected and remove stale entries"""
        to_remove = []
        
        for track_id, holder in self.gun_holders.items():
            frames_since = self.current_frame - holder['last_seen_frame']
            
            # Apply confidence decay
            if self.config["GUN_HOLDER_DECAY_CONFIDENCE"]:
                decay = frames_since * self.config["CONFIDENCE_DECAY_RATE"]
                holder['confidence'] = max(
                    self.config["MIN_PERSISTENT_CONFIDENCE"],
                    holder['confidence'] - decay
                )
            
            # Remove if too old or confidence too low
            if (frames_since > self.config["GUN_HOLDER_MEMORY_FRAMES"] or
                holder['confidence'] < self.config["MIN_PERSISTENT_CONFIDENCE"]):
                to_remove.append(track_id)
        
        for track_id in to_remove:
            holder = self.gun_holders[track_id]
            if self.config.get("VERBOSE"):
                log(f"✓ Cleared armed status: ID {track_id} "
                    f"(frames armed: {holder['last_seen_frame'] - holder['first_seen_frame']}, "
                    f"detections: {holder['detections']})", 
                    self.config.get("VERBOSE", True))
            del self.gun_holders[track_id]
    
    def is_armed(self, track_id: int) -> bool:
        """Check if a track_id is currently considered armed"""
        return track_id in self.gun_holders
    
    def get_status(self, track_id: int) -> Dict[str, Any]:
        """Get detailed status for a track_id"""
        if track_id in self.gun_holders:
            holder = self.gun_holders[track_id]
            return {
                'armed': True,
                'confidence': holder['confidence'],
                'max_confidence': holder['max_confidence'],
                'frames_since_detection': self.current_frame - holder['last_seen_frame'],
                'total_detections': holder['detections'],
                'frames_tracked': self.current_frame - holder['first_seen_frame']
            }
        return {'armed': False}
    
    def get_all_armed(self) -> List[int]:
        """Get all currently armed track_ids"""
        return list(self.gun_holders.keys())
    
    def advance_frame(self):
        """Call at the start of each new frame"""
        self.current_frame += 1
        self.decay_and_cleanup()
    
    def reset(self):
        """Reset all memory (useful for new video/stream)"""
        self.gun_holders.clear()
        self.current_frame = 0

# ---------------- Ensemble detection ----------------
def run_gun_ensemble(frame: np.ndarray, 
                     gun_models: List[Dict[str, Any]]) -> List[Tuple[np.ndarray, float]]:
    """Run ensemble gun detection across multiple models"""
    preds = []
    for m in gun_models:
        try:
            r = m["model"].predict(frame, conf=m["conf"], verbose=False)[0]
            if len(r.boxes) == 0:
                continue
            
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            
            # Filter for target class (gun)
            mask = cls == m["target_class_id"]
            for b, s in zip(boxes[mask], scores[mask]):
                preds.append((b, s * m["weight"]))
        except Exception:
            continue
    
    return preds

def weighted_boxes_fusion(preds: List[Tuple[np.ndarray, float]], 
                          config: Dict[str, Any]) -> List[Tuple[np.ndarray, float]]:
    """Apply weighted boxes fusion with agreement bonus"""
    fused = []
    for i, (b, s) in enumerate(preds):
        # Count how many other boxes agree with this one
        agreements = sum(
            iou(tuple(b), tuple(p[0])) > config["WBF_IOU_THR"] 
            for p in preds
        )
        
        # Boost confidence if multiple models agree
        bonus = config["AGREEMENT_BONUS"] if agreements > 1 else 0
        final_score = min(1.0, s + bonus)
        fused.append((b, final_score))
    
    return fused

# ---------------- Pose detection ----------------
def detect_persons(frame: np.ndarray, 
                   pose_model, 
                   conf_thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run pose detection and extract person boxes, scores, and keypoints
    """
    r = pose_model.predict(frame, conf=conf_thr, verbose=False)[0]
    
    boxes = np.zeros((0, 4))
    scores = np.zeros(0)
    kpts = np.zeros((0, 17, 3))
    
    if len(r.boxes):
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        try:
            kpts = r.keypoints.data.cpu().numpy()
        except Exception:
            try:
                kpts = r.keypoints.cpu().numpy()
            except Exception:
                kpts = np.zeros((len(boxes), 17, 3))
    
    return boxes, scores, kpts

# ---------------- Model loader ----------------
def model_fn(model_dir_or_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load and return model dict with gun models, pose model, tracker, etc.
    Accepts a dict of overrides or None to use DEFAULTS.
    """
    cfg = DEFAULTS.copy()
    if model_dir_or_config:
        if isinstance(model_dir_or_config, dict):
            cfg.update(model_dir_or_config)
        else:
            # If string path provided, update model paths
            model_dir = str(model_dir_or_config)
            for gm in cfg["GUN_MODELS"]:
                gm["path"] = os.path.join(model_dir, os.path.basename(gm["path"]))
            cfg["NON_WEAPON_MODEL"]["path"] = os.path.join(
                model_dir, 
                os.path.basename(cfg["NON_WEAPON_MODEL"]["path"])
            )
            cfg["POSE_MODEL_PATH"] = os.path.join(
                model_dir, 
                os.path.basename(cfg["POSE_MODEL_PATH"])
            )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    verbose = cfg.get("VERBOSE", True)
    
    if verbose:
        log(f"Using device: {device}", verbose)
    
    # Load gun detection models
    gun_models = []
    for gm_cfg in cfg["GUN_MODELS"]:
        try:
            m = YOLO(gm_cfg["path"])
            gm_copy = gm_cfg.copy()
            gm_copy["model"] = m
            gun_models.append(gm_copy)
            if verbose:
                log(f"✓ Gun model loaded: {gm_cfg['name']}", verbose)
        except Exception as e:
            if verbose:
                log(f"✗ Failed to load gun model {gm_cfg['path']}: {e}", verbose)
    
    # Load non-weapon verifier
    verifier = NonWeaponVerifier(cfg["NON_WEAPON_MODEL"], verbose)
    
    # Load pose model
    pose_model = YOLO(cfg["POSE_MODEL_PATH"])
    if verbose:
        log("✓ Pose model loaded", verbose)
    
    # Load OSNet for ReID
    osnet = None
    try:
        osnet = load_osnet(device)
        if verbose:
            log("✓ OSNet ReID loaded - persistent tracking enabled", verbose)
    except Exception as e:
        if verbose:
            log(f"⚠ OSNet failed to load: {e}. Continuing without ReID.", verbose)
    
    # Initialize DeepSort tracker
    tracker = DeepSort(
        max_age=cfg["TRACKER_MAX_AGE"],
        n_init=cfg["TRACKER_N_INIT"],
        max_iou_distance=cfg["TRACKER_MAX_IOU_DISTANCE"],
        embedder=None,  # We provide embeddings directly
        max_cosine_distance=cfg["TRACKER_MAX_COSINE_DISTANCE"]
    )
    if verbose:
        log("✓ DeepSort tracker initialized", verbose)
    
    # Initialize gun holder memory
    gun_holder_memory = GunHolderMemory(cfg)
    if verbose:
        log("✓ Gun holder memory system initialized", verbose)
    
    # Initialize alert manager
    alert_manager = AlertManager(cfg)
    if verbose:
        log("✓ Smart alert system initialized", verbose)
        if cfg["ALERT_ON_FIRST_DETECTION_ONLY"]:
            log("  - Alert mode: First detection only", verbose)
        else:
            cooldown_sec = cfg["ALERT_COOLDOWN_FRAMES"] / 30.0
            log(f"  - Alert cooldown: {cooldown_sec:.1f} seconds", verbose)
    
    return {
        "gun_models": gun_models,
        "verifier": verifier,
        "pose_model": pose_model,
        "tracker": tracker,
        "osnet": osnet,
        "gun_holder_memory": gun_holder_memory,
        "alert_manager": alert_manager,
        "device": device,
        "config": cfg
    }

# ---------------- Single-frame handlers ----------------
def input_frame_fn(request_body: Any, content_type: str = "application/json") -> Dict[str, Any]:
    """
    Parse input frame from JSON payload with base64 JPEG
    """
    if content_type != "application/json":
        raise ValueError("Expected application/json with base64 JPEG in 'encoding' field")
    
    if isinstance(request_body, str):
        payload = json.loads(request_body)
    else:
        payload = request_body
    
    cam_id = payload.get("cam_id", -1)
    org_id = payload.get("org_id", None)
    user_id = payload.get("user_id", None)
    frame_number = payload.get("frame_number", 0)
    
    b64 = payload.get("encoding") or payload.get("image")
    if not b64:
        raise ValueError("Payload must include 'encoding' with base64 jpeg data")
    
    if isinstance(b64, str) and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    
    try:
        decoded = base64.b64decode(b64)
        arr = np.frombuffer(decoded, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}")
    
    return {
        "cam_id": cam_id,
        "org_id": org_id,
        "user_id": user_id,
        "frame_number": frame_number,
        "frame": img,
        "raw_payload": payload
    }

def predict_frame_fn(input_data: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run single-frame inference with persistent tracking and smart alerts
    """
    pose_model = model["pose_model"]
    tracker = model["tracker"]
    verifier = model["verifier"]
    osnet = model["osnet"]
    gun_holder_memory = model["gun_holder_memory"]
    alert_manager = model["alert_manager"]
    device = model["device"]
    cfg = model["config"]
    
    frame = input_data["frame"]
    cam_id = input_data.get("cam_id", -1)
    frame_number = input_data.get("frame_number", 0)
    
    try:
        h, w = frame.shape[:2]
        
        # Advance frame counters
        gun_holder_memory.advance_frame()
        alert_manager.advance_frame()
        
        # ---- Pose detection ----
        p_boxes, p_scores, kpts = detect_persons(
            frame, pose_model, cfg["CONF_THR_POSE"]
        )
        
        # Prepare detections for DeepSort with OSNet embeddings
        detections_ds = []
        embeddings = []
        
        for b, s in zip(p_boxes, p_scores):
            x1, y1, x2, y2 = map(int, b)
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue
            if (x2 - x1) * (y2 - y1) < cfg["MIN_PERSON_AREA"]:
                continue
            
            # DeepSort format: [x, y, w, h]
            detections_ds.append(([x1, y1, x2 - x1, y2 - y1], float(s), "person"))
            
            # Extract ReID embedding
            crop = frame[y1:y2, x1:x2]
            emb = osnet_encode(osnet, crop, device) if osnet is not None else None
            embeddings.append(emb)
        
        # Update tracker with embeddings
        tracks = tracker.update_tracks(
            detections_ds,
            embeds=embeddings,
            frame=frame
        )
        
        # Align keypoints to tracked persons
        persons = []
        pose_boxes = p_boxes if len(p_boxes) > 0 else np.zeros((0, 4))
        aligned_kpts_for_tracks = {}
        
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            
            l_t, t_t, r_t, b_t = map(float, t.to_ltrb())
            track_box = np.array([l_t, t_t, r_t, b_t], dtype=float)
            
            # Find best matching pose box
            best_i, best_iou = -1, 0.0
            for i, pb in enumerate(pose_boxes):
                iou_val = iou(tuple(track_box), tuple(pb))
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_i = i
            
            if best_i >= 0 and best_iou >= 0.3:
                aligned_kpts_for_tracks[t.track_id] = (
                    kpts[best_i] if len(kpts) > best_i else None
                )
            else:
                aligned_kpts_for_tracks[t.track_id] = None
        
        # Build person list with track IDs
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            
            l, t_, r, b = map(int, t.to_ltrb())
            w_track, h_track = r - l, b - t_
            
            if (w_track < cfg["MIN_PERSON_WIDTH"] or 
                h_track < cfg["MIN_PERSON_HEIGHT"]):
                continue
            
            persons.append([l, t_, r, b, t.track_id])
        
        aligned_kpts_list = [
            aligned_kpts_for_tracks.get(p[4], None) for p in persons
        ]
        
        # ---- Gun ensemble detection ----
        preds = run_gun_ensemble(frame, model["gun_models"])
        fused = weighted_boxes_fusion(preds, cfg)
        
        current_frame_detections = []
        rejected = []
        verified = []
        alerts_to_send = []
        
        for box, score in fused:
            if score < 0:
                continue
            
            # Non-weapon verification
            is_gun = verifier.verify(frame, tuple(box))
            if not is_gun:
                rejected.append({"bbox": box.tolist(), "score": float(score)})
                continue
            
            verified.append({"bbox": box.tolist(), "score": float(score)})
            
            # Associate gun with person via wrist proximity
            holder_id = None
            ga = area_of_box(tuple(box))
            min_dist = float("inf")
            
            for idx, person in enumerate(persons):
                l, t_, r, b, pid = person
                kp = aligned_kpts_list[idx] if idx < len(aligned_kpts_list) else None
                
                if kp is None:
                    continue
                
                # Check both wrists
                for wi in (LEFT_WRI, RIGHT_WRI):
                    if kp.shape[0] <= wi or kp[wi][2] < cfg["CONF_THR_WRIST"]:
                        continue
                    
                    wx, wy = float(kp[wi][0]), float(kp[wi][1])
                    wb = [
                        wx - cfg["WRIST_HALF"], 
                        wy - cfg["WRIST_HALF"],
                        wx + cfg["WRIST_HALF"], 
                        wy + cfg["WRIST_HALF"]
                    ]
                    
                    if ga <= 0:
                        continue
                    
                    inter_frac = intersect_area(tuple(box), tuple(wb)) / max(1, ga)
                    if inter_frac < cfg["MIN_INTERSECTION_FRAC"]:
                        continue
                    
                    # Calculate distance from gun to person center
                    cx, cy = (l + r) / 2.0, (t_ + b) / 2.0
                    dist = math.hypot(cx - wx, cy - wy)
                    
                    if dist < min_dist:
                        min_dist = dist
                        holder_id = pid
            
            if holder_id is None:
                continue
            
            # Final confidence check
            if score >= cfg["FINAL_CONFIDENCE_THRESHOLD"]:
                # Update persistent memory and check if this is new
                is_new_armed = gun_holder_memory.update(holder_id, score, detected_now=True)
                
                # Determine alert level
                alert_level = (
                    "CRITICAL" if score >= cfg["CRITICAL_THRESHOLD"] else "HIGH"
                )
                
                # Check if we should send an alert
                should_alert, final_level = alert_manager.should_alert(
                    holder_id, score, is_new_armed
                )
                
                current_frame_detections.append({
                    "track_id": holder_id,
                    "bbox": [int(x) for x in box],
                    "score": float(score),
                    "alert": should_alert,
                    "alert_level": final_level if should_alert else alert_level,
                    "is_new_detection": is_new_armed
                })
                
                # Add to alerts list if needed
                if should_alert:
                    alerts_to_send.append({
                        "track_id": holder_id,
                        "level": final_level,
                        "confidence": float(score),
                        "bbox": [int(x) for x in box],
                        "first_detection": is_new_armed,
                        "frame_number": frame_number
                    })
        
        # Get all currently armed persons (including memory)
        all_armed_ids = gun_holder_memory.get_all_armed()
        
        # ---- Prepare annotated frame ----
        out_frame = frame.copy()
        
        # Draw rejected detections
        for rej in rejected:
            x1, y1, x2, y2 = map(int, rej["bbox"])
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
            cv2.putText(
                out_frame, "REJECTED", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1
            )
        
        # Draw gun detections from current frame
        for det in current_frame_detections:
            gb = det["bbox"]
            color = (0, 0, 255)  # Red for guns
            
            # Add alert indicator if this generated an alert
            if det["alert"]:
                if det["alert_level"] == "CRITICAL":
                    color = (0, 0, 255)  # Red
                    label = f"⚠ CRITICAL {det['score']:.2f}"
                else:
                    color = (0, 165, 255)  # Orange
                    label = f"⚠ HIGH {det['score']:.2f}"
            else:
                label = f"{det['score']:.2f}"
            
            cv2.rectangle(out_frame, (gb[0], gb[1]), (gb[2], gb[3]), color, 2)
            cv2.putText(
                out_frame, label, (gb[0], gb[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # Draw persons with armed status
        for idx, (l, t_, r, b, pid) in enumerate(persons):
            status = gun_holder_memory.get_status(pid)
            has_gun = status['armed']
            
            # Color coding
            if has_gun:
                detected_this_frame = any(
                    d["track_id"] == pid for d in current_frame_detections
                )
                alerted_this_frame = any(
                    a["track_id"] == pid for a in alerts_to_send
                )
                
                if alerted_this_frame:
                    color = (0, 0, 255)  # Red - new alert
                    label_suffix = " [🚨 ALERT]"
                elif detected_this_frame:
                    color = (255, 0, 255)  # Magenta - currently holding
                    label_suffix = " [ARMED]"
                else:
                    color = (0, 165, 255)  # Orange - armed from memory
                    frames_since = status['frames_since_detection']
                    label_suffix = f" [MEM {frames_since}f]"
            else:
                color = (0, 255, 0)  # Green - not armed
                label_suffix = ""
            
            thickness = 3 if has_gun else 2
            cv2.rectangle(out_frame, (l, t_), (r, b), color, thickness)
            
            label = f"ID:{pid}{label_suffix}"
            if has_gun:
                label += f" ({status['confidence']:.2f})"
            
            cv2.putText(
                out_frame, label, (l, t_ - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Draw wrists
            aligned_kpt = (
                aligned_kpts_list[idx] if idx < len(aligned_kpts_list) else None
            )
            if aligned_kpt is not None:
                for wi in (LEFT_WRI, RIGHT_WRI):
                    if (aligned_kpt.shape[0] > wi and 
                        aligned_kpt[wi][2] >= cfg["CONF_THR_WRIST"]):
                        wx, wy = int(aligned_kpt[wi][0]), int(aligned_kpt[wi][1])
                        cv2.circle(out_frame, (wx, wy), 3, (0, 255, 255), -1)
        
        # Add alert banner if there are alerts
        if alerts_to_send:
            alert_text = f"⚠ {len(alerts_to_send)} NEW ALERT(S)"
            cv2.rectangle(out_frame, (0, 0), (w, 40), (0, 0, 255), -1)
            cv2.putText(
                out_frame, alert_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
            )
        
        # Encode annotated frame to base64 JPEG
        ok, jpg = cv2.imencode('.jpg', out_frame)
        if not ok:
            raise RuntimeError("Failed to encode annotated frame to JPEG")
        b64_out = base64.b64encode(jpg.tobytes()).decode('utf-8')
        
        # Build output
        stats = {
            "ensemble_detections": len(fused),
            "verified": len(verified),
            "rejected_non_weapon": len(rejected),
            "current_frame_detections": len(current_frame_detections),
            "persistent_armed_persons": len(all_armed_ids),
            "armed_ids": all_armed_ids,
            "alerts_sent": len(alerts_to_send)
        }
        
        guns_out = []
        for det in current_frame_detections:
            guns_out.append({
                "track_id": det["track_id"],
                "bbox": det["bbox"],
                "score": det["score"],
                "alert": det["alert"],
                "alert_level": det.get("alert_level", ""),
                "is_new_detection": det["is_new_detection"]
            })
        
        return {
            "cam_id": cam_id,
            "frame_number": frame_number,
            "guns": guns_out,
            "gun_holders": all_armed_ids,
            "persons_present": [p[4] for p in persons],
            "alerts": alerts_to_send,
            "annotated_frame": b64_out,
            "stats": stats,
            "status": 0
        }
    
    except Exception as e:
        return {
            "cam_id": cam_id,
            "frame_number": frame_number,
            "guns": [],
            "gun_holders": [],
            "persons_present": [],
            "alerts": [],
            "annotated_frame": "",
            "stats": {},
            "status": 1,
            "error": str(e)
        }

def output_frame_fn(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final formatting for client response (strips debug info)
    """
    return {
        "cam_id": prediction.get("cam_id", -1),
        "frame_number": prediction.get("frame_number", 0),
        "guns": prediction.get("guns", []),
        "gun_holders": prediction.get("gun_holders", []),
        "persons_present": prediction.get("persons_present", []),
        "alerts": prediction.get("alerts", []),
        "annotated_frame": prediction.get("annotated_frame", ""),
        "stats": prediction.get("stats", {}),
        "status": prediction.get("status", 0)
    }

# ---------------- Video processing helper ----------------
def process_video(video_path: str, 
                  output_path: str = "output.mp4",
                  max_frames: Optional[int] = None,
                  config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process entire video file with persistent tracking and smart alerts.
    Returns summary statistics.
    """
    models = model_fn(config_overrides)
    cfg = models["config"]
    verbose = cfg.get("VERBOSE", True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    if verbose:
        log(f"Processing video: {total_frames} frames @ {fps:.2f} FPS", verbose)
        memory_duration = cfg["GUN_HOLDER_MEMORY_FRAMES"] / fps
        log(f"Gun holder memory: {cfg['GUN_HOLDER_MEMORY_FRAMES']} frames "
            f"({memory_duration:.1f} seconds)", verbose)
        
        if cfg["ALERT_ON_FIRST_DETECTION_ONLY"]:
            log("Alert mode: First detection only per person", verbose)
        else:
            cooldown_sec = cfg["ALERT_COOLDOWN_FRAMES"] / fps
            log(f"Alert cooldown: {cooldown_sec:.1f} seconds", verbose)
    
    writer = None
    frame_count = 0
    summary = {
        "total_frames": 0,
        "total_gun_detections": 0,
        "unique_armed_persons": set(),
        "total_alerts": 0,
        "alerts_by_level": {"HIGH": 0, "CRITICAL": 0}
    }
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if max_frames and frame_count > max_frames:
            break
        
        # Process frame
        input_data = {
            "cam_id": 0,
            "frame_number": frame_count,
            "frame": frame,
            "raw_payload": {}
        }
        
        result = predict_frame_fn(input_data, models)
        
        # Initialize writer
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {output_path}")
            if verbose:
                log(f"Output: {output_path} ({w}x{h} @ {fps:.2f} FPS)", verbose)
        
        # Decode and write annotated frame
        b64_frame = result["annotated_frame"]
        decoded = base64.b64decode(b64_frame)
        arr = np.frombuffer(decoded, dtype=np.uint8)
        annotated = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        writer.write(annotated)
        
        # Update summary
        summary["total_frames"] += 1
        summary["total_gun_detections"] += len(result["guns"])
        summary["unique_armed_persons"].update(result["gun_holders"])
        summary["total_alerts"] += len(result["alerts"])
        
        for alert in result["alerts"]:
            level = alert.get("level", "HIGH")
            summary["alerts_by_level"][level] = summary["alerts_by_level"].get(level, 0) + 1
        
        # Progress logging
        if verbose and frame_count % 30 == 0:
            log(f"Frame {frame_count}/{total_frames} | "
                f"Armed: {len(result['gun_holders'])} | "
                f"Detections: {len(result['guns'])} | "
                f"Alerts: {len(result['alerts'])}", verbose)
    
    cap.release()
    if writer is not None:
        writer.release()
    
    summary["unique_armed_persons"] = len(summary["unique_armed_persons"])
    
    if verbose:
        log(f"✓ Complete! {summary['total_frames']} frames processed", verbose)
        log(f"✓ Output saved: {output_path}", verbose)
        log(f"✓ Total gun detections: {summary['total_gun_detections']}", verbose)
        log(f"✓ Unique armed persons: {summary['unique_armed_persons']}", verbose)
        log(f"✓ Total alerts sent: {summary['total_alerts']}", verbose)
        log(f"  - HIGH alerts: {summary['alerts_by_level']['HIGH']}", verbose)
        log(f"  - CRITICAL alerts: {summary['alerts_by_level']['CRITICAL']}", verbose)
    
    return summary

# ---------------- Main ----------------
if __name__ == "__main__":
    # Process your video with smart alerts
    summary = process_video(
        video_path='E:\All_models\gun_detection\guntest1.mp4',
        output_path='E:\All_models\gun_detection\output_annotated.mp4',
        max_frames=None   # Process all frames
    )

    print("\n✅ Done!")
    print(f"Frames: {summary['total_frames']}")
    print(f"Detections: {summary['total_gun_detections']}")
    print(f"Armed persons: {summary['unique_armed_persons']}")
    print(f"Total alerts: {summary['total_alerts']}")
    print(f"  HIGH: {summary['alerts_by_level']['HIGH']}")
    print(f"  CRITICAL: {summary['alerts_by_level']['CRITICAL']}")
"""    
Updated entry point for gun detection with persistent tracking and smart alerts.
Compatible with inference_gun_detection_reid.py
"""

import cv2
import base64
import numpy as np
from typing import Dict, Any

# Import the functions from your main inference file
from inference_gun_detection_reid import (
    model_fn,
    input_frame_fn,
    predict_frame_fn,
    output_frame_fn
)

# ===================== CONFIGURATION =====================
VIDEO_PATH = r"E:\All_models\gun_detection\guntest1.mp4"

# Optional: Override default settings
CONFIG_OVERRIDES = {
    "VERBOSE": True,
    "ALERT_ON_FIRST_DETECTION_ONLY": True,  # Only alert on first detection per person
    "GUN_HOLDER_MEMORY_FRAMES": 150,  # 5 seconds @ 30fps
    "CRITICAL_THRESHOLD": 0.85,
    "HIGH_THRESHOLD": 0.65,
}

# ===================== HELPER FUNCTIONS =====================
def _decode_b64_image(b64_str: str) -> np.ndarray:
    """Decode base64 string to OpenCV image"""
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    decoded = base64.b64decode(b64_str)
    arr = np.frombuffer(decoded, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _print_frame_summary(result: Dict[str, Any]):
    """Print concise summary of detection results"""
    frame_num = result.get("frame_number", 0)
    guns = result.get("guns", [])
    gun_holders = result.get("gun_holders", [])
    alerts = result.get("alerts", [])
    stats = result.get("stats", {})
    
    # Print frame info
    print(f"\rFrame {frame_num:4d} | ", end="")
    print(f"Armed: {len(gun_holders):2d} | ", end="")
    print(f"Detections: {len(guns):2d} | ", end="")
    print(f"Alerts: {len(alerts):2d}", end="")
    
    # Print alert details if any
    if alerts:
        print()  # New line for alerts
        for alert in alerts:
            level = alert.get("level", "HIGH")
            track_id = alert.get("track_id", -1)
            confidence = alert.get("confidence", 0.0)
            is_first = alert.get("first_detection", False)
            
            icon = "🚨" if level == "CRITICAL" else "⚠️"
            status = "NEW" if is_first else "UPDATE"
            print(f"  {icon} {level} ALERT - ID:{track_id} ({confidence:.2f}) [{status}]")
    
    # Flush output
    print("", end="", flush=True)

# ===================== ENTRY POINTS =====================

# Initialize model once (persistent across frames for tracking)
_MODEL = model_fn(CONFIG_OVERRIDES)
_FRAME_COUNTER = 0

def run_inference(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process single frame with persistent tracking.
    
    Args:
        payload: Dict with keys:
            - encoding: base64 encoded JPEG
            - cam_id: camera ID (optional)
            - org_id: organization ID (optional)
            - user_id: user ID (optional)
            - frame_number: frame number for tracking (auto-assigned if missing)
    
    Returns:
        Dict with detection results including alerts
    """
    global _FRAME_COUNTER
    
    # Assign frame number if not provided
    if "frame_number" not in payload:
        payload["frame_number"] = _FRAME_COUNTER
        _FRAME_COUNTER += 1
    
    # Parse input
    input_data = input_frame_fn(payload, content_type="application/json")
    
    # Run detection
    prediction = predict_frame_fn(input_data, _MODEL)
    
    # Format output
    return output_frame_fn(prediction)

def live_inference(video_path: str, show_stats: bool = True):
    """
    Process video stream with persistent tracking and display results.
    
    Args:
        video_path: Path to video file or camera index (0 for webcam)
        show_stats: Print frame statistics to console
    """
    global _FRAME_COUNTER
    _FRAME_COUNTER = 0  # Reset counter for new video
    
    # Reset model state for new video
    _MODEL["gun_holder_memory"].reset()
    _MODEL["alert_manager"].reset()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎥 Processing video: {total_frames} frames @ {fps:.1f} FPS")
    print(f"📊 Alert mode: {'First detection only' if CONFIG_OVERRIDES.get('ALERT_ON_FIRST_DETECTION_ONLY') else 'With cooldown'}")
    print(f"💾 Memory duration: {CONFIG_OVERRIDES.get('GUN_HOLDER_MEMORY_FRAMES', 150) / fps:.1f} seconds")
    print("\nPress 'q' to quit, 'p' to pause/resume\n")
    
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Encode frame
            _, buf = cv2.imencode(".jpg", frame)
            b64_frame = base64.b64encode(buf).decode()
            
            # Build payload
            payload = {
                "cam_id": 1,
                "org_id": 1,
                "user_id": 1,
                "encoding": b64_frame,
                "frame_number": _FRAME_COUNTER
            }
            
            # Run inference
            result = run_inference(payload)
            
            # Print stats if enabled
            if show_stats:
                _print_frame_summary(result)
            
            # Decode annotated frame
            img = _decode_b64_image(result["annotated_frame"])
            
            # Add overlay with persistent tracking info
            h, w = img.shape[:2]
            overlay = img.copy()
            
            # Info panel
            stats = result.get("stats", {})
            armed_count = stats.get("persistent_armed_persons", 0)
            armed_ids = stats.get("armed_ids", [])
            
            info_lines = [
                f"Frame: {_FRAME_COUNTER}/{total_frames}",
                f"Armed: {armed_count} persons",
                f"IDs: {armed_ids}" if armed_ids else "IDs: []"
            ]
            
            y_offset = 60
            for line in info_lines:
                cv2.putText(overlay, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            # Blend overlay
            img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Display
        cv2.imshow("Gun Detection - Persistent Tracking", img)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print(f"\n{'⏸️  PAUSED' if paused else '▶️  RESUMED'}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final summary
    print("\n\n" + "="*60)
    print("📊 SESSION SUMMARY")
    print("="*60)
    gun_memory = _MODEL["gun_holder_memory"]
    all_armed = gun_memory.get_all_armed()
    
    print(f"Total frames processed: {_FRAME_COUNTER}")
    print(f"Currently armed persons: {len(all_armed)}")
    
    if all_armed:
        print("\nArmed person details:")
        for track_id in all_armed:
            status = gun_memory.get_status(track_id)
            print(f"  ID {track_id:3d}: conf={status['confidence']:.2f}, "
                  f"detections={status['total_detections']}, "
                  f"tracked={status['frames_tracked']} frames")
    
    print("="*60)

def batch_inference(video_path: str, output_path: str = None):
    """
    Process entire video and save annotated output.
    
    Args:
        video_path: Input video path
        output_path: Output video path (optional)
    """
    from inference_gun_detection_reid import process_video
    
    if output_path is None:
        output_path = video_path.replace(".mp4", "_annotated.mp4")
    
    summary = process_video(
        video_path=video_path,
        output_path=output_path,
        config_overrides=CONFIG_OVERRIDES
    )
    
    print("\n" + "="*60)
    print("✅ BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Input:  {video_path}")
    print(f"Output: {output_path}")
    print(f"\nFrames processed: {summary['total_frames']}")
    print(f"Gun detections: {summary['total_gun_detections']}")
    print(f"Unique armed persons: {summary['unique_armed_persons']}")
    print(f"Total alerts: {summary['total_alerts']}")
    print(f"  - HIGH: {summary['alerts_by_level']['HIGH']}")
    print(f"  - CRITICAL: {summary['alerts_by_level']['CRITICAL']}")
    print("="*60)

# ===================== MAIN =====================
if __name__ == "__main__":
    # Choose processing mode:
    
    # Option 1: Live inference with display (recommended for testing)
    live_inference(VIDEO_PATH, show_stats=True)
    
    # Option 2: Batch processing (faster, no display)
    # batch_inference(
    #     video_path=VIDEO_PATH,
    #     output_path=r"E:\All_models\gun_detection\output_persistent.mp4"
    # )by_level']['CRITICAL']}")