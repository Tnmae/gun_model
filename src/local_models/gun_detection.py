"""
gun_detection.py - Production-ready with per-camera persistent tracking

Drop-in replacement for existing gun_detection.py
Works with existing websocket without any changes!

Place this in: src/local_models/gun_detection.py
"""

import cv2
import base64
import numpy as np
import sys
import os
from typing import Dict, Any
from threading import Lock
from datetime import datetime

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try relative import first (for uvicorn), then absolute (for standalone)
try:
    from .inference_gun_detection_reid import (
        model_fn,
        input_frame_fn,
        predict_frame_fn,
        output_frame_fn
    )
except ImportError:
    from inference_gun_detection_reid import (
        model_fn,
        input_frame_fn,
        predict_frame_fn,
        output_frame_fn
    )

# ===================== CONFIGURATION =====================
CONFIG_OVERRIDES = {
    "VERBOSE": False,  # Set False for production (less console spam)

    # Alert behavior
    "ALERT_ON_FIRST_DETECTION_ONLY": True,  # Only alert once per person
    "ALERT_COOLDOWN_FRAMES": 90,            # 3 seconds @ 30fps (if not first-only)

    # Detection thresholds
    "CRITICAL_THRESHOLD": 0.85,             # Red/Critical alerts
    "HIGH_THRESHOLD": 0.65,                 # Orange/High alerts
    "FINAL_CONFIDENCE_THRESHOLD": 0.65,

    # Model confidence
    "CONF_THR_POSE": 0.25,
    "CONF_THR_WRIST": 0.20,

    # Parallel inference workers (pose + gun run concurrently, OSNet disabled)
    "INFERENCE_WORKERS": 2,

    # Gun model frame-skip: 1 = disabled (every frame)
    "GUN_SKIP_FRAMES": 1,

    # Gun inference input size (480 = ~30% faster than 640, dynamic engine)
    "GUN_INFER_IMGSZ": 480,

    # OSNet ReID disabled — saves ~35ms/frame, IoU-only DeepSort tracking
    "USE_OSNET": False,
}

# ===================== PER-CAMERA TRACKING MANAGER =====================
class CameraTracker:
    """
    Thread-safe manager for per-camera tracking state.
    Each camera maintains its own model instance and frame counter.
    """
    
    def __init__(self):
        self.cameras: Dict[int, Dict[str, Any]] = {}
        self.lock = Lock()
    
    def get_or_create(self, cam_id: int) -> Dict[str, Any]:
        """Get or create tracking state for a camera"""
        with self.lock:
            if cam_id not in self.cameras:
                # Create new model instance for this camera
                print(f"[CameraTracker] Initializing camera {cam_id}...")
                model = model_fn(CONFIG_OVERRIDES)
                self.cameras[cam_id] = {
                    "model": model,
                    "frame_counter": 0,
                    "initialized": True
                }
                print(f"[CameraTracker] ✓ Camera {cam_id} ready")
            return self.cameras[cam_id]
    
    def increment_frame(self, cam_id: int) -> int:
        """Increment and return frame counter for a camera"""
        with self.lock:
            if cam_id in self.cameras:
                self.cameras[cam_id]["frame_counter"] += 1
                return self.cameras[cam_id]["frame_counter"]
            return 0
    
    def reset(self, cam_id: int):
        """Reset tracking for a specific camera"""
        with self.lock:
            if cam_id in self.cameras:
                camera_state = self.cameras[cam_id]
                model = camera_state["model"]
                # Reset alert manager and gun tracker (correct model dict keys)
                model["alerts"].reset()
                model["gun_tracker"].reset()
                model["id_mapper"].reset()
                camera_state["frame_counter"] = 0
                print(f"[CameraTracker] ✓ Reset tracking for camera {cam_id}")
    
    def remove(self, cam_id: int):
        """Remove camera tracking state (cleanup)"""
        with self.lock:
            if cam_id in self.cameras:
                del self.cameras[cam_id]
                print(f"[CameraTracker] ✓ Removed camera {cam_id}")
    
    def get_stats(self, cam_id: int) -> Dict[str, Any]:
        """Get tracking statistics for a camera"""
        with self.lock:
            if cam_id not in self.cameras:
                return {"error": "Camera not found", "cam_id": cam_id}
            
            camera_state = self.cameras[cam_id]
            model = camera_state["model"]
            # Use alert manager history as proxy for armed track IDs
            alert_history = model["alerts"].history
            armed_ids = list(alert_history.keys())
            
            return {
                "cam_id": cam_id,
                "total_frames": camera_state["frame_counter"],
                "armed_count": len(armed_ids),
                "armed_ids": armed_ids
            }

# Global camera tracker instance
_CAMERA_TRACKER = CameraTracker()

# ===================== MAIN INFERENCE FUNCTION =====================
def run_inference(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main inference function — accepts a payload with a base64-encoded frame.
    Used by the standalone test harness and any caller that already has base64.

    Payload keys:
        encoding     – base64 JPEG string (required)
        cam_id       – int (optional, default 0)
        org_id       – int (optional)
        user_id      – int (optional)
    """
    try:
        cam_id = payload.get("cam_id", 0)
        camera_state = _CAMERA_TRACKER.get_or_create(cam_id)
        model = camera_state["model"]
        current_frame = _CAMERA_TRACKER.increment_frame(cam_id)
        payload["frame_number"] = current_frame

        input_data = input_frame_fn(payload, content_type="application/json")
        prediction = predict_frame_fn(input_data, model)
        return output_frame_fn(prediction)

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"[ERROR] Camera {payload.get('cam_id', -1)}: {error_msg}")
        print(traceback.format_exc())
        timestamp = datetime.utcnow().isoformat() + 'Z'
        return {
            "cam_id": payload.get("cam_id", -1),
            "org_id": payload.get("org_id", -1),
            "user_id": payload.get("user_id", -1),
            "frame_number": 0,
            "timestamp": timestamp,
            "guns": [],
            "gun_holders": [],
            "persons_present": [],
            "alerts": [],
            "annotated_frame": "",
            "stats": {},
            "status": 1,
            "error": error_msg
        }


def run_inference_raw(frame: np.ndarray, cam_id: int, org_id: int = -1, user_id: int = -1) -> Dict[str, Any]:
    """
    Inference entry point for callers that already have a decoded numpy frame
    (e.g. the websocket loop). Skips the base64 encode/decode round-trip.

    Args:
        frame:   BGR numpy array from cv2.VideoCapture
        cam_id:  camera identifier
        org_id:  organisation identifier
        user_id: user identifier

    Returns the same dict shape as run_inference().
    """
    try:
        camera_state = _CAMERA_TRACKER.get_or_create(cam_id)
        model = camera_state["model"]
        current_frame = _CAMERA_TRACKER.increment_frame(cam_id)

        input_data = {
            "frame":        frame,
            "cam_id":       cam_id,
            "org_id":       org_id,
            "user_id":      user_id,
            "frame_number": current_frame,
        }
        prediction = predict_frame_fn(input_data, model)
        return output_frame_fn(prediction)

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"[ERROR] Camera {cam_id}: {error_msg}")
        print(traceback.format_exc())
        timestamp = datetime.utcnow().isoformat() + 'Z'
        return {
            "cam_id": cam_id,
            "org_id": org_id,
            "user_id": user_id,
            "frame_number": 0,
            "timestamp": timestamp,
            "guns": [],
            "gun_holders": [],
            "persons_present": [],
            "alerts": [],
            "annotated_frame": "",
            "stats": {},
            "status": 1,
            "error": error_msg
        }

# ===================== OPTIONAL HELPER FUNCTIONS =====================
# Your websocket doesn't need to call these, but they're available if needed

def reset_camera(cam_id: int):
    """
    Reset tracking for a specific camera.
    Optional: Call when starting a new stream.
    """
    _CAMERA_TRACKER.reset(cam_id)

def cleanup_camera(cam_id: int):
    """
    Cleanup camera resources.
    Optional: Call when stream ends for memory cleanup.
    """
    _CAMERA_TRACKER.remove(cam_id)

def get_camera_stats(cam_id: int) -> Dict[str, Any]:
    """
    Get current tracking statistics.
    Optional: For monitoring/debugging.
    """
    return _CAMERA_TRACKER.get_stats(cam_id)

def get_all_cameras() -> Dict[int, Dict[str, Any]]:
    """
    Get stats for all active cameras.
    Optional: For system monitoring.
    """
    stats = {}
    with _CAMERA_TRACKER.lock:
        for cam_id in list(_CAMERA_TRACKER.cameras.keys()):
            stats[cam_id] = _CAMERA_TRACKER.get_stats(cam_id)
    return stats

# ===================== TESTING (STANDALONE MODE) =====================
def _decode_b64_image(b64_str: str) -> np.ndarray:
    """Helper: Decode base64 to OpenCV image"""
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    decoded = base64.b64decode(b64_str)
    arr = np.frombuffer(decoded, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def test_inference(video_path: str, cam_id: int = 1):
    """
    Benchmarked test using run_inference_raw() — raw numpy frames, no
    base64 encode/decode overhead. Measures pure inference latency.
    """
    import time
    import statistics

    print("=" * 70)
    print("BENCHMARKED GUN DETECTION TEST")
    print("=" * 70)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n  Source video : {total_frames} frames @ {src_fps:.1f} FPS")
    print(f"  Camera ID    : {cam_id}")
    print(f"  Alert mode   : {'First detection only' if CONFIG_OVERRIDES['ALERT_ON_FIRST_DETECTION_ONLY'] else 'With cooldown'}")
    print(f"  Inference    : parallel TensorRT (pose + gun), stream=True, letterbox")
    print(f"  OSNet ReID   : {'enabled' if CONFIG_OVERRIDES.get('USE_OSNET') else 'disabled (IoU tracking)'}")
    print(f"  INFER_IMGSZ  : {CONFIG_OVERRIDES.get('INFER_IMGSZ', 640)}px letterbox")
    print(f"  GUN_IMGSZ    : {CONFIG_OVERRIDES.get('GUN_INFER_IMGSZ', 480)}px\n")
    print("Press 'q' to quit early\n")

    t_inference = []
    frame_count  = 0
    total_alerts = 0
    wall_start   = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t0 = time.perf_counter()

        # run_inference_raw — no encode/decode, direct numpy path
        result = run_inference_raw(frame, cam_id=cam_id, org_id=1, user_id=1)
        dt_infer = (time.perf_counter() - t0) * 1000
        t_inference.append(dt_infer)

        if result.get("status", 0) != 0:
            print(f"\nError at frame {frame_count}: {result.get('error', 'Unknown')}")
            break

        alerts = result.get("alerts", [])
        total_alerts += len(alerts)
        if alerts:
            print()
            for alert in alerts:
                level = alert.get("level", "HIGH")
                icon  = "CRITICAL" if level == "CRITICAL" else "HIGH"
                print(f"  [{icon}] ALERT - ID:{alert.get('track_id')} "
                      f"(conf: {alert.get('confidence', 0):.2f})")

        live_fps = frame_count / (time.perf_counter() - wall_start)
        print(f"\r[F{frame_count:4d}/{total_frames}] "
              f"infer={dt_infer:6.1f}ms  "
              f"live={live_fps:5.1f}fps  "
              f"armed={len(result.get('gun_holders', []))}",
              end="", flush=True)

        # Display annotated frame
        b64 = result.get("annotated_frame", "")
        if b64:
            try:
                arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imshow(f"Gun Detection - Camera {cam_id}", img)
            except Exception:
                pass
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n\nStopped by user")
            break

    wall_elapsed = time.perf_counter() - wall_start
    cap.release()
    cv2.destroyAllWindows()

    if not t_inference:
        print("\nNo frames processed.")
        return

    # Exclude first frame (TRT cold-start) from stats
    steady = t_inference[1:] if len(t_inference) > 1 else t_inference

    TARGET_FPS = 15.0
    achieved_fps = frame_count / wall_elapsed
    med_fps      = 1000.0 / statistics.median(steady)
    avg_fps      = 1000.0 / statistics.mean(steady)

    print("\n\n" + "=" * 70)
    print("  BENCHMARK REPORT")
    print("=" * 70)
    print(f"  Frames processed : {frame_count}  (frame 1 excluded from stats = TRT warmup)")
    print(f"  Wall time        : {wall_elapsed:.2f}s")
    print(f"  Source FPS       : {src_fps:.1f}")
    print()
    print(f"  {'Metric':<22} {'min':>7} {'avg':>7} {'med':>7} {'max':>7} {'p95':>7}")
    print(f"  {'-'*57}")
    p95 = sorted(steady)[int(len(steady) * 0.95)]
    print(f"  {'inference (ms)':<22} "
          f"{min(steady):>7.1f} "
          f"{statistics.mean(steady):>7.1f} "
          f"{statistics.median(steady):>7.1f} "
          f"{max(steady):>7.1f} "
          f"{p95:>7.1f}")
    print()
    print(f"  Wall-clock FPS   : {achieved_fps:6.2f}  (includes TRT warmup frame)")
    print(f"  Steady-state avg : {avg_fps:6.2f}  fps")
    print(f"  Steady-state med : {med_fps:6.2f}  fps")
    print(f"  Target FPS       : {TARGET_FPS:.1f}")
    print()

    if med_fps >= TARGET_FPS:
        print(f"  PASS -- {med_fps:.1f} fps median >= {TARGET_FPS} fps target")
    else:
        gap = TARGET_FPS - med_fps
        print(f"  BELOW TARGET -- {med_fps:.1f} fps median (need {gap:.1f} more fps)")

    print()
    print(f"  Total alerts     : {total_alerts}")
    stats = get_camera_stats(cam_id)
    print(f"  Armed IDs        : {stats.get('armed_ids', [])}")
    print("=" * 70)

    cleanup_camera(cam_id)

# ===================== MAIN =====================
if __name__ == "__main__":
    # Test the inference function
    VIDEO_PATH = r"video6.mp4"
    
    print("\n🧪 TESTING MODE")
    print("This simulates how your websocket calls run_inference()\n")
    
    test_inference(VIDEO_PATH, cam_id=1)