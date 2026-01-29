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
    "ALERT_COOLDOWN_FRAMES": 90,  # 3 seconds @ 30fps (if not first-only)
    "ALERT_ON_CONFIDENCE_INCREASE": False,
    "CONFIDENCE_JUMP_THRESHOLD": 0.15,
    
    # Memory persistence
    "GUN_HOLDER_MEMORY_FRAMES": 150,  # 5 seconds @ 30fps
    "GUN_HOLDER_DECAY_CONFIDENCE": True,
    "CONFIDENCE_DECAY_RATE": 0.02,
    "MIN_PERSISTENT_CONFIDENCE": 0.40,
    
    # Detection thresholds
    "CRITICAL_THRESHOLD": 0.85,  # Red/Critical alerts
    "HIGH_THRESHOLD": 0.65,       # Orange/High alerts
    "FINAL_CONFIDENCE_THRESHOLD": 0.65,
    
    # Model confidence
    "CONF_THR_POSE": 0.25,
    "CONF_THR_WRIST": 0.20,
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
                camera_state["model"]["gun_holder_memory"].reset()
                camera_state["model"]["alert_manager"].reset()
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
            gun_memory = model["gun_holder_memory"]
            
            all_armed = gun_memory.get_all_armed()
            
            return {
                "cam_id": cam_id,
                "total_frames": camera_state["frame_counter"],
                "armed_count": len(all_armed),
                "armed_ids": all_armed
            }

# Global camera tracker instance
_CAMERA_TRACKER = CameraTracker()

# ===================== MAIN INFERENCE FUNCTION =====================
def run_inference(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main inference function - COMPATIBLE WITH EXISTING WEBSOCKET.
    
    Your websocket sends:
    {
        "cam_id": 123,
        "org_id": 2,
        "user_id": 2,
        "encoding": "<base64_jpeg>"
    }
    
    This function:
    - Auto-assigns frame_number internally per camera
    - Maintains persistent tracking per camera
    - Returns same format as before + new tracking features
    
    Returns:
    {
        "cam_id": 123,
        "org_id": 2,
        "user_id": 2,
        "frame_number": 42,
        "timestamp": "2026-01-28T12:34:56.789Z",
        "guns": [...],
        "gun_holders": [5, 7],
        "persons_present": [5, 7, 12],
        "alerts": [{...}],
        "annotated_frame": "<base64_jpeg>",
        "stats": {...},
        "status": 0
    }
    """
    try:
        # Get camera ID (default to 0 if missing)
        cam_id = payload.get("cam_id", 0)
        
        # Get or create tracking state for this camera
        camera_state = _CAMERA_TRACKER.get_or_create(cam_id)
        model = camera_state["model"]
        
        # Auto-assign frame number (increments automatically per camera)
        current_frame = _CAMERA_TRACKER.increment_frame(cam_id)
        payload["frame_number"] = current_frame
        
        # Parse input using standard function
        input_data = input_frame_fn(payload, content_type="application/json")
        
        # Run detection with persistent tracking
        prediction = predict_frame_fn(input_data, model)
        
        # Format and return output (includes timestamp from predict_frame_fn)
        return output_frame_fn(prediction)
    
    except Exception as e:
        # Return error response (backward compatible)
        import traceback
        error_msg = str(e)
        print(f"[ERROR] Camera {payload.get('cam_id', -1)}: {error_msg}")
        print(traceback.format_exc())
        
        # Generate timestamp for error response
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        return {
            "cam_id": payload.get("cam_id", -1),
            "org_id": payload.get("org_id", -1),
            "user_id": payload.get("user_id", -1),
            "frame_number": 0,
            "timestamp": timestamp,  # ← ADDED
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
    Test function - simulates how your websocket uses run_inference.
    Run this to verify everything works before deploying.
    """
    print("="*70)
    print("TESTING GUN DETECTION - Simulating Websocket Behavior")
    print("="*70)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n🎥 Video: {total_frames} frames @ {fps:.1f} FPS")
    print(f"📹 Camera ID: {cam_id}")
    print(f"⚙️  Alert mode: {'First detection only' if CONFIG_OVERRIDES['ALERT_ON_FIRST_DETECTION_ONLY'] else 'With cooldown'}")
    print(f"💾 Memory: {CONFIG_OVERRIDES['GUN_HOLDER_MEMORY_FRAMES']/fps:.1f} seconds\n")
    print("Press 'q' to quit, 's' to show stats\n")
    
    frame_count = 0
    total_alerts = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Encode frame (EXACTLY like your websocket does)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        base64_frame = base64.b64encode(buffer).decode('utf-8')
        
        # Build payload (EXACTLY like your websocket)
        payload = {
            "cam_id": cam_id,
            "org_id": 1,
            "user_id": 1,
            "encoding": base64_frame
            # NOTE: NO frame_number - it's auto-assigned!
        }
        
        # Call run_inference (same as websocket)
        result = run_inference(payload)
        
        # Check for errors
        if result.get("status", 0) != 0:
            print(f"\n❌ Error at frame {frame_count}: {result.get('error', 'Unknown')}")
            break
        
        # Extract results
        guns = result.get("guns", [])
        gun_holders = result.get("gun_holders", [])
        alerts = result.get("alerts", [])
        timestamp = result.get("timestamp", "")
        total_alerts += len(alerts)
        
        # Print progress
        print(f"\r[Cam {cam_id}] Frame {result['frame_number']:4d}/{total_frames} | "
              f"Armed: {len(gun_holders):2d} | "
              f"Detections: {len(guns):2d} | "
              f"Alerts: {len(alerts):2d} (Total: {total_alerts})", 
              end="", flush=True)
        
        # Show alert details
        if alerts:
            print()  # New line for alerts
            for alert in alerts:
                level = alert.get("level", "HIGH")
                track_id = alert.get("track_id", -1)
                confidence = alert.get("confidence", 0.0)
                alert_ts = alert.get("timestamp", timestamp)
                icon = "🚨" if level == "CRITICAL" else "⚠️"
                print(f"  {icon} {level} ALERT - ID:{track_id} (conf: {confidence:.2f}) @ {alert_ts}")
        
        # Display annotated frame
        if result.get("annotated_frame"):
            try:
                img = _decode_b64_image(result["annotated_frame"])
                cv2.imshow(f"Gun Detection - Camera {cam_id}", img)
            except Exception as e:
                print(f"\n⚠️  Display error: {e}")
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n\n⏹️  Stopped by user")
            break
        elif key == ord("s"):
            print("\n")
            stats = get_camera_stats(cam_id)
            print(f"  📊 Stats: {stats}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final summary
    print("\n\n" + "="*70)
    print("📊 TEST COMPLETE")
    print("="*70)
    
    stats = get_camera_stats(cam_id)
    print(f"Camera: {cam_id}")
    print(f"Frames processed: {stats['total_frames']}")
    print(f"Currently armed: {stats['armed_count']} persons")
    print(f"Armed IDs: {stats['armed_ids']}")
    print(f"Total alerts sent: {total_alerts}")
    print("="*70)
    
    # Optional cleanup
    cleanup_camera(cam_id)
    print(f"\n✓ Camera {cam_id} cleaned up")

# ===================== MAIN =====================
if __name__ == "__main__":
    # Test the inference function
    VIDEO_PATH = r"E:\All_models\gun_detection\guntest1.mp4"
    
    print("\n🧪 TESTING MODE")
    print("This simulates how your websocket calls run_inference()\n")
    
    test_inference(VIDEO_PATH, cam_id=1)