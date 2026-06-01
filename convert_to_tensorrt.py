"""
convert_to_tensorrt.py

Converts YOLO models to TensorRT engines for maximum inference speed.

Usage:
    python convert_to_tensorrt.py

Outputs (in project root):
    best.engine           — gun detection TensorRT engine (YOLOv8n fine-tune)
    yolo11n-pose.engine   — pose estimation TensorRT engine

Requirements:
    - NVIDIA GPU with CUDA
    - TensorRT installed (comes with ultralytics + torch)
    - Run ONCE before starting the server

Notes:
    - FP16 is used by default (2x faster than FP32, negligible accuracy loss)
    - dynamic=True lets you tune GUN_INFER_IMGSZ at runtime (e.g. 480 for speed)
    - The engine is device-specific — regenerate if you change GPU
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def convert(model_path: str, output_name: str, imgsz: int = 640, half: bool = True,
            dynamic: bool = False):
    """Export a YOLO .pt model to a TensorRT .engine file."""
    from ultralytics import YOLO
    import torch

    if not torch.cuda.is_available():
        print(f"[ERROR] CUDA not available — TensorRT export requires a GPU.")
        sys.exit(1)

    abs_path = os.path.join(_HERE, model_path)
    if not os.path.exists(abs_path):
        print(f"[SKIP] {model_path} not found at {abs_path}")
        return None

    engine_path = os.path.join(_HERE, output_name)
    if os.path.exists(engine_path):
        print(f"[SKIP] {output_name} already exists — delete it to re-export.")
        return engine_path

    print(f"\n[EXPORT] {model_path} → {output_name}")
    print(f"         imgsz={imgsz}, half={half}, dynamic={dynamic}")

    model = YOLO(abs_path)
    exported = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,          # FP16 — fastest on modern GPUs
        dynamic=dynamic,    # dynamic=True allows variable input sizes at runtime
        device=0,           # GPU 0
        workspace=4,        # GB of TRT workspace
        verbose=False,
    )
    print(f"[OK]   Exported → {exported}")
    return exported


if __name__ == "__main__":
    print("=" * 60)
    print("  YOLO -> TensorRT Conversion")
    print("=" * 60)

    # 1. Gun detection model — dynamic shapes so we can run at 416/480 for speed
    convert("best.pt", "best.engine", imgsz=640, half=True, dynamic=True)

    # 2. Pose model — nano variant (fastest), dynamic shapes
    convert("yolo11n-pose.pt", "yolo11n-pose.engine", imgsz=640, half=True, dynamic=True)

    print("\n[DONE] All conversions complete.")
    print("       Start the server — it will load .engine files automatically.")
    print()
    print("  TIP: engines exported with dynamic=True support variable input sizes.")
    print("       Set GUN_INFER_IMGSZ=480 in CONFIG_OVERRIDES for ~30% faster gun inference.")
