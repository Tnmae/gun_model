# Gun Detection Service

Real-time firearm detection service built for security and surveillance. Streams video from any camera source, runs parallel TensorRT inference to detect guns and identify the person holding them, raises graded alerts, and persists annotated evidence frames to AWS S3 with metadata in PostgreSQL. Exposed over a FastAPI WebSocket so clients can start/stop streams and receive live detection payloads.

---

## Use Case

- Public spaces — schools, malls, banks, transit stations
- Private premises and facility monitoring
- Live CCTV / IP camera streams via AWS Kinesis Video Streams (KVS), RTSP, or HLS
- Local video files for testing and development

Key capabilities:

- Detects guns and identifies the **person holding** the gun via pose estimation and wrist-proximity association.
- Maintains **persistent per-camera tracking** so the same individual is not re-alerted on every frame.
- Sends graded alerts (`HIGH`, `CRITICAL`) based on detection confidence.
- Stores **only frames containing weapons** to S3 to minimise storage cost.
- Supports multiple concurrent cameras, each with isolated model state.

---

## Architecture

```
                ┌──────────────────────────┐
   Client ───►  │  /ws/gundetection/       │  WebSocket (FastAPI)
                │      {client_id}         │
                └────────────┬─────────────┘
                             │  {"action":"start_stream", ...}
                             ▼
                ┌──────────────────────────┐
                │  gun_handler.py          │  Validates stream_name:
                │                          │  • local file / rtsp:// / http(s)://
                │                          │    → passed directly to VideoCapture
                │                          │  • bare KVS stream name
                │                          │    → resolved via get_kvs_hls_url()
                └────────────┬─────────────┘
                             │  loop.run_in_executor(ThreadPoolExecutor)
                             ▼
            ┌────────────────────────────────────┐
            │  gun_detection_websocket.py        │
            │  cv2.VideoCapture(url)             │
            │  for each frame:                   │
            │    run_inference_raw(frame, cam_id)│──────────────────────┐
            └──────────────────┬─────────────────┘                      │
                               │                                         ▼
                               │                    ┌────────────────────────────────┐
                               │                    │  gun_detection.py              │
                               │                    │  CameraTracker (per-camera)    │
                               │                    │    model_fn() on first call    │
                               │                    │    predict_frame_fn()          │
                               │                    │    output_frame_fn()           │
                               │                    └────────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                      ▼
   ws.send_text(detections JSON)       Storage thread (daemon)
   every frame                         only when guns detected:
                                         upload_to_s3()
                                         insert_data() → Postgres
```

### Inference pipeline (per frame)

```
Raw frame (BGR numpy)
        │
        ▼
Letterbox resize to 640×640
(eliminates internal TRT resize overhead)
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
yolo11n-pose.engine (TRT, FP16)         best.engine (TRT, FP16)
Pose + person detection                 Gun detection (YOLOv8n fine-tune)
stream=True, parallel thread            stream=True, parallel thread
        │                                          │
        ▼                                          ▼
Unscale boxes to original coords        Unscale boxes to original coords
Build DeepSort detections               NMS + confidence filter
Unit-vector embeddings (IoU tracking)   Size filter (rejects cars/walls)
        │                                          │
        └──────────────┬───────────────────────────┘
                       ▼
              DeepSort person tracker
              (IoU-only, OSNet disabled)
                       │
                       ▼
              Wrist-proximity association
              (gun bbox ↔ person wrist keypoint)
                       │
                       ▼
              GunTracker (IoU-based, stable G-IDs)
                       │
                       ▼
              AlertManager (per-person, first-detection)
                       │
                       ▼
              Annotated frame + detection JSON
```

### Performance (benchmarked on local GPU)

| Metric | Value |
|---|---|
| Steady-state median latency | ~46 ms |
| Throughput median | **21.7 fps** |
| Throughput avg | 19.9 fps |
| p95 latency | ~71 ms |
| Target | 15 fps |

---

## Project Structure

```
gun_model/
├── app.py                               # FastAPI entry point — WebSocket endpoint
├── Dockerfile                           # CUDA 12.1 + PyTorch + TensorRT base image
├── requirements.txt                     # Python dependencies (torch from base image)
├── convert_to_tensorrt.py               # One-time .pt → .engine conversion script
├── test_websocket.py                    # End-to-end WebSocket test client
├── best.pt                              # YOLOv8n fine-tuned gun detector weights
├── best.engine                          # TensorRT FP16 engine (generated by convert script)
├── yolo11n-pose.pt                      # YOLO11n pose model weights
├── yolo11n-pose.engine                  # TensorRT FP16 engine (generated by convert script)
└── src/
    ├── websocket/
    │   └── gun_detection_websocket.py   # Frame loop, run_inference_raw, storage queue
    ├── handlers/
    │   └── gun_handler.py               # WebSocket lifecycle, stream URL resolution
    ├── local_models/
    │   ├── gun_detection.py             # CameraTracker, run_inference_raw, test harness
    │   └── inference_gun_detection_reid.py  # Model load, parallel predict, annotate
    ├── store_s3/
    │   └── gun_store.py                 # S3 upload helpers
    ├── database/
    │   └── gun_query.py                 # Postgres insert for detection metadata
    └── utils/
        └── kvs_stream.py                # Resolve AWS KVS stream name → HLS URL
```

---

## Models

| File | Description | Size |
|---|---|---|
| `best.pt` | YOLOv8n fine-tuned for gun detection (HuggingFace) | ~6 MB |
| `best.engine` | TensorRT FP16 engine built from `best.pt` | ~9 MB |
| `yolo11n-pose.pt` | YOLO11n pose estimation (auto-downloaded) | ~6 MB |
| `yolo11n-pose.engine` | TensorRT FP16 engine built from pose model | ~10 MB |

Both `.engine` files are **device-specific** — regenerate them if you change GPU (see [TensorRT Conversion](#tensorrt-conversion)).

The `non_weapons.pt` cross-verification model has been removed. The size filter, confidence threshold, and wrist-proximity association provide sufficient false-positive suppression without the latency cost of a third model.

---

## Installation

### Option A — Docker (recommended for production)

The image is built on `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`. TensorRT is installed on top. The `convert_to_tensorrt.py` script runs at build time to bake the `.engine` files into the image.

```bash
# Build (requires NVIDIA GPU at build time for TRT conversion)
docker build -t gun-detection:latest .

# Run
docker run --gpus all -p 8004:8004 \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_DEFAULT_REGION=ap-south-1 \
  -e DB_HOST=... -e DB_NAME=... -e DB_USER=... -e DB_PASSWORD=... \
  gun-detection:latest
```

If no GPU is available at build time, the TRT conversion step is skipped with a warning and the service falls back to `.pt` weights automatically at runtime.

### Option B — Local Python (development)

Requires Python 3.10+, CUDA 12.1 drivers, and NVIDIA TensorRT.

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install PyTorch with CUDA (not in requirements.txt — provided by Docker base image)
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install TensorRT
pip install tensorrt

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Convert models to TensorRT (one-time, requires GPU)
python convert_to_tensorrt.py
```

### Environment variables

| Variable | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | S3 upload + KVS stream resolution |
| `AWS_DEFAULT_REGION` | Default AWS region |
| `S3_BUCKET` | Bucket where annotated frames are stored |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | Postgres connection |

---

## TensorRT Conversion

Run once before starting the server (or after changing GPU):

```bash
python convert_to_tensorrt.py
```

This exports:
- `best.pt` → `best.engine` (FP16, dynamic shapes)
- `yolo11n-pose.pt` → `yolo11n-pose.engine` (FP16, dynamic shapes)

Dynamic shapes means `GUN_INFER_IMGSZ` can be tuned at runtime without re-exporting. The server automatically uses `.engine` files when present and falls back to `.pt` if they are missing.

---

## Running the Service

```bash
# Local development
.venv\Scripts\uvicorn app:app --host 0.0.0.0 --port 8004

# With auto-reload during development
.venv\Scripts\uvicorn app:app --host 0.0.0.0 --port 8004 --reload
```

The server listens on `ws://localhost:8004/ws/gundetection/{client_id}`.

---

## WebSocket API

### Connect

```
ws://localhost:8004/ws/gundetection/<client_id>
```

### Start a stream

```json
{
    "action": "start_stream",
    "stream_name": "Cam424",
    "camera_id": 10,
    "user_id": 10,
    "org_id": 10,
    "region": "ap-south-1"
}
```

`stream_name` routing:

| Value | Behaviour |
|---|---|
| Local file path (e.g. `video.mp4`, `C:\videos\test.mp4`) | Passed directly to `cv2.VideoCapture` |
| `rtsp://...` or `rtsps://...` | Passed directly to `cv2.VideoCapture` |
| `http://...` or `https://...` | Passed directly to `cv2.VideoCapture` |
| Bare stream name (e.g. `Cam424`) | Resolved via AWS KVS → HLS URL |

### Stop a stream

```json
{ "action": "stop_stream" }
```

### Detection payload (received per frame)

```json
{
  "detections": {
    "cam_id": 10,
    "org_id": 10,
    "user_id": 10,
    "frame_number": 142,
    "timestamp": "2026-05-25T12:34:56.789+00:00",
    "guns": [
      {
        "gun_id": 1,
        "bbox": [x1, y1, x2, y2],
        "score": 0.87,
        "holder_id": 5
      }
    ],
    "gun_holders": [
      { "track_id": 5, "confidence": 0.87 }
    ],
    "persons_present": [5, 7],
    "alerts": [
      {
        "track_id": 5,
        "confidence": 0.91,
        "level": "CRITICAL",
        "timestamp": "2026-05-25T12:34:56.789+00:00"
      }
    ],
    "annotated_frame": "<base64-encoded JPEG>",
    "stats": {
      "raw_preds": 2,
      "verified_guns": 1,
      "guns_drawn": 1,
      "holders_drawn": 1,
      "persons_tracked": 3
    },
    "status": 0
  }
}
```

`status: 0` = success, `status: 1` = inference error (includes `"error"` key).

---

## Testing

### Standalone inference test (no server required)

Runs the full inference pipeline against `video.mp4` and prints a benchmark report:

```bash
python -m src.local_models.gun_detection
```

Output includes per-frame latency (min / avg / median / max / p95) and a pass/fail against the 15 fps target.

### End-to-end WebSocket test

Requires the server to be running first.

```bash
# Terminal 1 — start server
.venv\Scripts\uvicorn app:app --host 0.0.0.0 --port 8004

# Terminal 2 — run test
python test_websocket.py

# Options
python test_websocket.py --max-frames 100          # quick smoke test
python test_websocket.py --url rtsp://192.168.1.10:554/stream
python test_websocket.py --host 10.0.0.5 --port 8004 --camera-id 3
```

The test client sends the same `start_stream` JSON a real frontend sends, measures round-trip frame intervals, and prints a throughput report with pass/fail against 15 fps.

---

## Configuration

All tuning knobs are in `CONFIG_OVERRIDES` inside [`src/local_models/gun_detection.py`](src/local_models/gun_detection.py):

| Key | Default | Description |
|---|---|---|
| `FINAL_CONFIDENCE_THRESHOLD` | `0.65` | Minimum score for a gun detection to be reported |
| `CONF_THR_POSE` | `0.25` | Minimum pose detection confidence |
| `CONF_THR_WRIST` | `0.20` | Minimum wrist keypoint confidence for holder association |
| `ALERT_ON_FIRST_DETECTION_ONLY` | `True` | Alert once per tracked person; set `False` for repeat alerts |
| `ALERT_COOLDOWN_FRAMES` | `90` | Frames between repeat alerts when first-only is off (~3 s @ 30 fps) |
| `ALERT_THRESHOLD` | `0.60` | Minimum confidence to trigger an alert |
| `GUN_SKIP_FRAMES` | `1` | Run gun model every N frames (1 = every frame, no skipping) |
| `GUN_INFER_IMGSZ` | `480` | Gun model input size — smaller is faster (requires dynamic engine) |
| `INFER_IMGSZ` | `640` | Letterbox target before both models — must match engine build size |
| `USE_OSNET` | `False` | Enable OSNet ReID for appearance-based tracking (+35 ms/frame) |
| `INFERENCE_WORKERS` | `2` | Thread pool size for parallel pose + gun inference |

---

## Storage Behaviour

- Annotated frames are queued for S3 upload **only when guns are detected** (`result["guns"]` is non-empty).
- Storage runs in a **daemon thread** per camera session — S3/DB I/O never blocks inference.
- The storage queue is bounded at 500 items; frames are dropped with a warning if the queue fills (slow S3 connection).
- Each camera session gets its own storage thread, started when the stream begins and shut down cleanly when it ends.

---

## Notes

- Each `cam_id` gets its own `model_fn()` instance and frame counter — safe for multi-camera concurrent streaming.
- The first frame of each session is slow (~2–3 s) due to TRT execution context initialisation. Subsequent frames run at full speed.
- `run_inference_raw()` accepts a raw BGR numpy array and skips the base64 encode/decode round-trip used by the WebSocket path — use it for any caller that already has a decoded frame.
- S3 credential errors during local testing are expected and do not affect inference.
