# Gun Detection Model

A real-time gun (weapon) detection service that streams video from a camera source, runs YOLO-based inference with person tracking and re-identification, raises alerts when armed individuals are detected, and persists annotated evidence frames to AWS S3 along with metadata in PostgreSQL. The service is exposed over a FastAPI WebSocket so a client can start/stop a stream and receive live detection payloads.

---

## Use Case

This model is built for **security and surveillance** scenarios where early detection of firearms is critical:

- Public spaces (schools, malls, banks, transit stations)
- Private premises and facility monitoring
- Live CCTV / IP camera streams via AWS Kinesis Video Streams (KVS)
- Any HTTP-accessible video source (HLS, MP4, RTSP-over-HLS)

Key capabilities:

- Detects guns and identifies the **person holding** the gun (gun-holder association via pose + wrist proximity).
- Maintains **persistent per-camera tracking** so the same individual isn't re-alerted on every frame.
- Sends graded alerts (`HIGH`, `CRITICAL`) based on detection confidence.
- Stores **only frames containing weapons** to S3 to minimize storage cost.
- Stream-based architecture supports multiple concurrent cameras.

---

## Project Structure

```
gun_model/
├── app.py                          # FastAPI entry point — exposes WebSocket
├── Dockerfile                      # CUDA + PyTorch base image
├── requirements.txt                # Python dependencies (torch comes from base image)
├── gunn2.pt                        # YOLO weights — gun detector
├── non_weapons.pt                  # YOLO weights — false-positive suppressor
└── src/
    ├── websocket/
    │   └── gun_detection_websocket.py   # Frame loop + inference + storage queue
    ├── handlers/
    │   └── gun_handler.py               # WebSocket lifecycle, start/stop actions
    ├── local_models/
    │   ├── gun_detection.py             # Per-camera tracker + run_inference()
    │   └── inference_gun_detection_reid.py  # Model load, predict, annotate
    ├── store_s3/
    │   └── gun_store.py                 # S3 upload helpers
    ├── database/
    │   └── gun_query.py                 # Postgres insert for detections
    └── utils/
        └── kvs_stream.py                # Resolve AWS KVS stream → HLS URL
```

---

## Architecture / Flow

```
                ┌─────────────────────┐
   Client ───►  │  /ws/gundetection/  │  WebSocket (FastAPI)
                │      {client_id}    │
                └──────────┬──────────┘
                           │  {"action":"start_stream", stream_name, camera_id, ...}
                           ▼
                ┌─────────────────────┐
                │  gun_handler.py     │  Accept, validate, resolve KVS → HLS URL
                └──────────┬──────────┘
                           │  run_in_executor(...)
                           ▼
            ┌─────────────────────────────┐
            │ gun_detection_websocket.py  │  cv2.VideoCapture(hls_url)
            │   • Read frame              │
            │   • Encode → base64         │
            │   • run_inference(payload)  │──┐
            └──────────────┬──────────────┘  │
                           │                 │
                           │                 ▼
                           │     ┌──────────────────────────┐
                           │     │ gun_detection.py         │
                           │     │  CameraTracker (per-cam) │
                           │     │   • model_fn             │
                           │     │   • predict_frame_fn     │
                           │     │   • output_frame_fn      │
                           │     └──────────────────────────┘
                           │
            ┌──────────────┴────────────────┐
            ▼                               ▼
   WebSocket send                Multiprocess Storage Worker
   (detections JSON              (only when guns detected)
   + annotated frame)                ├── upload_to_s3()
                                     └── insert_data() → Postgres
```

### Detection Logic (high level)

- Pose estimation locates wrists; gun bounding boxes are associated with the nearest wrist to identify the **gun holder**.
- A `gun_holder_memory` keeps each armed track alive for ~150 frames (5 s @ 30 fps) so brief occlusions don't cause flicker.
- Alerts fire **once per person** by default (`ALERT_ON_FIRST_DETECTION_ONLY = True`), tuned by `HIGH_THRESHOLD` (0.65) and `CRITICAL_THRESHOLD` (0.85).
- Configuration knobs live in `CONFIG_OVERRIDES` inside [src/local_models/gun_detection.py](src/local_models/gun_detection.py).

---

## Installation

### Option A — Docker (recommended, GPU)

The Docker image is built on `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`, so CUDA-enabled PyTorch ships with the base image.

```bash
# Build
docker build -t gun-detection:latest .

# Run (requires NVIDIA Container Toolkit on the host for GPU)
docker run --gpus all -p 8004:8004 \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e DB_HOST=... -e DB_NAME=... -e DB_USER=... -e DB_PASSWORD=... \
  gun-detection:latest
```

The container exposes port **8004** and starts `uvicorn app:app`.

### Option B — Local Python (development)

Requires Python 3.10+, CUDA 12.1 drivers (for GPU), and a working OpenCV install.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell)
# source .venv/bin/activate     # Linux / macOS

# 2. Install PyTorch matching your CUDA (not in requirements.txt by design)
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install the rest
pip install -r requirements.txt
```

### Required environment variables

| Variable | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | S3 upload + KVS access |
| `AWS_DEFAULT_REGION` | Default AWS region (overridable per request) |
| `S3_BUCKET` | Bucket name where annotated frames are stored |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | Postgres connection for detection metadata |

### Model weights

Both `gunn2.pt` and `non_weapons.pt` must be present at the project root. They are loaded on the first inference call for each camera.

---

## Running the Service

### Start the server

```bash
# Docker (already running via CMD)
# Or locally:
uvicorn app:app --host 0.0.0.0 --port 8004 --reload
```

The server is now listening on `ws://localhost:8004/ws/gundetection/{client_id}`.

### Connect a client

Open a WebSocket to `ws://localhost:8004/ws/gundetection/<your-client-id>` and send a JSON `start_stream` message:

```json
{
    "action": "start_stream",
    "stream_name": "Cam424",
    "camera_id": 10,
    "user_id": 10,
    "org_id": 10,
    "threshold": 80,
    "alert_rate": 90,
    "region": "us-east-1"
}
```

- If `stream_name` starts with `https`, it is used directly as the video URL (HLS/MP4).
- Otherwise it is resolved as an AWS KVS stream name in the given `region`.

Example direct URL:

```
https://andymerry.s3.us-east-1.amazonaws.com/1191560-hd_1920_1080_25fps.mp4
```

### Stop a stream

```json
{ "action": "stop_stream" }
```

### Detection payload (per frame)

```json
{
  "detections": {
    "cam_id": 10,
    "org_id": 10,
    "user_id": 10,
    "frame_number": 142,
    "timestamp": "2026-05-25T12:34:56.789Z",
    "guns": [ { "bbox": [...], "confidence": 0.87 } ],
    "gun_holders": [5, 7],
    "persons_present": [5, 7, 12],
    "alerts": [
      { "level": "CRITICAL", "track_id": 5, "confidence": 0.91, "timestamp": "..." }
    ],
    "annotated_frame": "<base64-jpeg>",
    "stats": { ... },
    "status": 0
  }
}
```

---

## Standalone Testing (no WebSocket)

A built-in test harness runs the inference loop against a local video file and shows the annotated frames in an OpenCV window:

```bash
python -m src.local_models.gun_detection
```

Edit the `VIDEO_PATH` variable at the bottom of [src/local_models/gun_detection.py](src/local_models/gun_detection.py#L384) before running. Press `q` to quit, `s` to print live tracker stats.

---

## Tuning

Open [src/local_models/gun_detection.py](src/local_models/gun_detection.py#L41) and adjust `CONFIG_OVERRIDES`:

- `ALERT_ON_FIRST_DETECTION_ONLY` — alert once per tracked person vs. every cooldown window.
- `ALERT_COOLDOWN_FRAMES` — frames between repeat alerts when first-only is off.
- `GUN_HOLDER_MEMORY_FRAMES` — how long an armed identity persists across occlusions.
- `HIGH_THRESHOLD` / `CRITICAL_THRESHOLD` — alert level cutoffs.
- `CONF_THR_POSE`, `CONF_THR_WRIST` — pose-keypoint confidence floors used to bind a gun to a person.

---

## Notes

- Frames without any gun detection are **not** stored to S3; only annotated frames where `result["guns"]` is non-empty are queued for upload.
- Storage runs in a separate process (`multiprocessing.Process`) so disk/network I/O never blocks inference.
- Each camera (`cam_id`) gets its own model instance and frame counter — the service is safe for multi-camera concurrent streaming.

