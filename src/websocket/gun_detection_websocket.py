import cv2
import json
import base64
import asyncio
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Full
from threading import Thread

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.local_models.gun_detection import run_inference_raw

logger = logging.getLogger("gun-detection")
logger.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Storage worker — runs in a background THREAD (not a separate process).
# Multiprocessing on Windows requires if __name__ == "__main__" guards and
# is extremely expensive per-connection. A daemon thread is sufficient here
# because S3 + DB I/O is the bottleneck, not CPU.
# ─────────────────────────────────────────────────────────────────────────────
def _storage_worker(q: Queue, client_id: str) -> None:
    """
    Background thread: dequeues (frame_num, annotated_b64, detections) tuples
    and persists them to S3 + Postgres. Imports are deferred so that missing
    AWS/DB credentials only crash the storage thread, not the inference loop.
    """
    try:
        from src.store_s3.gun_store import upload_to_s3
        from src.database.gun_query import insert_data
    except Exception as exc:
        logger.error("[%s] Storage worker failed to import dependencies: %s", client_id, exc)
        # Drain the queue so the sentinel is still consumed
        while True:
            item = q.get()
            if item is None:
                break
        return

    logger.info("[%s] Storage worker started", client_id)
    while True:
        item = q.get()
        if item is None:
            break
        frame_num, annotated_b64, detections = item
        try:
            s3_url = upload_to_s3(annotated_b64, frame_num)
            insert_data(detections, s3_url)
            logger.info("[%s] Stored frame %d → %s", client_id, frame_num, s3_url)
        except Exception as exc:
            logger.error("[%s] Storage error on frame %d: %s", client_id, frame_num, exc)
    logger.info("[%s] Storage worker exiting", client_id)



# ─────────────────────────────────────────────────────────────────────────────
# Main detection loop
# ─────────────────────────────────────────────────────────────────────────────
def run_gun_detection_detection(
    client_id: str,
    video_url: str,
    camera_id: int,
    user_id: int,
    org_id: int,
    sessions: dict,
    loop: asyncio.AbstractEventLoop,
    storage_executor: ThreadPoolExecutor,   # kept for API compatibility, unused
) -> None:
    """
    Runs gun detection in a background thread (launched by gun_handler via
    ThreadPoolExecutor). Reads frames directly from cv2.VideoCapture and calls
    run_inference_raw() to avoid the base64 encode→decode round-trip.

    Key fixes vs previous version:
    - Frames with no gun detections are sent normally (not treated as errors).
    - Storage runs in a daemon thread, not a multiprocessing.Process.
    - Stream only stops on explicit stop_stream, VideoCapture EOF, or exception.
    """
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        logger.error("[%s] Cannot open video source: %s", client_id, video_url)
        _send(sessions, client_id, loop, {"status": "error", "message": f"Cannot open stream: {video_url}"})
        return

    # Storage queue — bounded to avoid unbounded memory growth under slow I/O
    store_queue: Queue = Queue(maxsize=500)
    storage_thread = Thread(
        target=_storage_worker,
        args=(store_queue, client_id),
        daemon=True,
        name=f"storage-{client_id}",
    )
    storage_thread.start()

    frame_num = 0
    logger.info("[%s] Gun detection started on %s", client_id, video_url)

    try:
        while cap.isOpened() and sessions.get(client_id, {}).get("streaming", False):
            ret, frame = cap.read()
            if not ret:
                logger.info("[%s] Stream ended (no more frames)", client_id)
                break

            frame_num += 1

            try:
                result = run_inference_raw(frame, camera_id, org_id, user_id)
            except Exception as exc:
                logger.exception("[%s] Inference error on frame %d", client_id, frame_num)
                continue   # skip this frame, keep streaming

            # Always send the result — even frames with no detections are valid
            # (the client needs them to know the stream is alive).
            # Only omit annotated_frame when it's empty to save bandwidth.
            payload = {"detections": result}
            _send(sessions, client_id, loop, payload)

            # Queue for storage only when guns are detected
            if result.get("guns") and result.get("annotated_frame"):
                try:
                    store_queue.put_nowait((frame_num, result["annotated_frame"], result))
                except Full:
                    logger.warning("[%s] Storage queue full; frame %d dropped", client_id, frame_num)

    except Exception:
        logger.exception("[%s] Unexpected error in detection loop", client_id)
    finally:
        cap.release()
        store_queue.put(None)          # signal storage thread to exit
        storage_thread.join(timeout=10)
        if client_id in sessions:
            sessions[client_id]["streaming"] = False
        logger.info("[%s] Gun detection stopped, resources released", client_id)


def _send(sessions: dict, client_id: str, loop: asyncio.AbstractEventLoop, payload: dict) -> None:
    """Thread-safe WebSocket send. Silently drops if the session is gone."""
    try:
        ws = sessions.get(client_id, {}).get("ws")
        if ws is None:
            return
        asyncio.run_coroutine_threadsafe(
            ws.send_text(json.dumps(payload)),
            loop,
        )
    except Exception as exc:
        logger.warning("[%s] WebSocket send failed: %s", client_id, exc)
