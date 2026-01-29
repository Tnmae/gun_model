import cv2
import json
import base64
import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import numpy as np

# Add <project_root>/src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.local_models.gun_detection import run_inference

from src.store_s3.gun_store import upload_to_s3
from src.database.gun_query import insert_data
from multiprocessing import Process, Queue

from PIL import Image

logger = logging.getLogger("gun-detection")
logger.setLevel(logging.INFO)


def base64_to_cv2(b64_string):
    # Remove header if present
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]

    # Decode base64 to bytes
    img_bytes = base64.b64decode(b64_string)

    # Convert bytes to numpy array
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)

    # Decode image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return img


# ---------------------------------------------------------
# MULTIPROCESSING STORAGE WORKER
# ---------------------------------------------------------

def run_storage_worker(q, client_id):
    """
    Runs in a SEPARATE PROCESS.
    Handles S3 upload + DB insert.
    """

    logger.info(f"[{client_id}] Storage worker started.")

    while True:
        item = q.get()

        # Sentinel: exit
        if item is None:
            break

        frame_id, annotated_frame, detections = item

        try:
            # Upload to S3
            s3_url = upload_to_s3(annotated_frame, frame_id)

            # DB insert
            insert_data(detections, s3_url)

            logger.info(f"[{client_id}] Stored frame {frame_id}")

        except Exception as e:
            logger.error(f"[{client_id}] Error storing frame {frame_id}: {e}")

    logger.info(f"[{client_id}] Storage worker exiting...")

    

def run_gun_detection_detection(client_id: str, video_url: str, camera_id: int, user_id: int, org_id: int, sessions: dict, loop: asyncio.AbstractEventLoop, storage_executor: ThreadPoolExecutor):
    """
    Runs GUN detection in a separate thread.
    Sends WebSocket messages safely and stores frames to S3/DB in background threads to avoid blocking inference.
    """
    cap = cv2.VideoCapture(video_url)
    frame_num = 0
    # ---------------------------------------------------------
    # START MULTIPROCESS STORAGE WORKER
    # ---------------------------------------------------------
    store_queue = Queue(maxsize=1000)

    storage_process = Process(
        target=run_storage_worker,
        args=(store_queue, client_id),
        daemon=True
    )
    storage_process.start()

    

    while cap.isOpened() and sessions.get(client_id, {}).get("streaming", False):
        ret, frame = cap.read()
        if not ret:
            # No more frames
            break

        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        base64_frame = base64.b64encode(buffer).decode('utf-8')
        
        frame_num += 1
        try:
            # ---------------- PPE inference ----------------

            # Prepare input
            input_data = {
                "cam_id": camera_id,
                "org_id": org_id,
                "user_id": user_id,
                "encoding": base64_frame
            }

            result = run_inference(input_data)
            
            # ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            payload = {}

            ws = sessions[client_id]["ws"]

            if result and result.get('annotated_frame'):

                payload = {
                   "detections": result,
                   
                }
            # ---------------- WebSocket send ----------------
            
                if payload:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_text(json.dumps(payload)),
                        loop
                    )
                else:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_text(json.dumps({"error":"error in frame proecssing"})),
                        loop
                    )
                    break

                #------------------ STORE EVERY 20th FRAME -----------------
                annotated_frame=result.get('annotated_frame')
                if result["guns"]:

                    if annotated_frame is not None:
                        # JSON COPY to avoid race condition
                        safe_copy = json.loads(json.dumps(result))

                        try:
                            store_queue.put_nowait(
                                (frame_num, annotated_frame, safe_copy)
                            )
                        except:
                            logger.warning(
                                f"[{client_id}] Storage queue full; frame {frame_num} dropped."
                            )

            else:
                if ws:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_text(json.dumps({"success": False, "message": "error"})),
                        loop
                    )
                logger.warning(f"[{client_id}] Frame {frame_num}: No detections - error")
                break


        except Exception as e:
            print(f"[{client_id}] Frame {frame_num} error -> {e}")

    cap.release()

    #STOP STORAGE PROCESS
    store_queue.put(None)
    storage_process.join(timeout=5)
    
    if client_id in sessions:
        sessions[client_id]["streaming"] = False

    logger.info(f"[{client_id}] Gun Detection stopped and resources released")









