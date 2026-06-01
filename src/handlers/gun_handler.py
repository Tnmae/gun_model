import asyncio
import json
import logging
import os
from fastapi import WebSocket, WebSocketDisconnect
from src.utils.kvs_stream import get_kvs_hls_url

logger = logging.getLogger("websockets")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


async def gun_websocket_handler(executor, storage_executor, ws: WebSocket, client_id: str, sessions: dict, run_detection_fn, stream_type: str):
    await ws.accept()
    loop = asyncio.get_running_loop()  # get the loop inside the coroutine

    sessions[client_id] = {
        "ws": ws,
        "streaming": False,
        "inference_tasks": []
    }
    logger.info("[%s] %s WebSocket connected", client_id, stream_type)

    try:
        while True:
            try:
                msg = await ws.receive_text()
                if not msg.strip():
                    continue
                data = json.loads(msg)
            except json.JSONDecodeError:
                logger.warning("[%s] Received invalid JSON: %s", client_id, msg)
                continue
            except WebSocketDisconnect:
                logger.info("[%s] %s client disconnected", client_id, stream_type)
                break
            except Exception:
                logger.exception("[%s] Failed to read %s WebSocket message", client_id, stream_type)
                continue

            action = data.get("action")

            if action == "start_stream":
                try:
                    stream_name = data["stream_name"]
                    user_id = data["user_id"]
                    camera_id = data["camera_id"]
                    org_id = data["org_id"]
                    region = data.get("region", "ap-south-1")

                    # Pass the stream_name directly to cv2.VideoCapture when it is:
                    #   - an absolute/relative file path  (local video file)
                    #   - an rtsp:// or http:// URL       (IP camera / HLS direct)
                    #   - an https:// URL                 (HLS CDN)
                    # Only call get_kvs_hls_url() for bare KVS stream names.
                    _direct = (
                        stream_name.startswith("https://")
                        or stream_name.startswith("http://")
                        or stream_name.startswith("rtsp://")
                        or stream_name.startswith("rtsps://")
                        or os.path.exists(stream_name)   # local file
                    )
                    kvs_url = stream_name if _direct else get_kvs_hls_url(stream_name, region)
                    
                    # --------- Handle missing/invalid KVS URL gracefully ----------
                    if not kvs_url or kvs_url in ("None", "", None):
                        msg = f"no HLS URL on attempts: {stream_name}"
                        logger.warning("[%s] %s", client_id, msg)
                        try:
                            await ws.send_json({
                                "status": "error",
                                "message": msg,
                                "camera_id": camera_id,
                                "client_id": client_id
                            })
                        except Exception:
                            logger.exception("[%s] Failed to send error message to client", client_id)
                        continue  # skip detection start

                    client_args = (client_id, kvs_url, camera_id, user_id, org_id, sessions, loop, storage_executor)

                    sessions[client_id]["streaming"] = True

                    # Run detection in a separate thread
                    future = loop.run_in_executor(executor, run_detection_fn, *client_args)
                    sessions[client_id]["inference_tasks"].append(future)

                    logger.info("[%s] %s detection started in a separate thread", client_id, stream_type)

                except Exception:
                    logger.exception("[%s] Failed to start %s stream", client_id, stream_type)

            elif action == "stop_stream":
                sessions[client_id]["streaming"] = False
                for task in sessions[client_id]["inference_tasks"]:
                    task.cancel()  # best effort; the detection function should check streaming flag
                sessions[client_id]["inference_tasks"] = []
                logger.info("[%s] %s inference tasks stopped", client_id, stream_type)

    except Exception:
        logger.exception("[%s] Unexpected error in %s WebSocket", client_id, stream_type)

    finally:
        sessions[client_id]["streaming"] = False
        for task in sessions[client_id].get("inference_tasks", []):
            task.cancel()
        sessions.pop(client_id, None)
        logger.info("[%s] %s session cleaned up", client_id, stream_type)
