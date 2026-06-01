import os
import cv2
import numpy as np
import base64
import time
import logging
from PIL import Image

logger = logging.getLogger("s3_utils_gundetections")

# Lazy-initialised so that missing AWS credentials don't crash the server at
# import time — only the storage thread fails, inference keeps running.
_s3_client = None
_S3_BUCKET  = None


def _get_s3():
    """Return a cached boto3 S3 client, creating it on first call."""
    global _s3_client, _S3_BUCKET
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client("s3")
        _S3_BUCKET = os.getenv("S3_BUCKET", "gun-detections")
    return _s3_client, _S3_BUCKET


def upload_to_s3(frame, frame_num: int) -> str:
    """
    Upload an annotated frame to S3 and return its public URL.

    Accepts:
        frame – base64 string, PIL Image, or BGR numpy array
        frame_num – frame counter used in the S3 key

    Returns:
        str – S3 URL of the uploaded object
    """
    if frame is None:
        raise ValueError(f"Frame {frame_num} is None, cannot upload")

    # ── Normalise to numpy BGR array ──────────────────────────────────────
    if isinstance(frame, Image.Image):
        frame = np.array(frame.convert("RGB"))[:, :, ::-1]  # RGB→BGR

    elif isinstance(frame, str):
        try:
            b64 = frame.split(",", 1)[-1] if "," in frame else frame
            img_bytes = base64.b64decode(b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as exc:
            raise ValueError(f"Frame {frame_num}: invalid base64 image — {exc}") from exc

    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Frame {frame_num}: expected ndarray, got {type(frame)}")

    # ── Encode and upload ─────────────────────────────────────────────────
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError(f"Frame {frame_num}: cv2.imencode failed")

    s3, bucket = _get_s3()
    key = f"gun-detections/frame_{frame_num}_{int(time.time())}.jpg"

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.tobytes(),
        ContentType="image/jpeg",
    )

    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    logger.info("Frame %d uploaded → %s", frame_num, url)
    return url
