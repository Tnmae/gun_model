import os
import json
import logging
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("detection")

# ─────────────────────────────────────────────────────────────────────────────
# Lazy connection pool — created on first use so that missing DB credentials
# don't crash the server at import time. Only the storage thread will fail.
# ─────────────────────────────────────────────────────────────────────────────
_pool = None


def _get_pool():
    """Return a cached psycopg2 connection pool, creating it on first call."""
    global _pool
    if _pool is not None:
        return _pool

    from psycopg2.pool import SimpleConnectionPool

    db_config = {
        "host":     os.getenv("DB_HOST"),
        "dbname":   os.getenv("DB_NAME"),
        "user":     os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port":     int(os.getenv("DB_PORT", 5432)),
    }

    missing = [k for k, v in db_config.items() if not v and k != "port"]
    if missing:
        raise RuntimeError(f"Missing DB environment variables: {missing}")

    _pool = SimpleConnectionPool(minconn=1, maxconn=15, **db_config)
    logger.info("✅ PostgreSQL connection pool created")
    return _pool


# ─────────────────────────────────────────────────────────────────────────────
# Insert
# ─────────────────────────────────────────────────────────────────────────────
_INSERT_SQL = """
    INSERT INTO gun_detections (
        cam_id, org_id, user_id,
        persons, guns, gun_holders,
        s3_url, status, timestamp
    )
    VALUES (
        %s, %s, %s,
        %s::jsonb, %s::jsonb, %s::jsonb,
        %s, %s, %s
    )
    RETURNING id;
"""


def insert_data(d: dict, s3_url: str) -> bool:
    """
    Insert a gun detection record into the gun_detections table.

    Returns True on success, False on failure (errors are logged, not raised,
    so a DB outage doesn't kill the inference loop).
    """
    conn = None
    try:
        pool = _get_pool()
        conn = pool.getconn()
        cursor = conn.cursor()

        timestamp_str = d.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except Exception:
                timestamp = datetime.utcnow()
        else:
            timestamp = datetime.utcnow()

        cursor.execute(
            _INSERT_SQL,
            (
                d.get("cam_id", -1),
                d.get("org_id", -1),
                d.get("user_id", -1),
                json.dumps(d.get("persons_present", [])),
                json.dumps(d.get("guns", [])),
                json.dumps(d.get("gun_holders", [])),
                s3_url,
                d.get("status", 0),
                timestamp,
            ),
        )

        inserted_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        logger.info(
            "✅ Inserted | cam=%s org=%s id=%s guns=%d alerts=%d ts=%s",
            d.get("cam_id"), d.get("org_id"), inserted_id,
            len(d.get("guns", [])), len(d.get("alerts", [])),
            timestamp_str,
        )
        return True

    except Exception as exc:
        if conn:
            conn.rollback()
        logger.error("❌ DB insert failed: %s | keys=%s", exc, list(d.keys()))
        return False

    finally:
        if conn:
            try:
                _get_pool().putconn(conn)
            except Exception:
                pass
