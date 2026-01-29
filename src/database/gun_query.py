import os
import logging
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
import json
from datetime import datetime  # ← ADD THIS IMPORT

load_dotenv()
logger = logging.getLogger("detection")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

try:
    pool = SimpleConnectionPool(
        minconn=1,        
        maxconn=15,       
        **DB_CONFIG
    )
    logger.info("✅ PostgreSQL Connection Pool Created")
except Exception as e:
    logger.error(f"❌ Error creating connection pool: {e}")
    raise


def insert_data(d, s3_url):
    """
    Insert gun detection data into gun_detections table
    """
    conn = None
    try:
        conn = pool.getconn()
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO gun_detections (
                cam_id,
                org_id,
                user_id,
                persons,
                guns,
                gun_holders,
                s3_url,
                status,
                timestamp
            )
            VALUES (
                %s, %s, %s,
                %s::jsonb,
                %s::jsonb,
                %s::jsonb,
                %s, %s, %s
            )
            RETURNING id;
        """

        # Get timestamp from detection response
        timestamp_str = d.get("timestamp")
        
        if timestamp_str:
            # Parse ISO format timestamp from detection
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except Exception:
                # Fallback if parse fails
                timestamp = datetime.utcnow()
        else:
            # Fallback if timestamp missing
            timestamp = datetime.utcnow()

        cursor.execute(
            insert_query,
            (
                d.get("cam_id", -1),
                d.get("org_id", -1),
                d.get("user_id", -1),
                json.dumps(d.get("persons_present", [])),
                json.dumps(d.get("guns", [])),
                json.dumps(d.get("gun_holders", [])),
                s3_url,
                d.get("status", 0),
                timestamp
            )
        )

        inserted_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        logger.info(
            f"✅ Gun detection inserted | cam_id={d.get('cam_id')} | "
            f"org_id={d.get('org_id')} | id={inserted_id} | "
            f"guns={len(d.get('guns', []))} | alerts={len(d.get('alerts', []))} | "
            f"timestamp={timestamp_str}"
        )
        return True

    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"❌ DB insert failed: {e}")
        logger.error(f"   Data keys available: {list(d.keys())}")
        return False

    finally:
        if conn:
            pool.putconn(conn)