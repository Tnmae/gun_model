"""
test_websocket.py — WebSocket end-to-end test for gun detection.

Simulates exactly what a frontend client sends to the server.

Usage:
    # Terminal 1 — start the server
    .venv\\Scripts\\uvicorn app:app --host 0.0.0.0 --port 8004

    # Terminal 2 — run this test
    .venv\\Scripts\\python test_websocket.py

    # Optional overrides
    .venv\\Scripts\\python test_websocket.py --url rtsp://... --host localhost --port 8004
"""

import asyncio
import json
import time
import argparse
import statistics
import os
import sys

try:
    import websockets
except ImportError:
    print("[ERROR] websockets not installed. Run: pip install websockets")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_HOST      = "localhost"
DEFAULT_PORT      = 8004
DEFAULT_VIDEO_URL = os.path.join(os.path.dirname(__file__), "video.mp4")
DEFAULT_CLIENT_ID = "test-client-001"
DEFAULT_CAMERA_ID = 1
DEFAULT_ORG_ID    = 1
DEFAULT_USER_ID   = 1
TARGET_FPS        = 15.0


# ─────────────────────────────────────────────────────────────────────────────
# Test client
# ─────────────────────────────────────────────────────────────────────────────
async def run_test(host: str, port: int, video_url: str,
                   client_id: str, camera_id: int,
                   org_id: int, user_id: int,
                   max_frames: int = 0) -> None:

    uri = f"ws://{host}:{port}/ws/gundetection/{client_id}"
    print("=" * 70)
    print("  GUN DETECTION — WEBSOCKET TEST")
    print("=" * 70)
    print(f"  Server    : {uri}")
    print(f"  Stream    : {video_url}")
    print(f"  Camera ID : {camera_id}")
    print(f"  Max frames: {'all' if max_frames == 0 else max_frames}")
    print()

    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=30,
            max_size=10 * 1024 * 1024,   # 10 MB — large enough for annotated frames
        ) as ws:
            print(f"  [OK] Connected to {uri}")

            # ── Send start_stream ─────────────────────────────────────────
            start_msg = {
                "action":      "start_stream",
                "stream_name": video_url,
                "camera_id":   camera_id,
                "user_id":     user_id,
                "org_id":      org_id,
                "region":      "ap-south-1",
            }
            await ws.send(json.dumps(start_msg))
            print(f"  [>>] Sent start_stream\n")

            # ── Receive loop ──────────────────────────────────────────────
            frame_count   = 0
            total_guns    = 0
            total_alerts  = 0
            errors        = 0
            latencies_ms  = []   # round-trip: send → receive
            t_start       = time.perf_counter()
            t_last_frame  = t_start

            try:
                async for raw in ws:
                    t_recv = time.perf_counter()

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        print(f"\n  [WARN] Non-JSON message received")
                        continue

                    # ── Error from server ─────────────────────────────────
                    if msg.get("status") == "error":
                        print(f"\n  [ERROR] Server: {msg.get('message', msg)}")
                        errors += 1
                        break

                    # ── Detection payload ─────────────────────────────────
                    det = msg.get("detections", {})
                    if not det:
                        continue

                    if det.get("status", 0) != 0:
                        print(f"\n  [ERROR] Inference: {det.get('error', 'unknown')}")
                        errors += 1
                        continue

                    frame_count  += 1
                    frame_num     = det.get("frame_number", frame_count)
                    guns          = det.get("guns", [])
                    gun_holders   = det.get("gun_holders", [])
                    alerts        = det.get("alerts", [])
                    stats         = det.get("stats", {})

                    total_guns   += len(guns)
                    total_alerts += len(alerts)

                    # Round-trip latency (time since last frame received)
                    dt = (t_recv - t_last_frame) * 1000
                    latencies_ms.append(dt)
                    t_last_frame = t_recv

                    live_fps = frame_count / (t_recv - t_start)

                    # ── Print alerts immediately ──────────────────────────
                    if alerts:
                        print()
                        for a in alerts:
                            lvl  = a.get("level", "HIGH")
                            tag  = "[CRITICAL]" if lvl == "CRITICAL" else "[HIGH]   "
                            print(f"  {tag} ALERT  ID:{a.get('track_id')}  "
                                  f"conf:{a.get('confidence', 0):.2f}  "
                                  f"@ {a.get('timestamp', '')}")

                    # ── Live progress line ────────────────────────────────
                    print(f"\r  [F{frame_num:4d}] "
                          f"dt={dt:6.1f}ms  "
                          f"live={live_fps:5.1f}fps  "
                          f"guns={len(guns)}  "
                          f"armed={len(gun_holders)}  "
                          f"persons={stats.get('persons_tracked', 0)}",
                          end="", flush=True)

                    if max_frames and frame_count >= max_frames:
                        print(f"\n\n  [INFO] Reached max_frames={max_frames}, stopping.")
                        break

            except websockets.exceptions.ConnectionClosedOK:
                print("\n  [INFO] Server closed connection (stream ended).")
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"\n  [ERROR] Connection closed unexpectedly: {e}")

            # ── Send stop_stream ──────────────────────────────────────────
            try:
                await ws.send(json.dumps({"action": "stop_stream"}))
                print("  [>>] Sent stop_stream")
            except Exception:
                pass

            # ── Report ────────────────────────────────────────────────────
            wall = time.perf_counter() - t_start
            print("\n\n" + "=" * 70)
            print("  WEBSOCKET TEST REPORT")
            print("=" * 70)
            print(f"  Frames received  : {frame_count}")
            print(f"  Wall time        : {wall:.2f}s")
            print(f"  Errors           : {errors}")
            print(f"  Total guns det.  : {total_guns}")
            print(f"  Total alerts     : {total_alerts}")
            print()

            if latencies_ms and len(latencies_ms) > 1:
                steady = latencies_ms[1:]   # skip first (includes model load)
                p95    = sorted(steady)[int(len(steady) * 0.95)]
                avg_fps = 1000.0 / statistics.mean(steady)
                med_fps = 1000.0 / statistics.median(steady)

                print(f"  {'Metric':<22} {'min':>7} {'avg':>7} {'med':>7} {'max':>7} {'p95':>7}")
                print(f"  {'-'*57}")
                print(f"  {'frame interval (ms)':<22} "
                      f"{min(steady):>7.1f} "
                      f"{statistics.mean(steady):>7.1f} "
                      f"{statistics.median(steady):>7.1f} "
                      f"{max(steady):>7.1f} "
                      f"{p95:>7.1f}")
                print()
                print(f"  Throughput avg   : {avg_fps:6.2f} fps")
                print(f"  Throughput med   : {med_fps:6.2f} fps")
                print(f"  Target FPS       : {TARGET_FPS:.1f}")
                print()
                if med_fps >= TARGET_FPS:
                    print(f"  PASS -- {med_fps:.1f} fps median >= {TARGET_FPS} fps")
                else:
                    print(f"  BELOW TARGET -- {med_fps:.1f} fps median "
                          f"(need {TARGET_FPS - med_fps:.1f} more fps)")

            print("=" * 70)

    except ConnectionRefusedError:
        print(f"\n  [ERROR] Could not connect to {uri}")
        print(f"          Is the server running?")
        print(f"          Start it with:")
        print(f"          .venv\\Scripts\\uvicorn app:app --host 0.0.0.0 --port {port}")
    except Exception as e:
        print(f"\n  [ERROR] {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Gun detection WebSocket test client")
    parser.add_argument("--host",       default=DEFAULT_HOST,      help="Server host")
    parser.add_argument("--port",       default=DEFAULT_PORT,      type=int, help="Server port")
    parser.add_argument("--url",        default=DEFAULT_VIDEO_URL, help="Video file path or RTSP URL")
    parser.add_argument("--client-id",  default=DEFAULT_CLIENT_ID, help="WebSocket client ID")
    parser.add_argument("--camera-id",  default=DEFAULT_CAMERA_ID, type=int)
    parser.add_argument("--org-id",     default=DEFAULT_ORG_ID,    type=int)
    parser.add_argument("--user-id",    default=DEFAULT_USER_ID,   type=int)
    parser.add_argument("--max-frames", default=0, type=int,
                        help="Stop after N frames (0 = run to end)")
    args = parser.parse_args()

    asyncio.run(run_test(
        host       = args.host,
        port       = args.port,
        video_url  = args.url,
        client_id  = args.client_id,
        camera_id  = args.camera_id,
        org_id     = args.org_id,
        user_id    = args.user_id,
        max_frames = args.max_frames,
    ))


if __name__ == "__main__":
    main()
