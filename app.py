from fastapi import FastAPI, WebSocket
from src.websocket.gun_detection_websocket import run_gun_detection_detection
from src.handlers.gun_handler import gun_websocket_handler
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Session Stores ----------------
gun_sessions = {}


detection_executor = ThreadPoolExecutor(max_workers=10)
storage_executor = ThreadPoolExecutor(max_workers=5)


#--------------------------------------------------------------------------- WebSocket for all Models ------------------------------------------------------------------------------#



# ---------------- PPE WebSocket ----------------
@app.websocket("/ws/gundetection/{client_id}")
async def websocket_ppe(ws: WebSocket,client_id: str):
    await gun_websocket_handler(detection_executor, storage_executor, ws,client_id, gun_sessions, run_gun_detection_detection, "Gun_Detection")



