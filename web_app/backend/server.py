"""
FastAPI Server and WebSockets Broadcaster for SDM-EON Digital Twin Web App.
Runs inside WSL, listens on 0.0.0.0:8000, and is accessible from host Windows 11 browser.
"""

import os
import shutil
import json
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from web_app.backend.topology_parser import validate_and_parse_topology
from web_app.backend.trainer_runner import TrainingRunner
import torch

# Directory paths: --->
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPOLOGIES_DIR = os.path.join(BASE_DIR, "topologies")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "uploads", "checkpoints")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(TOPOLOGIES_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Copy nsfnet.json to topologies directory if not present: --->
root_nsfnet = os.path.join(os.path.dirname(BASE_DIR), "nsfnet.json")
topologies_nsfnet = os.path.join(TOPOLOGIES_DIR, "nsfnet.json")
if os.path.isfile(root_nsfnet) and not os.path.isfile(topologies_nsfnet):
    shutil.copy(root_nsfnet, topologies_nsfnet)

app = FastAPI(title="SDM-EON Digital Twin & RL Fault Management Web App")

# Connection Manager for WebSocket Telemetry: --->
class TelemetryBroadcaster:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.loop = None
        self.history_buffer = []
        self.max_history = 100

    def set_event_loop(self, loop):
        self.loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WebSocket] Client connected ({len(self.active_connections)} active)")
        # Send history sync on connect
        if self.history_buffer:
            await websocket.send_text(json.dumps({
                "event": "history_sync",
                "data": {"frames": self.history_buffer}
            }))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WebSocket] Client disconnected ({len(self.active_connections)} active)")

    def broadcast_sync(self, data: dict):
        """Thread-safe call from background runner to push data to asyncio WebSocket clients."""
        if self.loop and self.active_connections:
            asyncio.run_coroutine_threadsafe(self.broadcast(data), self.loop)

    async def broadcast(self, data: dict):
        if not self.active_connections:
            return
        
        # Buffer telemetry frames
        if data.get("event") == "telemetry_frame":
            self.history_buffer.append(data["data"])
            if len(self.history_buffer) > self.max_history:
                self.history_buffer.pop(0)

        msg = json.dumps(data)
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(msg)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

broadcaster = TelemetryBroadcaster()
runner = TrainingRunner(broadcast_callback=broadcaster.broadcast_sync)


@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_running_loop()
    broadcaster.set_event_loop(loop)
    print("=" * 70)
    print("  SDM-EON Digital Twin & RL Web Backend initialized!")
    print("  Server listening on http://0.0.0.0:8000")
    print("  Access from Windows 11 browser: http://localhost:8000")
    print("=" * 70)


@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await broadcaster.connect(websocket)
    try:
        # Send current state on connection: --->
        await websocket.send_text(json.dumps({
            "event": "session_status",
            "data": runner.get_state()
        }))
        while True:
            # Keep-alive loop: --->
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"event": "pong"}))
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)
    except Exception as e:
        print(f"[WebSocket Error] {e}")
        broadcaster.disconnect(websocket)


# ═══════════════════════════════════════════════════════════════════
# REST API Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def get_status():
    return runner.get_state()


@app.get("/api/topologies")
async def list_topologies():
    files = []
    # Check topologies dir and workspace root: --->
    search_paths = [TOPOLOGIES_DIR, os.path.dirname(BASE_DIR)]
    seen = set()

    for path in search_paths:
        if os.path.exists(path):
            for f in os.listdir(path):
                if f.endswith(".json") and f not in seen:
                    full_p = os.path.join(path, f)
                    is_valid, msg, graph = validate_and_parse_topology(full_p)
                    files.append({
                        "name": f,
                        "path": full_p,
                        "is_valid": is_valid,
                        "num_nodes": graph.get("num_nodes", 0) if is_valid else 0,
                        "num_edges": graph.get("num_edges", 0) if is_valid else 0
                    })
                    seen.add(f)

    return {"topologies": files}


@app.get("/api/topology_graph")
async def get_topology_graph(name: str):
    search_paths = [TOPOLOGIES_DIR, os.path.dirname(BASE_DIR)]
    for path in search_paths:
        full_p = os.path.join(path, name)
        if os.path.isfile(full_p):
            is_valid, msg, graph = validate_and_parse_topology(full_p)
            if is_valid:
                return {"graph": graph}
    raise HTTPException(status_code=404, detail="Topology not found")


@app.post("/api/upload_topology")
async def upload_topology(file: UploadFile = File(...)):
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="File must be a JSON file (.json)")

    dest_path = os.path.join(TOPOLOGIES_DIR, file.filename)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    is_valid, msg, graph = validate_and_parse_topology(dest_path)
    if not is_valid:
        os.remove(dest_path)
        raise HTTPException(status_code=400, detail=f"Invalid topology schema: {msg}")

    return {
        "status": "success",
        "message": f"Topology '{file.filename}' uploaded and validated successfully!",
        "filename": file.filename,
        "path": dest_path,
        "graph": graph
    }


@app.get("/api/checkpoints")
async def list_checkpoints():
    root_dir = os.path.dirname(BASE_DIR)
    search_dirs = [
        os.path.join(root_dir, "models", "CNN_PPO"),
        os.path.join(root_dir, "models", "GNN_PPO"),
        CHECKPOINTS_DIR
    ]

    cnn_ckpts = []
    gnn_ckpts = []

    for d in search_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".pt") or f.endswith(".pth"):
                    full_p = os.path.join(d, f)
                    item = {"name": f, "path": full_p, "dir": d}
                    if "cnn" in f.lower():
                        cnn_ckpts.append(item)
                    elif "gnn" in f.lower():
                        gnn_ckpts.append(item)
                    else:
                        cnn_ckpts.append(item)
                        gnn_ckpts.append(item)

    return {
        "cnn_checkpoints": cnn_ckpts,
        "gnn_checkpoints": gnn_ckpts
    }


@app.post("/api/upload_checkpoint")
async def upload_checkpoint(
    file: UploadFile = File(...),
    agent_type: str = Form(...)  # "cnn" or "gnn"
):
    if not (file.filename.endswith(".pt") or file.filename.endswith(".pth")):
        raise HTTPException(status_code=400, detail="File must be a PyTorch model checkpoint (.pt / .pth)")

    dest_filename = f"{agent_type}_{file.filename}"
    dest_path = os.path.join(CHECKPOINTS_DIR, dest_filename)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Validate model checkpoint structure: --->
    try:
        checkpoint = torch.load(dest_path, map_location=torch.device('cpu'))
        if not isinstance(checkpoint, dict) and not hasattr(checkpoint, 'state_dict'):
             raise ValueError("Uploaded file is not a valid PyTorch state dictionary.")
    except Exception as e:
        os.remove(dest_path)
        raise HTTPException(status_code=400, detail=f"Invalid PyTorch checkpoint: {str(e)}")

    return {
        "status": "success",
        "message": f"Checkpoint '{file.filename}' uploaded successfully for {agent_type.upper()} agent!",
        "path": dest_path,
        "agent_type": agent_type
    }


@app.post("/api/control/start")
async def start_session(config: dict):
    # Resolve topology path if relative name provided: --->
    topo_name = config.get("topology_path", "nsfnet.json")
    if not os.path.isabs(topo_name):
        topologies_path = os.path.join(TOPOLOGIES_DIR, topo_name)
        root_path = os.path.join(os.path.dirname(BASE_DIR), topo_name)
        if os.path.isfile(topologies_path):
            config["topology_path"] = topologies_path
        elif os.path.isfile(root_path):
            config["topology_path"] = root_path

    runner.start_session(config)
    return {"status": "started", "config": config}


@app.post("/api/control/pause")
async def pause_session():
    runner.pause_session()
    return {"status": "paused"}


@app.post("/api/control/resume")
async def resume_session():
    runner.resume_session()
    return {"status": "running"}


@app.post("/api/control/step")
async def step_session():
    runner.step_forward()
    return {"status": "running"}


@app.post("/api/control/stop")
async def stop_session():
    runner.stop_session()
    return {"status": "stopped"}


@app.post("/api/control/speed")
async def set_speed(payload: dict):
    step_delay_ms = payload.get("step_delay_ms", 100)
    runner.set_speed(step_delay_ms)
    return {"status": "updated", "step_delay_ms": step_delay_ms}


# Serve static web frontend: --->
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
