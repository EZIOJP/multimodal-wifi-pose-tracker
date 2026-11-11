"""
FastAPI + WebSocket Server for 3D Digital Twin Web Visualizer.
Streams real-time 3D skeletal landmarks (Camera OR WiFi CSI DensePose) to Three.js web clients.
"""

import os
import sys
import json
import asyncio
import cv2
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from activity_tracker.ruview_bridge import RuViewReceiver
from activity_tracker.vision_engine import VisionTracker
from activity_tracker.fusion_engine import ActivityFusionEngine
from activity_tracker.calibrator import CalibrationProfile
from activity_tracker.temporal_buffer import TemporalSequenceBuffer

app = FastAPI(title="3D Spatial Digital Twin Server")

connected_clients = set()
latest_telemetry = {}

@app.get("/")
async def get_index():
    index_path = os.path.join(current_dir, "web", "index.html")
    return FileResponse(index_path)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            if latest_telemetry:
                await websocket.send_text(json.dumps(latest_telemetry))
            await asyncio.sleep(0.033)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
    except Exception:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

async def tracking_loop():
    global latest_telemetry
    profile = CalibrationProfile.load()
    rf_receiver = RuViewReceiver(udp_port=8888)
    rf_receiver.start()

    models_dir = os.path.join(current_dir, "models")
    vision_tracker = VisionTracker(models_dir=models_dir, profile=profile)
    temporal_buffer = TemporalSequenceBuffer(window_size=45)
    fusion_engine = ActivityFusionEngine()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            now = time.time()
            ret, frame = cap.read() if cap.isOpened() else (False, None)
            
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                v_metrics, _ = vision_tracker.process_frame(frame)
                
                temporal_buffer.push(
                    timestamp=now,
                    shoulder_y=v_metrics.shoulder_y,
                    hand_mouth_dist=v_metrics.hand_to_mouth_dist,
                    eyes_closed=v_metrics.eyes_closed,
                    head_yaw=v_metrics.head_yaw,
                    head_pitch=v_metrics.head_pitch,
                    is_present=v_metrics.is_present
                )
                temp_feat = temporal_buffer.extract_features()
                
                rf_receiver.inject_vision_hint(
                    is_present=v_metrics.is_present,
                    is_standing=(v_metrics.posture_state == "STANDING"),
                    activity=v_metrics.primary_activity
                )
            else:
                v_metrics = None
                temp_feat = None
                rf_receiver.inject_vision_hint(is_present=False, is_standing=False, activity="AWAY")

            rf_telemetry = rf_receiver.get_telemetry()
            
            # Fuse
            if v_metrics:
                fused = fusion_engine.fuse(v_metrics, rf_telemetry, temp_feat)
            else:
                from activity_tracker.vision_engine import VisionMetrics
                fused = fusion_engine.fuse(VisionMetrics(), rf_telemetry, None)

            # Determine 3D landmarks source
            lm_list = []
            cloud_list = []
            tracking_source = "CAMERA"
            
            if v_metrics and v_metrics.is_present and v_metrics.world_landmarks:
                tracking_source = "OPTICAL CAMERA (In FOV)"
                for lm in v_metrics.world_landmarks:
                    lm_list.append({"x": round(lm.x, 3), "y": round(lm.y, 3), "z": round(lm.z, 3)})
            elif rf_telemetry.dense_pose and rf_telemetry.dense_pose.landmarks:
                tracking_source = "WiFi CSI DensePose (Camera: Out of FOV)"
                for lm in rf_telemetry.dense_pose.landmarks:
                    lm_list.append({"x": round(lm.x, 3), "y": round(lm.y, 3), "z": round(lm.z, 3)})
                for pt in rf_telemetry.dense_pose.rf_point_cloud[:35]:
                    cloud_list.append([round(pt[0], 2), round(pt[1], 2), round(pt[2], 2)])

            elev = v_metrics.posture_height_pct if v_metrics and v_metrics.is_present else (100.0 if "STAND" in fused.posture else 0.0)

            latest_telemetry = {
                "activity": fused.activity_name,
                "posture": fused.posture,
                "elevation": elev,
                "attention": fused.attention_index,
                "respiration": fused.vitals_respiration_rpm,
                "burnout": fused.burnout_risk_pct,
                "sitting_mins": fused.continuous_sitting_mins,
                "source": tracking_source,
                "landmarks": lm_list,
                "rf_cloud": cloud_list
            }

            await asyncio.sleep(0.033)
    finally:
        if cap.isOpened():
            cap.release()
        rf_receiver.stop()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(tracking_loop())

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  3D SPATIAL DIGITAL TWIN WEB SERVER RUNNING")
    print("  Open in browser: http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run("activity_tracker.server_3d:app", host="0.0.0.0", port=8000, reload=False)
