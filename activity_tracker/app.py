"""
Multimodal Human Activity, Posture (Sit/Stand), and Attention Recognition HUD
with Real-Time 3D Spatial Digital Twin Room Environment (Vision + WiFi CSI DensePose).
"""

import sys
import os
import cv2
import time
import math
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from activity_tracker.ruview_bridge import RuViewReceiver
from activity_tracker.vision_engine import VisionTracker
from activity_tracker.fusion_engine import ActivityFusionEngine
from activity_tracker.calibrator import CalibrationProfile, InteractiveCalibrator
from activity_tracker.temporal_buffer import TemporalSequenceBuffer
from activity_tracker.visualizer_3d import Spatial3DVisualizer

def draw_hud(frame, v_metrics, rf_telemetry, fused, debug_draw, fps, profile):
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    # 1. Top Header Bar
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 24, 30), -1)
    # 2. Left Stats Card
    cv2.rectangle(overlay, (15, 95), (290, 475), (20, 24, 30), -1)
    # 3. Bottom Info Bar
    cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 18, 22), -1)
    
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    
    # --- Top Header: Activity & Confidence ---
    act_name = fused.activity_name
    
    if "WORKING" in act_name:
        badge_color = (0, 230, 115)       # Bright Green
    elif "STANDING" in act_name:
        badge_color = (255, 180, 0)       # Cyan/Blue
    elif "EATING" in act_name:
        badge_color = (0, 215, 255)       # Amber Gold
    elif "SLEEPING" in act_name:
        badge_color = (50, 50, 255)       # Red
    elif "PACING" in act_name or "WiFi CSI" in act_name:
        badge_color = (0, 240, 255)       # Electric Cyan (WiFi DensePose)
    elif "SLOUCHING" in act_name:
        badge_color = (0, 140, 255)       # Orange
    else:
        badge_color = (180, 180, 180)     # Gray
        
    cv2.putText(frame, "ACTIVITY:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 170, 185), 1)
    cv2.putText(frame, act_name, (110, 32), cv2.FONT_HERSHEY_DUPLEX, 0.75, badge_color, 2)
    
    conf_str = f"CONFIDENCE: {int(fused.confidence * 100)}%"
    cv2.putText(frame, conf_str, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 230, 200), 1)
    
    mode_tag = "CAMERA IN FOV" if v_metrics.is_present else "WiFi CSI DENSEPOSE (Camera Blindspot)"
    mode_col = (0, 240, 120) if v_metrics.is_present else (0, 220, 255)
    cv2.putText(frame, f"TRACKER: {mode_tag}", (175, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.40, mode_col, 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1)

    # --- Left Card: Detailed Gauges ---
    y_card = 120
    
    # A. Posture Indicator (Sit vs Stand)
    cv2.putText(frame, "POSTURE (Sit / Stand)", (30, y_card), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    posture_text = fused.posture
    posture_color = (255, 200, 0) if "STAND" in posture_text else (0, 220, 100)
    cv2.putText(frame, posture_text, (30, y_card + 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, posture_color, 2)
    
    h_bar_x, h_bar_y, h_bar_w, h_bar_h = 30, y_card + 33, 230, 9
    cv2.rectangle(frame, (h_bar_x, h_bar_y), (h_bar_x + h_bar_w, h_bar_y + h_bar_h), (50, 55, 65), -1)
    fill_w = int(h_bar_w * (v_metrics.posture_height_pct / 100.0))
    cv2.rectangle(frame, (h_bar_x, h_bar_y), (h_bar_x + fill_w, h_bar_y + h_bar_h), posture_color, -1)
    cv2.putText(frame, f"Elevation: {v_metrics.posture_height_pct:.0f}%", 
                (h_bar_x, h_bar_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 170, 180), 1)

    # B. Attention Index Gauge
    y_card += 68
    cv2.putText(frame, "ATTENTION INDEX", (30, y_card), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    att_val = fused.attention_index
    att_color = (0, 240, 120) if att_val >= 60 else ((0, 180, 255) if att_val >= 35 else (60, 60, 255))
    cv2.putText(frame, f"{att_val:.1f}%", (30, y_card + 23), cv2.FONT_HERSHEY_DUPLEX, 0.65, att_color, 2)
    cv2.rectangle(frame, (30, y_card + 33), (260, y_card + 42), (50, 55, 65), -1)
    cv2.rectangle(frame, (30, y_card + 33), (30 + int(230 * (att_val / 100.0)), y_card + 42), att_color, -1)

    # C. Cognitive Burnout & Sedentary Fatigue
    y_card += 68
    cv2.putText(frame, "COGNITIVE BURNOUT RISK", (30, y_card), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    b_val = fused.burnout_risk_pct
    b_color = (60, 60, 255) if b_val >= 70 else ((0, 180, 255) if b_val >= 40 else (0, 230, 120))
    cv2.putText(frame, f"{b_val:.0f}% [{fused.burnout_tier}]", (30, y_card + 23), cv2.FONT_HERSHEY_DUPLEX, 0.62, b_color, 2)
    cv2.rectangle(frame, (30, y_card + 33), (260, y_card + 42), (50, 55, 65), -1)
    cv2.rectangle(frame, (30, y_card + 33), (30 + int(230 * (b_val / 100.0)), y_card + 42), b_color, -1)
    cv2.putText(frame, f"Seated: {fused.continuous_sitting_mins:.1f} mins", (165, y_card + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # D. Drowsiness / Sleep Tracker
    y_card += 68
    cv2.putText(frame, "DROWSINESS / SLEEP", (30, y_card), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    drow_val = fused.drowsiness_index
    drow_color = (60, 60, 255) if drow_val > 50 else (120, 200, 120)
    cv2.putText(frame, f"{drow_val:.0f}%", (30, y_card + 23), cv2.FONT_HERSHEY_DUPLEX, 0.62, drow_color, 2)
    if v_metrics.eyes_closed:
        cv2.putText(frame, f"Eyes Closed: {v_metrics.eye_closed_duration:.1f}s", (100, y_card + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 100, 255), 1)

    # E. RuView WiFi CSI Vitals & Room Coordinates
    y_card += 58
    cv2.putText(frame, "RUVIEW RF VITALS & DENSEPOSE", (30, y_card), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1)
    vitals_str = f"Resp: {fused.vitals_respiration_rpm} RPM"
    if rf_telemetry.dense_pose:
        vitals_str += f" | Room Pos: ({rf_telemetry.dense_pose.room_x:+.1f}m, {rf_telemetry.dense_pose.room_z:+.1f}m)"
    cv2.putText(frame, vitals_str, (30, y_card + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 240), 1)
    rf_status = "ONLINE [ESP32 CSI]" if rf_telemetry.connected else "WiFi CSI DensePose Active"
    cv2.putText(frame, rf_status, (30, y_card + 37), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 160, 180), 1)

    # Vision Landmarks Overlay
    if debug_draw["shoulders"] is not None:
        sh_l, sh_r = debug_draw["shoulders"]
        cv2.line(frame, sh_l, sh_r, posture_color, 3)
        mid_sh = ((sh_l[0] + sh_r[0]) // 2, (sh_l[1] + sh_r[1]) // 2)
        cv2.circle(frame, mid_sh, 5, (255, 255, 255), -1)

    if debug_draw["mouth"] is not None:
        m_pt = debug_draw["mouth"]
        m_color = (0, 215, 255) if v_metrics.hand_near_mouth else (120, 120, 120)
        cv2.circle(frame, m_pt, 8, m_color, 2)
        if v_metrics.hand_near_mouth:
            cv2.putText(frame, "EATING / DRINKING", (m_pt[0] + 15, m_pt[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 2)

    for hand_pt in debug_draw["hands"]:
        cv2.circle(frame, hand_pt, 6, (0, 255, 120), -1)

    hotkeys = "[V] Toggle 3D View   [A/D/W/S] Orbit 3D   [K] Calibrate Wizard   [C] Quick Height   [Q] Quit"
    cv2.putText(frame, hotkeys, (25, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 180, 200), 1)

def run_tracker(camera_index=0):
    print("=" * 65)
    print("  Multimodal Activity, Posture & Attention Recognition System")
    print("  Webcam Vision + RuView WiFi CSI DensePose + 3D Spatial Twin")
    print("=" * 65)
    
    profile = CalibrationProfile.load()
    
    print("[1/5] Initializing RuView WiFi CSI Bridge & DensePose Engine...")
    rf_receiver = RuViewReceiver(udp_port=8888)
    rf_receiver.start()
    
    print("[2/5] Loading MediaPipe Vision Models with Calibration Profile...")
    models_dir = os.path.join(current_dir, "models")
    vision_tracker = VisionTracker(models_dir=models_dir, profile=profile)
    
    print("[3/5] Initializing 45-frame Temporal Sequence Buffer...")
    temporal_buffer = TemporalSequenceBuffer(window_size=45)
    
    print("[4/5] Initializing Sensor Fusion Engine & Fatigue Analyzer...")
    fusion_engine = ActivityFusionEngine()
    
    print("[5/5] Initializing 3D Spatial Environment Engine...")
    visualizer_3d = Spatial3DVisualizer(width=640, height=720)
    
    view_mode = 0
    
    print(f"[Camera] Opening webcam index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[Camera Error] Could not access webcam index {camera_index}!")
        rf_receiver.stop()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    prev_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
                
            now = time.time()
            fps = 1.0 / max(0.001, now - prev_time)
            prev_time = now
            
            frame = cv2.flip(frame, 1)
            frame_resized = cv2.resize(frame, (640, 720))
            
            # Step A: Vision tracking
            v_metrics, debug_draw = vision_tracker.process_frame(frame)
            
            # Step B: Push frame into temporal buffer
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
            
            # Step C: Ground truth to RuView bridge
            rf_receiver.inject_vision_hint(
                is_present=v_metrics.is_present,
                is_standing=(v_metrics.posture_state == "STANDING"),
                activity=v_metrics.primary_activity
            )
            rf_telemetry = rf_receiver.get_telemetry()
            
            # Step D: Multimodal Fusion
            fused_state = fusion_engine.fuse(v_metrics, rf_telemetry, temp_feat)
            
            # Step E: Render 3D Spatial Digital Twin (Optical OR WiFi CSI DensePose)
            canvas_3d = visualizer_3d.render(
                optical_landmarks=v_metrics.world_landmarks,
                wifi_dense_pose=rf_telemetry.dense_pose,
                posture=fused_state.posture,
                activity=fused_state.activity_name,
                resp_rpm=fused_state.vitals_respiration_rpm,
                attention_score=fused_state.attention_index,
                is_camera_active=v_metrics.is_present
            )
            
            # Step F: Draw 2D Camera HUD
            draw_hud(frame_resized, v_metrics, rf_telemetry, fused_state, debug_draw, fps, vision_tracker.profile)
            
            if view_mode == 0:
                display_frame = np.hstack((frame_resized, canvas_3d))
            elif view_mode == 1:
                display_frame = cv2.resize(canvas_3d, (1280, 720))
            else:
                display_frame = cv2.resize(frame_resized, (1280, 720))
            
            cv2.imshow("Multimodal Activity Tracker (RuView + Vision + 3D Twin)", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('v') or key == ord('V'):
                view_mode = (view_mode + 1) % 3
                modes = ["Split-Screen Dual View", "3D Digital Twin Fullscreen", "Camera Feed Only"]
                print(f"[View Mode] Switched to: {modes[view_mode]}")
            elif key == ord('a') or key == ord('A'):
                visualizer_3d.set_camera_rotation(delta_yaw=-8.0)
            elif key == ord('d') or key == ord('D'):
                visualizer_3d.set_camera_rotation(delta_yaw=8.0)
            elif key == ord('w') or key == ord('W'):
                visualizer_3d.set_camera_rotation(delta_pitch=5.0)
            elif key == ord('s') or key == ord('S'):
                visualizer_3d.set_camera_rotation(delta_pitch=-5.0)
            elif key == ord('c') or key == ord('C'):
                if v_metrics.is_present:
                    vision_tracker.quick_calibrate_sitting(v_metrics.shoulder_y)
                    print(f"\n[Quick Calibrated] Sitting baseline locked to shoulder Y = {v_metrics.shoulder_y:.3f}!")
                else:
                    print("\n[Calibration Notice] Please sit in front of the camera to calibrate.")
            elif key == ord('k') or key == ord('K'):
                print("\n[Launching Wizard] Pausing tracker for Calibration Wizard...")
                cap.release()
                cv2.destroyAllWindows()
                
                wizard = InteractiveCalibrator(camera_index=camera_index)
                if wizard.run_wizard():
                    vision_tracker.profile = CalibrationProfile.load()
                    print("\n[Wizard Success] Tracker resumed with updated calibration profile.")
                
                cap = cv2.VideoCapture(camera_index)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rf_receiver.stop()
        print("[Done] System closed.")

if __name__ == "__main__":
    cam_id = 0
    if len(sys.argv) > 1:
        try:
            cam_id = int(sys.argv[1])
        except ValueError:
            cam_id = sys.argv[1]
    run_tracker(cam_id)
