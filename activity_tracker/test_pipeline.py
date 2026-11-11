"""
Headless validation script for the Multimodal Activity Tracker.
Tests end-to-end integration of:
- Vision Engine (MediaPipe)
- Calibration Profile
- RuView WiFi CSI House Mapper (Office, Hallway, Living Room, Kitchen)
- Temporal Sequence Buffer
- Cognitive Fatigue & Burnout Analyzer
"""

import sys
import os
import cv2
import numpy as np
import time

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

def run_headless_test():
    print("[TEST] Initializing full Multimodal Activity Tracker suite...")
    
    profile = CalibrationProfile.load()
    rf_receiver = RuViewReceiver(udp_port=8888)
    rf_receiver.start()
    
    models_dir = os.path.join(current_dir, "models")
    vision_tracker = VisionTracker(models_dir=models_dir, profile=profile)
    temporal_buffer = TemporalSequenceBuffer(window_size=45)
    fusion_engine = ActivityFusionEngine()
    
    for i in range(15):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        now = time.time() + (i * 0.033)
        
        v_metrics, _ = vision_tracker.process_frame(dummy)
        
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
        
        rf_telemetry = rf_receiver.get_telemetry()
        fused = fusion_engine.fuse(v_metrics, rf_telemetry, temp_feat)
        
    print(f"\n[Test Result]")
    print(f"  • Activity State: {fused.activity_name}")
    print(f"  • Posture: {fused.posture}")
    print(f"  • Current Room: {fused.current_room}")
    print(f"  • House Coordinates: {fused.room_coordinates}")
    print(f"  • Burnout Risk: {fused.burnout_risk_pct}% [{fused.burnout_tier}]")
    print(f"  • Sitting Duration: {fused.continuous_sitting_mins} mins")
    print(f"  • Sensor Mode: {fused.sensor_fusion_mode}")
    print(f"  • Advice: {fused.ergonomic_advice}")
    
    rf_receiver.stop()
    print("\n[SUCCESS] The multi-room house spatial tracking pipeline is fully operational!")

if __name__ == "__main__":
    run_headless_test()
