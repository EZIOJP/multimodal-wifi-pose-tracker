"""
Multimodal Fusion Engine (Multi-Room House Spatial Localization Edition)
Fuses Camera Vision with WiFi CSI Spatial House Mapping.
"""

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

from activity_tracker.vision_engine import VisionMetrics
from activity_tracker.ruview_bridge import RuViewTelemetry
from activity_tracker.temporal_buffer import TemporalFeatures
from activity_tracker.fatigue_analyzer import FatigueAnalyzer, FatigueReport

@dataclass
class FusedActivityState:
    timestamp: float
    
    # Unified Activity Label
    activity_name: str                  # WORKING, EATING, STANDING, SITTING, SLOUCHING, MOVING_BETWEEN_ROOMS
    activity_category: str              # PRODUCTIVITY, HEALTH, POSTURE, REST, ABSENT
    confidence: float                   # 0.0 to 1.0
    
    # Detailed Fused Metrics
    posture: str                        # SITTING, STANDING, TRANSITION_RISING, TRANSITION_LOWERING, WALKING
    attention_index: float              # 0 to 100%
    current_room: str                   # "Home Office / Study", "Living Room", "Hallway", "Kitchen"
    room_zone_id: str                   # "office", "hallway", "living_room", "kitchen"
    room_coordinates: str               # "(X, Z)"
    drowsiness_index: float             # 0 to 100%
    ergonomics_status: str              # OPTIMAL, SLOUCHING, BREAK_RECOMMENDED
    
    # Cognitive Fatigue & Burnout Metrics
    burnout_risk_pct: float             # 0% to 100%
    burnout_tier: str                   # LOW, MODERATE, HIGH, CRITICAL
    continuous_sitting_mins: float
    break_recommended: bool
    ergonomic_advice: str
    
    # System & Multimodal Diagnostics
    camera_active: bool
    ruview_connected: bool
    rf_presence: bool
    sensor_fusion_mode: str

class ActivityFusionEngine:
    def __init__(self):
        self.fatigue_analyzer = FatigueAnalyzer(break_reminder_interval_mins=45.0)
        self.last_seen_camera_time = time.time()
        self.standing_streak = 0
        self.sitting_streak = 0

    def fuse(self, v: VisionMetrics, rf: RuViewTelemetry, temporal: Optional[TemporalFeatures] = None) -> FusedActivityState:
        now = time.time()
        
        # Determine Sensor Modality
        if v.is_present and rf.connected:
            fusion_mode = "CAMERA + WiFi CSI House SLAM"
        elif v.is_present:
            fusion_mode = "CAMERA DOMINANT (Office Desk Zone)"
        elif rf.connected:
            fusion_mode = "WiFi CSI HOUSE SPATIAL TRACKING"
        else:
            fusion_mode = "WiFi CSI HOUSE MAPPING (Camera Out of FOV)"

        if v.is_present:
            self.last_seen_camera_time = now

        # Posture Fusion
        posture = "UNKNOWN"
        if v.is_present:
            if temporal and temporal.is_rising_trans:
                posture = "TRANSITION_RISING"
            elif temporal and temporal.is_lowering_trans:
                posture = "TRANSITION_LOWERING"
            elif v.posture_state == "STANDING":
                self.standing_streak += 1
                self.sitting_streak = 0
                posture = "STANDING" if self.standing_streak >= 2 else "TRANSITION_RISING"
            elif v.posture_state == "SITTING":
                self.sitting_streak += 1
                self.standing_streak = 0
                posture = "SITTING" if self.sitting_streak >= 2 else "TRANSITION_LOWERING"
            else:
                posture = v.posture_state
        else:
            if rf.dense_pose:
                posture = rf.dense_pose.posture
            elif rf.presence:
                posture = "WALKING" if rf.motion_energy > 0.20 else "RESTING"
            else:
                posture = "AWAY"

        # Ergonomics
        is_slouching = False
        ergonomics = "OPTIMAL"
        if posture == "SITTING" and v.is_present:
            if v.posture_height_pct < 5.0 and v.head_pitch < -12.0:
                ergonomics = "SLOUCHING"
                is_slouching = True

        perclos_val = temporal.perclos_score if temporal else (1.0 if v.eyes_closed else 0.0)
        fatigue_rep: FatigueReport = self.fatigue_analyzer.update(
            posture=posture,
            is_slouching=is_slouching,
            perclos=perclos_val,
            rrv_stress="FOCUSED_FLOW"
        )
        if fatigue_rep.break_recommended:
            ergonomics = "BREAK_RECOMMENDED"

        drowsiness = 0.0
        if v.eyes_closed and v.eye_closed_duration > 1.5:
            drowsiness = min(100.0, v.eye_closed_duration * 30.0)
        elif perclos_val > 0.35:
            drowsiness = min(90.0, perclos_val * 100.0)
        elif v.is_yawning:
            drowsiness = 65.0

        # Room location details
        cur_room = rf.dense_pose.current_room if rf.dense_pose else rf.current_room_name
        r_id = rf.dense_pose.room_id if rf.dense_pose else rf.room_zone
        coords = f"({rf.dense_pose.room_x:+.1f}m, {rf.dense_pose.room_z:+.1f}m)" if rf.dense_pose else "(0.0m, 0.0m)"

        # Final Activity Classification
        activity_name = "IDLE"
        activity_category = "GENERAL"
        conf = 0.88

        if v.eye_closed_duration > 2.8:
            activity_name = "SLEEPING"
            activity_category = "REST"
            conf = 0.95
        elif v.is_present and (
            (temporal and temporal.is_eating_motion) or 
            v.hand_near_mouth or 
            v.primary_activity == "EATING_DRINKING"
        ):
            activity_name = "EATING / DRINKING"
            activity_category = "HEALTH"
            conf = 0.93
        elif v.is_present and v.primary_activity == "PHONE_CALL":
            activity_name = "PHONE CALL / DEVICE"
            activity_category = "PRODUCTIVITY"
            conf = 0.88
        elif v.is_present and posture in ["STANDING", "TRANSITION_RISING"]:
            activity_name = "STANDING AT DESK"
            activity_category = "POSTURE"
            conf = 0.94
        elif v.is_present and posture in ["SITTING", "TRANSITION_LOWERING"]:
            if is_slouching:
                activity_name = "SLOUCHING AT DESK"
                activity_category = "POSTURE"
                conf = 0.89
            elif v.attention_score >= 60.0:
                activity_name = "WORKING (FOCUSED)"
                activity_category = "PRODUCTIVITY"
                conf = 0.94
            else:
                activity_name = "DESK IDLE / DISTRACTED"
                activity_category = "PRODUCTIVITY"
                conf = 0.82
        elif not v.is_present and rf.presence:
            # Traversed to another room in the house via WiFi CSI
            if rf.dense_pose and rf.dense_pose.posture == "WALKING":
                activity_name = f"WALKING IN {cur_room.upper()} (WiFi CSI)"
            elif rf.dense_pose and rf.dense_pose.posture == "STANDING":
                activity_name = f"STANDING IN {cur_room.upper()} (WiFi CSI)"
            else:
                activity_name = f"RESTING IN {cur_room.upper()} (WiFi CSI)"
            activity_category = "HOUSE_TRAVERSAL"
            conf = 0.90
        else:
            activity_name = "AWAY / HOUSE EMPTY"
            activity_category = "ABSENT"
            conf = 0.98

        return FusedActivityState(
            timestamp=now,
            activity_name=activity_name,
            activity_category=activity_category,
            confidence=round(conf, 2),
            posture=posture,
            attention_index=v.attention_score if v.is_present else 0.0,
            current_room=cur_room,
            room_zone_id=r_id,
            room_coordinates=coords,
            drowsiness_index=round(drowsiness, 1),
            ergonomics_status=ergonomics,
            burnout_risk_pct=fatigue_rep.burnout_risk_pct,
            burnout_tier=fatigue_rep.burnout_tier,
            continuous_sitting_mins=fatigue_rep.continuous_sitting_mins,
            break_recommended=fatigue_rep.break_recommended,
            ergonomic_advice=fatigue_rep.ergonomic_advice,
            camera_active=v.is_present,
            ruview_connected=rf.connected,
            rf_presence=rf.presence,
            sensor_fusion_mode=fusion_mode
        )
