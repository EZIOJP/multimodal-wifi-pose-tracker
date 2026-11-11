"""
Multimodal Fusion Engine (Temporal Sequence + CSI Sanitizer + Fatigue Analytics)
Fuses high-precision Camera Vision with RuView WiFi CSI Spatial & Vitals Intelligence.
"""

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

from activity_tracker.vision_engine import VisionMetrics
from activity_tracker.ruview_bridge import RuViewTelemetry
from activity_tracker.temporal_buffer import TemporalFeatures
from activity_tracker.csi_sanitizer import CSISanitizer, SanitizedCSIResult
from activity_tracker.fatigue_analyzer import FatigueAnalyzer, FatigueReport

@dataclass
class FusedActivityState:
    timestamp: float
    
    # Unified Activity Label
    activity_name: str                  # WORKING, EATING, SLEEPING, STANDING, SITTING, SLOUCHING, PACING, AWAY
    activity_category: str              # PRODUCTIVITY, HEALTH, POSTURE, REST, ABSENT
    confidence: float                   # 0.0 to 1.0
    
    # Detailed Fused Metrics
    posture: str                        # SITTING, STANDING, TRANSITION_RISING, TRANSITION_LOWERING
    attention_index: float              # 0 to 100%
    vitals_respiration_rpm: float       # Breaths/min
    drowsiness_index: float             # 0 to 100%
    ergonomics_status: str              # OPTIMAL, SLOUCHING, STRETCH_RECOMMENDED
    
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
        self.csi_sanitizer = CSISanitizer()
        self.fatigue_analyzer = FatigueAnalyzer(break_reminder_interval_mins=45.0)
        self.last_seen_camera_time = time.time()
        self.standing_streak = 0
        self.sitting_streak = 0

    def fuse(self, v: VisionMetrics, rf: RuViewTelemetry, temporal: Optional[TemporalFeatures] = None) -> FusedActivityState:
        now = time.time()
        
        # 1. Clean CSI via SenseFi Hampel filter
        clean_rf: SanitizedCSIResult = self.csi_sanitizer.process(rf.respiration_rpm, rf.motion_energy)
        
        # 2. Determine Sensor Modality
        if v.is_present and rf.connected:
            fusion_mode = "CAMERA_PLUS_WIFI (Full CSI Multimodal)"
        elif v.is_present:
            fusion_mode = "CAMERA_DOMINANT (Adaptive RF Sync)"
        elif rf.connected:
            fusion_mode = "RUVIEW_WIFI_ONLY (Through-Wall / Out-of-FOV)"
        else:
            fusion_mode = "OFFLINE_OR_EMPTY"

        if v.is_present:
            self.last_seen_camera_time = now

        # 3. Posture Fusion (Sit vs Stand) with Temporal Trajectory
        posture = "UNKNOWN"
        if v.is_present:
            # Check dynamic temporal trajectory first
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
            if rf.presence:
                posture = "STANDING_MOVING" if clean_rf.clean_motion > 0.20 else "RESTING_NEARBY"
            else:
                posture = "AWAY"

        # 4. Ergonomics check
        is_slouching = False
        ergonomics = "OPTIMAL"
        if posture == "SITTING" and v.is_present:
            if v.posture_height_pct < 5.0 and v.head_pitch < -12.0:
                ergonomics = "SLOUCHING"
                is_slouching = True

        # 5. Fatigue & Burnout Analysis
        perclos_val = temporal.perclos_score if temporal else (1.0 if v.eyes_closed else 0.0)
        fatigue_rep: FatigueReport = self.fatigue_analyzer.update(
            posture=posture,
            is_slouching=is_slouching,
            perclos=perclos_val,
            rrv_stress=clean_rf.stress_indicator
        )
        if fatigue_rep.break_recommended:
            ergonomics = "BREAK_RECOMMENDED"

        # 6. Drowsiness Score
        drowsiness = 0.0
        if v.eyes_closed and v.eye_closed_duration > 1.5:
            drowsiness = min(100.0, v.eye_closed_duration * 30.0)
        elif perclos_val > 0.35:
            drowsiness = min(90.0, perclos_val * 100.0)
        elif v.is_yawning:
            drowsiness = 65.0
        elif v.attention_score < 40.0 and clean_rf.clean_motion < 0.03:
            drowsiness = 40.0

        # 7. Final Activity Classification
        activity_name = "IDLE"
        activity_category = "GENERAL"
        conf = 0.85

        # Priority 1: Sleeping
        if v.eye_closed_duration > 2.8 or (drowsiness > 80.0 and clean_rf.clean_motion < 0.04):
            activity_name = "SLEEPING"
            activity_category = "REST"
            conf = 0.95

        # Priority 2: Eating & Drinking (verified with temporal trajectory)
        elif v.is_present and (
            (temporal and temporal.is_eating_motion) or 
            v.hand_near_mouth or 
            v.primary_activity == "EATING_DRINKING"
        ):
            activity_name = "EATING / DRINKING"
            activity_category = "HEALTH"
            conf = 0.93

        # Priority 3: Phone Call / Distraction
        elif v.is_present and v.primary_activity == "PHONE_CALL":
            activity_name = "PHONE CALL / DEVICE"
            activity_category = "PRODUCTIVITY"
            conf = 0.88

        # Priority 4: Standing & Moving
        elif posture in ["STANDING", "TRANSITION_RISING"]:
            activity_name = "STANDING"
            activity_category = "POSTURE"
            conf = 0.94

        # Priority 5: Sitting & Working (Desk Focus)
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

        # Priority 6: Out of camera FOV (RuView WiFi CSI)
        elif not v.is_present and rf.presence:
            if clean_rf.clean_motion > 0.25:
                activity_name = "PACING / ACTIVE IN ROOM (WiFi CSI)"
                activity_category = "GENERAL"
                conf = 0.87
            else:
                activity_name = "RESTING OUT OF CAMERA (WiFi CSI)"
                activity_category = "REST"
                conf = 0.82

        # Priority 7: Empty Room
        else:
            activity_name = "AWAY / ROOM EMPTY"
            activity_category = "ABSENT"
            conf = 0.98

        return FusedActivityState(
            timestamp=now,
            activity_name=activity_name,
            activity_category=activity_category,
            confidence=round(conf, 2),
            posture=posture,
            attention_index=v.attention_score if v.is_present else 0.0,
            vitals_respiration_rpm=clean_rf.clean_rpm,
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
