"""
Vision Engine for Pose, Attention, Face Blendshapes, and Activity Recognition.
Uses MediaPipe Tasks (PoseLandmarker, FaceLandmarker, GestureRecognizer)
and integrates personalized Calibration Profiles.
"""

import os
import time
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from activity_tracker.calibrator import CalibrationProfile

@dataclass
class VisionMetrics:
    is_present: bool = False
    world_landmarks: Optional[List[Any]] = None
    
    # Posture (Sit vs Stand)
    posture_state: str = "UNKNOWN"       # SITTING, STANDING, TRANSITION_RISING, TRANSITION_LOWERING
    shoulder_y: float = 0.5              # 0.0 (top) to 1.0 (bottom)
    shoulder_width: float = 0.3
    posture_height_pct: float = 0.0      # 0% = sitting baseline, 100% = standing baseline
    
    # Attention & Vitals
    attention_score: float = 0.0         # 0.0 to 100.0%
    head_yaw: float = 0.0                # Degrees (-left, +right)
    head_pitch: float = 0.0              # Degrees (-down, +up)
    head_roll: float = 0.0               # Degrees
    eyes_closed: bool = False
    eye_closed_duration: float = 0.0     # Seconds
    is_yawning: bool = False
    jaw_open_score: float = 0.0
    
    # Hand & Action Detection
    hand_to_mouth_dist: float = 1.0      # Min distance from hand/wrist to mouth
    hand_near_mouth: bool = False
    hand_near_ear: bool = False
    hands_on_desk: bool = False
    recognized_gestures: List[str] = field(default_factory=list)
    
    # High-level Activity Candidate
    primary_activity: str = "NO_PERSON"
    confidence: float = 0.0

class VisionTracker:
    def __init__(self, models_dir: str = None, profile: Optional[CalibrationProfile] = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        
        self.models_dir = models_dir
        self.pose_model_path = os.path.join(models_dir, "pose_landmarker.task")
        self.face_model_path = os.path.join(models_dir, "face_landmarker.task")
        self.gesture_model_path = os.path.join(models_dir, "gesture_recognizer.task")
        
        # Load or assign profile
        self.profile = profile if profile is not None else CalibrationProfile.load()
        
        self._init_models()
        
        # Velocity & Smoothing state
        self.prev_shoulder_y = self.profile.sitting_shoulder_y
        self.shoulder_vel_y = 0.0
        self.prev_time = time.time()
        
        self.eyes_closed_start: Optional[float] = None
        self.chew_cycles = 0
        self.prev_jaw_open = 0.0
        self.smooth_attention = 50.0

    def _init_models(self):
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.pose_model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
        
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.face_model_path),
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            num_faces=1
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(face_options)
        
        gesture_options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=self.gesture_model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2
        )
        self.gesture_detector = vision.GestureRecognizer.create_from_options(gesture_options)

    def quick_calibrate_sitting(self, cur_shoulder_y: float):
        """Quickly updates the sitting baseline shoulder Y"""
        self.profile.sitting_shoulder_y = cur_shoulder_y
        self.profile.shoulder_elevation_span = max(0.12, self.profile.sitting_shoulder_y - self.profile.standing_shoulder_y)
        self.profile.is_calibrated = True
        self.profile.calibrated_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.profile.save()

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[VisionMetrics, dict]:
        now = time.time()
        dt = max(0.001, now - self.prev_time)
        self.prev_time = now
        
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        metrics = VisionMetrics()
        debug_draw = {
            "mouth": None,
            "hands": [],
            "shoulders": None,
            "gaze_vector": None
        }
        
        # --- 1. Pose Analysis (Sit vs Stand) ---
        pose_res = self.pose_detector.detect(mp_image)
        has_pose = bool(pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0)
        
        if has_pose:
            metrics.is_present = True
            if pose_res.pose_world_landmarks and len(pose_res.pose_world_landmarks) > 0:
                metrics.world_landmarks = pose_res.pose_world_landmarks[0]
            pl = pose_res.pose_landmarks[0]
            left_sh = pl[11]
            right_sh = pl[12]
            
            cur_shoulder_y = (left_sh.y + right_sh.y) / 2.0
            cur_width = abs(right_sh.x - left_sh.x)
            metrics.shoulder_y = cur_shoulder_y
            metrics.shoulder_width = cur_width
            
            sh_px_l = (int(left_sh.x * w), int(left_sh.y * h))
            sh_px_r = (int(right_sh.x * w), int(right_sh.y * h))
            debug_draw["shoulders"] = (sh_px_l, sh_px_r)
            
            # Vertical velocity calculation (negative = moving up)
            vel = (cur_shoulder_y - self.prev_shoulder_y) / dt
            self.shoulder_vel_y = 0.7 * self.shoulder_vel_y + 0.3 * vel
            self.prev_shoulder_y = cur_shoulder_y
            
            # Compute calibrated relative elevation height
            # 0% = sitting height, 100% = standing height
            elevation = self.profile.sitting_shoulder_y - cur_shoulder_y
            norm_height = np.clip(elevation / max(0.10, self.profile.shoulder_elevation_span), 0.0, 1.0)
            metrics.posture_height_pct = round(norm_height * 100.0, 1)
            
            # Classification
            if self.shoulder_vel_y < -0.40:
                metrics.posture_state = "TRANSITION_RISING"
            elif self.shoulder_vel_y > 0.40:
                metrics.posture_state = "TRANSITION_LOWERING"
            elif norm_height >= 0.50 or cur_shoulder_y < (self.profile.standing_shoulder_y + 0.08):
                metrics.posture_state = "STANDING"
            else:
                metrics.posture_state = "SITTING"

        # --- 2. Face, Blendshapes, and Attention ---
        face_res = self.face_detector.detect(mp_image)
        has_face = bool(face_res.face_landmarks and len(face_res.face_landmarks) > 0)
        mouth_center = None
        ear_left = None
        ear_right = None
        
        if has_face:
            metrics.is_present = True
            fl = face_res.face_landmarks[0]
            
            lip_top = fl[13]
            lip_bot = fl[14]
            mouth_center = np.array([(lip_top.x + lip_bot.x)/2.0, (lip_top.y + lip_bot.y)/2.0])
            debug_draw["mouth"] = (int(mouth_center[0] * w), int(mouth_center[1] * h))
            
            ear_right = np.array([fl[234].x, fl[234].y])
            ear_left = np.array([fl[454].x, fl[454].y])
            
            if face_res.face_blendshapes and len(face_res.face_blendshapes) > 0:
                bs_map = {category.category_name: category.score for category in face_res.face_blendshapes[0]}
                blink_l = bs_map.get("eyeBlinkLeft", 0.0)
                blink_r = bs_map.get("eyeBlinkRight", 0.0)
                jaw_open = bs_map.get("jawOpen", 0.0)
                metrics.jaw_open_score = jaw_open
                
                # Eye closure using calibrated sensitivity
                if blink_l > self.profile.blink_sensitivity and blink_r > self.profile.blink_sensitivity:
                    metrics.eyes_closed = True
                    if self.eyes_closed_start is None:
                        self.eyes_closed_start = now
                    metrics.eye_closed_duration = now - self.eyes_closed_start
                else:
                    metrics.eyes_closed = False
                    self.eyes_closed_start = None
                    metrics.eye_closed_duration = 0.0
                
                if jaw_open > 0.60:
                    metrics.is_yawning = True
                
                if abs(jaw_open - self.prev_jaw_open) > 0.12:
                    self.chew_cycles += 1
                self.prev_jaw_open = jaw_open

            # Head Pose (offset by calibrated neutral angles)
            nose_pt = fl[1]
            chin_pt = fl[152]
            forehead_pt = fl[10]
            left_eye_outer = fl[263]
            right_eye_outer = fl[33]
            
            dist_l = abs(nose_pt.x - left_eye_outer.x)
            dist_r = abs(nose_pt.x - right_eye_outer.x)
            total_eye_dist = max(0.001, dist_l + dist_r)
            yaw_ratio = (dist_l - dist_r) / total_eye_dist
            raw_yaw = yaw_ratio * 75.0
            metrics.head_yaw = round(raw_yaw - self.profile.neutral_head_yaw, 1)
            
            mid_eye_y = (left_eye_outer.y + right_eye_outer.y) / 2.0
            face_h = max(0.001, chin_pt.y - forehead_pt.y)
            nose_rel = (nose_pt.y - mid_eye_y) / face_h
            raw_pitch = (nose_rel - 0.38) * 90.0
            metrics.head_pitch = round(raw_pitch - self.profile.neutral_head_pitch, 1)
            
            # Calibrated Attention Score
            yaw_penalty = min(50.0, abs(metrics.head_yaw) * 1.6)
            pitch_penalty = min(40.0, max(0.0, abs(metrics.head_pitch) - 10.0) * 1.5)
            blink_penalty = 60.0 if metrics.eyes_closed else 0.0
            raw_attention = max(0.0, 100.0 - yaw_penalty - pitch_penalty - blink_penalty)
            self.smooth_attention = 0.85 * self.smooth_attention + 0.15 * raw_attention
            metrics.attention_score = round(self.smooth_attention, 1)

        # --- 3. Hand & Gesture Proximity ---
        gesture_res = self.gesture_detector.recognize(mp_image)
        min_mouth_dist = 999.0
        min_ear_dist = 999.0
        hands_y = []
        
        if gesture_res.hand_landmarks:
            for hand_lm in gesture_res.hand_landmarks:
                wrist = hand_lm[0]
                index_tip = hand_lm[8]
                hands_y.append(wrist.y)
                
                hand_pt = np.array([index_tip.x, index_tip.y])
                debug_draw["hands"].append((int(index_tip.x * w), int(index_tip.y * h)))
                
                if mouth_center is not None:
                    d_mouth = np.linalg.norm(hand_pt - mouth_center)
                    min_mouth_dist = min(min_mouth_dist, d_mouth)
                
                if ear_left is not None and ear_right is not None:
                    d_ear = min(np.linalg.norm(hand_pt - ear_left), np.linalg.norm(hand_pt - ear_right))
                    min_ear_dist = min(min_ear_dist, d_ear)
                    
            if gesture_res.gestures:
                for g_list in gesture_res.gestures:
                    for g in g_list:
                        if g.score > 0.6 and g.category_name != "None":
                            metrics.recognized_gestures.append(g.category_name)

        metrics.hand_to_mouth_dist = round(min_mouth_dist, 3)
        metrics.hand_near_mouth = (min_mouth_dist < self.profile.mouth_hand_dist_threshold)
        metrics.hand_near_ear = (min_ear_dist < 0.12)
        metrics.hands_on_desk = any(y > 0.65 for y in hands_y)

        # --- 4. Composite Activity Recognition ---
        if not metrics.is_present:
            metrics.primary_activity = "NO_PERSON"
            metrics.confidence = 0.95
        elif metrics.eye_closed_duration > self.profile.sleep_trigger_seconds or (metrics.eyes_closed and metrics.head_pitch < -25.0):
            metrics.primary_activity = "SLEEPING_DROWSY"
            metrics.confidence = 0.94
        elif metrics.hand_near_mouth and (metrics.jaw_open_score > 0.18 or self.chew_cycles > 2):
            metrics.primary_activity = "EATING_DRINKING"
            metrics.confidence = 0.90
            self.chew_cycles = max(0, self.chew_cycles - 1)
        elif metrics.hand_near_ear and abs(metrics.head_yaw) > 15.0:
            metrics.primary_activity = "PHONE_CALL"
            metrics.confidence = 0.86
        elif metrics.posture_state == "STANDING":
            metrics.primary_activity = "STANDING"
            metrics.confidence = 0.93
        elif metrics.posture_state.startswith("TRANSITION"):
            metrics.primary_activity = metrics.posture_state
            metrics.confidence = 0.85
        elif metrics.attention_score < 40.0:
            metrics.primary_activity = "DISTRACTED_LOOKING_AWAY"
            metrics.confidence = 0.84
        elif metrics.hands_on_desk or metrics.attention_score >= 60.0:
            metrics.primary_activity = "SITTING_WORKING"
            metrics.confidence = 0.92
        else:
            metrics.primary_activity = "SITTING_PASSIVE"
            metrics.confidence = 0.76

        return metrics, debug_draw
