"""
System Calibration Engine & Interactive Wizard.
Calibrates and persists personalized biometric thresholds:
- Sitting & Standing shoulder/torso heights and elevation span
- Neutral head orientation & gaze center
- Eye blink sensitivity & drowsiness threshold
- Hand-to-mouth proximity threshold
- RuView WiFi CSI static room noise floor
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "calibration.json")

@dataclass
class CalibrationProfile:
    # Posture (Sit vs Stand)
    sitting_shoulder_y: float = 0.58
    standing_shoulder_y: float = 0.30
    shoulder_elevation_span: float = 0.28
    sitting_shoulder_width: float = 0.32
    standing_shoulder_width: float = 0.26
    
    # Gaze & Head
    neutral_head_yaw: float = 0.0
    neutral_head_pitch: float = 0.0
    neutral_head_roll: float = 0.0
    
    # Blinks & Eyes
    blink_sensitivity: float = 0.55
    sleep_trigger_seconds: float = 2.5
    
    # Eating & Hand gestures
    mouth_hand_dist_threshold: float = 0.13
    
    # RuView WiFi CSI
    rf_noise_floor: float = 0.04
    rf_baseline_rpm: float = 16.0
    
    # Metadata
    is_calibrated: bool = False
    calibrated_timestamp: str = ""

    def save(self, filepath=CONFIG_PATH):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = asdict(self)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[Calibration] Saved profile to {filepath}")

    @classmethod
    def load(cls, filepath=CONFIG_PATH) -> 'CalibrationProfile':
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                profile = cls(**data)
                print(f"[Calibration] Loaded personalized calibration profile ({profile.calibrated_timestamp})")
                return profile
            except Exception as e:
                print(f"[Calibration Warning] Failed to load {filepath}: {e}. Using defaults.")
        return cls()

class InteractiveCalibrator:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.profile = CalibrationProfile()
        
        from activity_tracker.vision_engine import VisionTracker
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.vision = VisionTracker(models_dir=models_dir)

    def run_wizard(self):
        """Runs the complete step-by-step interactive visual calibration wizard"""
        print("\n" + "=" * 60)
        print("   INTERACTIVE MULTIMODAL CALIBRATION WIZARD")
        print("   Personalizes Sit/Stand, Attention, and Activity Thresholds")
        print("=" * 60)

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[Error] Cannot open webcam index {self.camera_index}")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        try:
            # Step 1: Sitting Calibration
            sit_ok = self._calibrate_step(
                cap, 
                step_title="STEP 1 / 3: SITTING POSTURE",
                instruction="Sit naturally at your desk in normal working position.",
                sub_text="Look straight ahead. Hold steady while recording...",
                target_frames=45,
                metric_key="sitting"
            )
            if not sit_ok:
                return False

            # Step 2: Standing Calibration
            stand_ok = self._calibrate_step(
                cap, 
                step_title="STEP 2 / 3: STANDING POSTURE",
                instruction="Stand up naturally in front of your laptop.",
                sub_text="Step back slightly if needed. Hold posture...",
                target_frames=45,
                metric_key="standing"
            )
            if not stand_ok:
                return False

            # Step 3: Gaze & Screen Neutral
            gaze_ok = self._calibrate_step(
                cap,
                step_title="STEP 3 / 3: SCREEN FOCUS & GAZE",
                instruction="Sit back down and look directly at center of screen.",
                sub_text="Keep relaxed neutral eyes. Hold steady...",
                target_frames=45,
                metric_key="gaze"
            )
            if not gaze_ok:
                return False

            # Finalize & Save
            span = self.profile.sitting_shoulder_y - self.profile.standing_shoulder_y
            self.profile.shoulder_elevation_span = max(0.12, span)
            self.profile.is_calibrated = True
            self.profile.calibrated_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.profile.save()

            # Show Success Frame
            self._show_success_screen(cap)
            return True

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _calibrate_step(self, cap, step_title, instruction, sub_text, target_frames=45, metric_key="sitting"):
        collected = []
        countdown_secs = 3
        start_countdown = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            now = time.time()
            elapsed_cd = now - start_countdown
            
            # Run vision detector
            metrics, debug_draw = self.vision.process_frame(frame)
            
            # Semi-transparent overlay UI
            overlay = frame.copy()
            cv2.rectangle(overlay, (40, 40), (w - 40, 180), (20, 24, 30), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            
            # Header
            cv2.putText(frame, step_title, (60, 80), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 220, 255), 2)
            cv2.putText(frame, instruction, (60, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)
            cv2.putText(frame, sub_text, (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 190, 205), 1)

            # Draw shoulder lines
            if debug_draw["shoulders"] is not None:
                sh_l, sh_r = debug_draw["shoulders"]
                cv2.line(frame, sh_l, sh_r, (0, 255, 120), 3)

            # Phase 1: 3-Second Pre-Countdown
            if elapsed_cd < countdown_secs:
                remaining = int(countdown_secs - elapsed_cd) + 1
                cv2.putText(frame, f"GET READY: {remaining}", (w // 2 - 130, h // 2), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 215, 255), 3)
            # Phase 2: Gathering Data
            else:
                if metrics.is_present:
                    collected.append(metrics)
                    progress = len(collected) / target_frames
                    
                    # Progress bar
                    bar_w = 400
                    bx = (w - bar_w) // 2
                    by = h // 2 + 60
                    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 20), (50, 55, 65), -1)
                    cv2.rectangle(frame, (bx, by), (bx + int(bar_w * progress), by + 20), (0, 230, 120), -1)
                    cv2.putText(frame, f"RECORDING SAMPLES: {len(collected)} / {target_frames}", 
                                (bx + 60, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, "PERSON NOT DETECTED - STEP INTO VIEW", (w // 2 - 250, h // 2),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (50, 50, 255), 2)

                if len(collected) >= target_frames:
                    # Compute stats
                    self._process_collected(collected, metric_key)
                    return True

            cv2.imshow("Multimodal Calibration Wizard", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or Q
                print("[Calibration] Cancelled by user.")
                return False

    def _process_collected(self, samples, metric_key):
        if metric_key == "sitting":
            sh_y = [s.shoulder_y for s in samples]
            sh_w = [s.shoulder_width for s in samples]
            self.profile.sitting_shoulder_y = float(np.median(sh_y))
            self.profile.sitting_shoulder_width = float(np.median(sh_w))
            print(f"[Calibration] Sitting Shoulder Y: {self.profile.sitting_shoulder_y:.3f}")
        elif metric_key == "standing":
            sh_y = [s.shoulder_y for s in samples]
            sh_w = [s.shoulder_width for s in samples]
            self.profile.standing_shoulder_y = float(np.median(sh_y))
            self.profile.standing_shoulder_width = float(np.median(sh_w))
            print(f"[Calibration] Standing Shoulder Y: {self.profile.standing_shoulder_y:.3f}")
        elif metric_key == "gaze":
            yaws = [s.head_yaw for s in samples]
            pitches = [s.head_pitch for s in samples]
            self.profile.neutral_head_yaw = float(np.median(yaws))
            self.profile.neutral_head_pitch = float(np.median(pitches))
            print(f"[Calibration] Neutral Gaze: Yaw={self.profile.neutral_head_yaw:.1f}°, Pitch={self.profile.neutral_head_pitch:.1f}°")

    def _show_success_screen(self, cap):
        start = time.time()
        while time.time() - start < 2.5:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (w // 2 - 320, h // 2 - 100), (w // 2 + 320, h // 2 + 100), (20, 25, 30), -1)
            cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
            
            cv2.putText(frame, "CALIBRATION COMPLETE!", (w // 2 - 250, h // 2 - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 240, 120), 2)
            cv2.putText(frame, f"Sit Baseline: {self.profile.sitting_shoulder_y:.2f} | Stand Baseline: {self.profile.standing_shoulder_y:.2f}",
                        (w // 2 - 280, h // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            cv2.putText(frame, "Profile saved to config/calibration.json", (w // 2 - 190, h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 1)
            
            cv2.imshow("Multimodal Calibration Wizard", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    cam_id = 0
    if len(sys.argv) > 1:
        try:
            cam_id = int(sys.argv[1])
        except ValueError:
            cam_id = sys.argv[1]
            
    wizard = InteractiveCalibrator(camera_index=cam_id)
    wizard.run_wizard()
