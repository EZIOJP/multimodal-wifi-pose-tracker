"""
Temporal Sequence Buffer (Inspired by PoseSense & Aditi1607 HAR)
Maintains a 30-60 frame FIFO window to analyze dynamic temporal trajectories:
- Hand approach-dwell-depart curves (Eating/Drinking vs Face touching)
- Shoulder elevation velocity profiles (Sit-to-Stand vs Stand-to-Sit)
- PERCLOS (Percentage of Eye Closure) and Blink Rate per minute
- Head nod / yaw variance over time
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class TemporalFeatures:
    # Eating trajectory
    is_eating_motion: bool = False
    eating_dwell_frames: int = 0
    
    # Sit / Stand dynamics
    is_rising_trans: bool = False
    is_lowering_trans: bool = False
    shoulder_vertical_delta: float = 0.0
    
    # Fatigue & PERCLOS
    perclos_score: float = 0.0          # Percentage of time eyes closed over window (0.0 to 1.0)
    blink_count_recent: int = 0         # Blinks in the recent buffer
    head_movement_variance: float = 0.0 # Low variance = static stare or sleep

class TemporalSequenceBuffer:
    def __init__(self, window_size=45):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        
        # State trackers
        self.hand_approaching = False
        self.hand_dwell_count = 0
        self.last_blink_state = False
        self.recent_blinks = deque(maxlen=100) # (timestamp)

    def push(self, timestamp: float, shoulder_y: float, hand_mouth_dist: float, 
             eyes_closed: bool, head_yaw: float, head_pitch: float, is_present: bool):
        if not is_present:
            self.history.clear()
            return

        frame_data = {
            "t": timestamp,
            "shoulder_y": shoulder_y,
            "dist": hand_mouth_dist,
            "eyes_closed": eyes_closed,
            "yaw": head_yaw,
            "pitch": head_pitch
        }
        self.history.append(frame_data)
        
        # Blink counter
        if eyes_closed and not self.last_blink_state:
            self.recent_blinks.append(timestamp)
        self.last_blink_state = eyes_closed

    def extract_features(self) -> TemporalFeatures:
        feat = TemporalFeatures()
        n = len(self.history)
        if n < 10:
            return feat
            
        now = time.time()
        
        # 1. PERCLOS Calculation (% of frames eyes were closed)
        closed_frames = sum(1 for f in self.history if f["eyes_closed"])
        feat.perclos_score = round(closed_frames / n, 2)
        
        # Clean expired blinks (> 60 seconds old)
        while self.recent_blinks and (now - self.recent_blinks[0]) > 60.0:
            self.recent_blinks.popleft()
        feat.blink_count_recent = len(self.recent_blinks)

        # 2. Shoulder Vertical Delta & Dynamic Velocity
        # Compare first 5 frames vs last 5 frames
        first_y = np.median([f["shoulder_y"] for f in list(self.history)[:5]])
        last_y = np.median([f["shoulder_y"] for f in list(self.history)[-5:]])
        delta_y = first_y - last_y  # Positive = moved UP (rising)
        feat.shoulder_vertical_delta = round(delta_y, 3)
        
        if delta_y > 0.10:
            feat.is_rising_trans = True
        elif delta_y < -0.10:
            feat.is_lowering_trans = True

        # 3. Eating / Drinking Trajectory (Approach -> Dwell -> Retreat)
        dists = [f["dist"] for f in self.history]
        min_dist = min(dists)
        recent_dist = dists[-1]
        
        # Hand near mouth
        if recent_dist < 0.14:
            self.hand_dwell_count += 1
        else:
            self.hand_dwell_count = max(0, self.hand_dwell_count - 1)
            
        feat.eating_dwell_frames = self.hand_dwell_count
        
        # A valid bite has an approach trajectory and dwells for 5 to 60 frames (~0.2s - 2.0s)
        if 4 <= self.hand_dwell_count <= 65 and min_dist < 0.12:
            feat.is_eating_motion = True

        # 4. Head movement variance
        yaws = [f["yaw"] for f in self.history]
        pitches = [f["pitch"] for f in self.history]
        feat.head_movement_variance = round(float(np.var(yaws) + np.var(pitches)), 2)

        return feat
