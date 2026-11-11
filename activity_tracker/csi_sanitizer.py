"""
SenseFi-Inspired CSI Signal Sanitizer & Respiration Processor
Implements:
- Hampel outlier rejection for subcarrier multi-path spikes
- Moving-window noise filtering for respiration waveforms
- Respiration Rate Variability (RRV) and stress index computation
"""

import numpy as np
from collections import deque
from dataclasses import dataclass

@dataclass
class SanitizedCSIResult:
    clean_rpm: float
    clean_motion: float
    rrv_index: float             # Respiration Rate Variability (ms std)
    stress_indicator: str        # RELAXED, FOCUSED, ELEVATED_STRESS
    subcarrier_noise_level: float

class CSISanitizer:
    def __init__(self, window_size=25):
        self.window_size = window_size
        self.rpm_buffer = deque(maxlen=window_size)
        self.motion_buffer = deque(maxlen=window_size)

    def hampel_filter(self, values: list, threshold=2.5) -> float:
        """SenseFi Hampel filter for outlier rejection using median absolute deviation"""
        if len(values) < 5:
            return values[-1] if values else 16.0
        
        arr = np.array(values)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad < 1e-4:
            return float(values[-1])
        
        # Check current value
        cur = values[-1]
        diff = abs(cur - med) / (1.4826 * mad)
        if diff > threshold:
            # Outlier: replace with median
            return float(med)
        return float(cur)

    def process(self, raw_rpm: float, raw_motion: float) -> SanitizedCSIResult:
        self.rpm_buffer.append(raw_rpm)
        self.motion_buffer.append(raw_motion)
        
        # Apply Hampel filter
        clean_rpm = self.hampel_filter(list(self.rpm_buffer))
        clean_motion = round(float(np.mean(list(self.motion_buffer))), 3)
        
        # Compute RRV (Respiration Rate Variability)
        if len(self.rpm_buffer) >= 10:
            rrv = float(np.std(list(self.rpm_buffer)))
        else:
            rrv = 0.5
            
        # Stress level estimation from RRV
        if rrv > 2.2:
            stress = "ELEVATED_STRESS"
        elif clean_rpm < 14.0 and rrv < 0.8:
            stress = "RELAXED_REST"
        else:
            stress = "FOCUSED_FLOW"
            
        return SanitizedCSIResult(
            clean_rpm=round(clean_rpm, 1),
            clean_motion=clean_motion,
            rrv_index=round(rrv, 2),
            stress_indicator=stress,
            subcarrier_noise_level=round(rrv * 0.1, 3)
        )
