"""
Student Cognitive Burnout & Ergonomic Fatigue Analyzer.
Tracks sedentary inertia, PERCLOS eye fatigue, slouching duration, and prompts breaks.
"""

import time
from dataclasses import dataclass

@dataclass
class FatigueReport:
    burnout_risk_pct: float       # 0% to 100%
    burnout_tier: str             # LOW, MODERATE, HIGH, CRITICAL
    continuous_sitting_mins: float
    slouch_accumulated_mins: float
    break_recommended: bool
    ergonomic_advice: str

class FatigueAnalyzer:
    def __init__(self, break_reminder_interval_mins=45.0):
        self.break_interval = break_reminder_interval_mins
        self.sitting_start_time = time.time()
        self.last_posture = "SITTING"
        self.total_slouch_seconds = 0.0
        self.last_update_time = time.time()
        self.standing_recovery_time = 0.0

    def update(self, posture: str, is_slouching: bool, perclos: float, rrv_stress: str) -> FatigueReport:
        now = time.time()
        dt = max(0.0, now - self.last_update_time)
        self.last_update_time = now

        # Posture transition tracking
        if posture == "STANDING":
            self.standing_recovery_time += dt
            # If user stands for > 60 seconds, reset continuous sitting counter
            if self.standing_recovery_time > 60.0:
                self.sitting_start_time = now
        else:
            self.standing_recovery_time = 0.0

        if is_slouching:
            self.total_slouch_seconds += dt

        sitting_mins = max(0.0, (now - self.sitting_start_time) / 60.0)
        slouch_mins = self.total_slouch_seconds / 60.0

        # --- Composite Burnout Score Calculation ---
        # 1. Sedentary component (up to 40 pts for > 50 mins sitting)
        sedentary_score = min(40.0, (sitting_mins / self.break_interval) * 35.0)
        
        # 2. Eye Strain / PERCLOS component (up to 30 pts)
        eye_score = min(30.0, perclos * 100.0 * 1.5)
        
        # 3. Slouch component (up to 15 pts)
        slouch_score = min(15.0, (slouch_mins / 10.0) * 15.0)
        
        # 4. Stress component (up to 15 pts)
        stress_score = 15.0 if rrv_stress == "ELEVATED_STRESS" else (5.0 if rrv_stress == "FOCUSED_FLOW" else 0.0)

        total_risk = min(100.0, sedentary_score + eye_score + slouch_score + stress_score)

        if total_risk >= 75.0 or sitting_mins >= self.break_interval:
            tier = "CRITICAL" if total_risk >= 85 else "HIGH"
            break_rec = True
            advice = "TAKE A 5-MIN BREAK: Stand up, hydrate, and rest your eyes (20-20-20 rule)."
        elif total_risk >= 45.0:
            tier = "MODERATE"
            break_rec = False
            advice = "CHECK POSTURE: Straighten spine, relax shoulders, and blink frequently."
        else:
            tier = "LOW"
            break_rec = False
            advice = "OPTIMAL STATE: Good ergonomics & focused attention."

        return FatigueReport(
            burnout_risk_pct=round(total_risk, 1),
            burnout_tier=tier,
            continuous_sitting_mins=round(sitting_mins, 1),
            slouch_accumulated_mins=round(slouch_mins, 1),
            break_recommended=break_rec,
            ergonomic_advice=advice
        )
