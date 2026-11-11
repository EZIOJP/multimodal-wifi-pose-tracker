"""
Real Windows WiFi RSSI Fingerprinting & Multi-Room Scanner.
Directly interfaces with the laptop's wireless network adapter (e.g. MediaTek Wi-Fi 6)
to measure real-time RSSI (dBm), signal %, and BSSID for genuine indoor room localization.
"""

import subprocess
import re
import time
import threading
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(CURRENT_DIR, "config")
WIFI_ROOMS_FILE = os.path.join(CONFIG_DIR, "real_wifi_rooms.json")

@dataclass
class RealWiFiMetric:
    connected: bool
    ssid: str
    bssid: str
    rssi_dbm: int
    signal_pct: int
    estimated_room: str
    room_confidence: float

class RealWiFiScanner:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        
        # Real-time hardware measurement
        self.latest = RealWiFiMetric(
            connected=False,
            ssid="Disconnected",
            bssid="",
            rssi_dbm=-99,
            signal_pct=0,
            estimated_room="Office / Desk (Default)",
            room_confidence=0.85
        )
        
        # Room RSSI Fingerprint Profile
        self.room_profiles: Dict[str, dict] = self._load_or_default_profiles()
        self._thread = None

    def _load_or_default_profiles(self) -> Dict[str, dict]:
        if os.path.exists(WIFI_ROOMS_FILE):
            try:
                with open(WIFI_ROOMS_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
                
        # Realistic default attenuation brackets based on typical home router distances
        return {
            "office": {
                "name": "Home Office / Desk",
                "min_rssi": -55,
                "max_rssi": -30,
                "target_rssi": -48,
                "x": 0.0, "z": 0.0
            },
            "hallway": {
                "name": "Central Hallway",
                "min_rssi": -65,
                "max_rssi": -56,
                "target_rssi": -60,
                "x": 0.0, "z": 2.2
            },
            "living_room": {
                "name": "Living Room",
                "min_rssi": -76,
                "max_rssi": -66,
                "target_rssi": -70,
                "x": 2.8, "z": 2.5
            },
            "kitchen": {
                "name": "Kitchen / Far Room",
                "min_rssi": -99,
                "max_rssi": -77,
                "target_rssi": -82,
                "x": -2.8, "z": 2.4
            }
        }

    def save_profiles(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(WIFI_ROOMS_FILE, "w") as f:
            json.dump(self.room_profiles, f, indent=4)
        print(f"[Real WiFi Scanner] Saved room profiles to {WIFI_ROOMS_FILE}")

    def calibrate_current_room(self, room_id: str, room_name: Optional[str] = None):
        """Calibrates the current physical location using the live WiFi signal"""
        metrics = self.poll_hardware()
        if not metrics.connected:
            print("[WiFi Error] Cannot calibrate: WiFi adapter not connected!")
            return False

        with self.lock:
            cur_rssi = metrics.rssi_dbm
            cur_sig = metrics.signal_pct
            
            if room_id not in self.room_profiles:
                self.room_profiles[room_id] = {
                    "name": room_name or room_id.replace("_", " ").title(),
                    "min_rssi": cur_rssi - 6,
                    "max_rssi": cur_rssi + 6,
                    "target_rssi": cur_rssi,
                    "x": 0.0, "z": 0.0
                }
            else:
                self.room_profiles[room_id]["target_rssi"] = cur_rssi
                self.room_profiles[room_id]["min_rssi"] = cur_rssi - 6
                self.room_profiles[room_id]["max_rssi"] = cur_rssi + 6
                if room_name:
                    self.room_profiles[room_id]["name"] = room_name

            self.save_profiles()
            print(f"[Calibrated] Room '{self.room_profiles[room_id]['name']}' locked to RSSI: {cur_rssi} dBm (Signal {cur_sig}%)")
            return True

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def poll_hardware(self) -> RealWiFiMetric:
        """Runs Windows netsh command to query real WiFi adapter status"""
        try:
            cmd = "netsh wlan show interfaces"
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore')
            
            rssi_m = re.search(r'Rssi\s*:\s*(-?\d+)', out)
            sig_m = re.search(r'Signal\s*:\s*(\d+)%', out)
            bssid_m = re.search(r'AP BSSID\s*:\s*([0-9a-fA-F:]+)', out)
            ssid_m = re.search(r'SSID\s*:\s*([^\r\n]+)', out)
            
            if rssi_m and sig_m:
                rssi = int(rssi_m.group(1))
                sig = int(sig_m.group(1))
                ssid = ssid_m.group(1).strip() if ssid_m else "WiFi Connected"
                bssid = bssid_m.group(1).strip() if bssid_m else ""
                
                # Match to closest room profile by RSSI distance
                best_room = "Office / Desk"
                best_diff = 999
                
                for r_id, prof in self.room_profiles.items():
                    diff = abs(rssi - prof["target_rssi"])
                    if diff < best_diff:
                        best_diff = diff
                        best_room = prof["name"]

                conf = max(0.5, 1.0 - (best_diff / 25.0))
                
                return RealWiFiMetric(
                    connected=True,
                    ssid=ssid,
                    bssid=bssid,
                    rssi_dbm=rssi,
                    signal_pct=sig,
                    estimated_room=best_room,
                    room_confidence=round(conf, 2)
                )
        except Exception:
            pass

        return RealWiFiMetric(
            connected=False,
            ssid="No Hardware Link",
            bssid="",
            rssi_dbm=-99,
            signal_pct=0,
            estimated_room="Office / Desk (Fallback)",
            room_confidence=0.5
        )

    def _monitor_loop(self):
        """Polls every 1.5 seconds in background"""
        while self.running:
            metric = self.poll_hardware()
            with self.lock:
                self.latest = metric
            time.sleep(1.5)

    def get_latest(self) -> RealWiFiMetric:
        with self.lock:
            return self.latest
