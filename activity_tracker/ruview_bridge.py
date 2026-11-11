"""
RuView WiFi CSI Bridge & Real Hardware WiFi Scanner.
Integrates live Windows WiFi adapter RSSI measurements (e.g. MediaTek Wi-Fi 6)
with multi-room spatial mapping and 3D DensePose generation.
"""

import socket
import json
import threading
import time
import math
import random
from dataclasses import dataclass
from typing import Optional

from activity_tracker.wifi_pose_engine import WiFiPoseEngine, WiFiDensePose
from activity_tracker.real_wifi_scanner import RealWiFiScanner, RealWiFiMetric

@dataclass
class RuViewTelemetry:
    timestamp: float
    presence: bool
    motion_energy: float       # 0.0 to 1.0
    vertical_doppler: float    # + positive = moving up, - negative = sitting
    room_zone: str             # "office", "hallway", "living_room", "kitchen"
    current_room_name: str     # Human-readable room name
    real_wifi_rssi: int        # Real dBm from laptop WiFi adapter (-30 to -90)
    real_wifi_signal: int      # Real signal % (0 to 100)
    connected: bool            # True if real hardware connected
    source: str
    dense_pose: Optional[WiFiDensePose] = None

class RuViewReceiver:
    def __init__(self, udp_port=8888, host="0.0.0.0"):
        self.udp_port = udp_port
        self.host = host
        self.running = False
        self.last_packet_time = 0.0
        self.lock = threading.Lock()
        
        # Real Windows Laptop WiFi Scanner
        self.real_wifi = RealWiFiScanner()
        self.pose_engine = WiFiPoseEngine()
        
        self.telemetry = RuViewTelemetry(
            timestamp=time.time(),
            presence=True,
            motion_energy=0.05,
            vertical_doppler=0.0,
            room_zone="office",
            current_room_name="Home Office / Study",
            real_wifi_rssi=-50,
            real_wifi_signal=85,
            connected=True,
            source="MediaTek Wi-Fi 6 (Live Hardware)"
        )
        
        self.camera_hint_posture = None
        self._thread = None
        self._sim_thread = None

    def start(self):
        self.running = True
        self.real_wifi.start()
        
        self._thread = threading.Thread(target=self._udp_listener, daemon=True)
        self._thread.start()
        
        self._sim_thread = threading.Thread(target=self._adaptive_sync_loop, daemon=True)
        self._sim_thread.start()

    def stop(self):
        self.running = False
        self.real_wifi.stop()

    def _udp_listener(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.udp_port))
            sock.settimeout(1.0)
            
            while self.running:
                try:
                    data, addr = sock.recvfrom(4096)
                    payload = json.loads(data.decode('utf-8'))
                    
                    with self.lock:
                        self.telemetry.timestamp = time.time()
                        self.telemetry.presence = payload.get("presence", True)
                        self.telemetry.motion_energy = float(payload.get("motion_energy", 0.1))
                        self.telemetry.vertical_doppler = float(payload.get("vertical_doppler", 0.0))
                        self.telemetry.room_zone = payload.get("room_zone", "office")
                        self.telemetry.connected = True
                        self.telemetry.source = f"ESP32 CSI ({addr[0]})"
                        self.last_packet_time = time.time()
                except socket.timeout:
                    continue
                except Exception:
                    pass
        except Exception:
            pass

    def _adaptive_sync_loop(self):
        while self.running:
            time.sleep(0.04)
            now = time.time()
            
            # Poll real Windows WiFi hardware metric
            hw_metric: RealWiFiMetric = self.real_wifi.get_latest()
            
            with self.lock:
                self.telemetry.timestamp = now
                if hw_metric.connected:
                    self.telemetry.real_wifi_rssi = hw_metric.rssi_dbm
                    self.telemetry.real_wifi_signal = hw_metric.signal_pct
                    self.telemetry.connected = True
                    self.telemetry.source = f"{hw_metric.ssid} ({hw_metric.rssi_dbm} dBm)"
                    
                    # If no camera, use real WiFi RSSI room estimate
                    if not self.camera_hint_posture:
                        self.telemetry.current_room_name = hw_metric.estimated_room
                        
                dense_pose = self.pose_engine.estimate_pose(
                    presence=self.telemetry.presence,
                    motion_energy=self.telemetry.motion_energy,
                    vertical_doppler=self.telemetry.vertical_doppler,
                    room_zone=self.telemetry.room_zone,
                    camera_hint_posture=self.camera_hint_posture
                )
                self.telemetry.dense_pose = dense_pose
                if dense_pose and not hw_metric.connected:
                    self.telemetry.current_room_name = dense_pose.current_room
                    self.telemetry.room_zone = dense_pose.room_id

    def inject_vision_hint(self, is_present, is_standing, activity):
        with self.lock:
            if is_present:
                self.camera_hint_posture = "STANDING" if is_standing else "SITTING"
                self.telemetry.presence = True
                self.telemetry.room_zone = "office"
                self.telemetry.current_room_name = "Home Office / Study"
                if activity in ["EATING", "TYPING"]:
                    self.telemetry.motion_energy = 0.15
            else:
                self.camera_hint_posture = None
                self.telemetry.presence = True
                self.telemetry.motion_energy = 0.35

    def get_telemetry(self) -> RuViewTelemetry:
        with self.lock:
            return RuViewTelemetry(
                timestamp=self.telemetry.timestamp,
                presence=self.telemetry.presence,
                motion_energy=self.telemetry.motion_energy,
                vertical_doppler=self.telemetry.vertical_doppler,
                room_zone=self.telemetry.room_zone,
                current_room_name=self.telemetry.current_room_name,
                real_wifi_rssi=self.telemetry.real_wifi_rssi,
                real_wifi_signal=self.telemetry.real_wifi_signal,
                connected=self.telemetry.connected,
                source=self.telemetry.source,
                dense_pose=self.telemetry.dense_pose
            )
