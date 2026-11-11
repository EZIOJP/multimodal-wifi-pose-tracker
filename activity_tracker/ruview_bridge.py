"""
RuView WiFi CSI Bridge & Receiver with Multi-Room Spatial Mapping.
Ingests WiFi CSI telemetry or synthesizes spatial trajectory tracking
across architectural house zones (Office, Hallway, Living Room, Kitchen).
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

@dataclass
class RuViewTelemetry:
    timestamp: float
    presence: bool
    motion_energy: float       # 0.0 (still) to 1.0 (running / violent movement)
    vertical_doppler: float    # + positive = moving up (standing), - negative = sitting
    room_zone: str             # "office", "hallway", "living_room", "kitchen"
    current_room_name: str     # Human-readable room name
    connected: bool            # True if actual hardware packets arriving
    source: str
    dense_pose: Optional[WiFiDensePose] = None

class RuViewReceiver:
    def __init__(self, udp_port=8888, host="0.0.0.0"):
        self.udp_port = udp_port
        self.host = host
        self.running = False
        self.last_packet_time = 0.0
        self.lock = threading.Lock()
        
        self.pose_engine = WiFiPoseEngine()
        
        self.telemetry = RuViewTelemetry(
            timestamp=time.time(),
            presence=True,
            motion_energy=0.05,
            vertical_doppler=0.0,
            room_zone="office",
            current_room_name="Home Office / Study",
            connected=False,
            source="initializing"
        )
        
        self.camera_hint_posture = None
        self._thread = None
        self._sim_thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._udp_listener, daemon=True)
        self._thread.start()
        
        self._sim_thread = threading.Thread(target=self._adaptive_sync_loop, daemon=True)
        self._sim_thread.start()

    def stop(self):
        self.running = False

    def _udp_listener(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.udp_port))
            sock.settimeout(1.0)
            print(f"[RuView Bridge] Listening for RuView CSI telemetry on UDP port {self.udp_port}...")
            
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
                        self.telemetry.source = f"hardware_{addr[0]}"
                        self.last_packet_time = time.time()
                except socket.timeout:
                    continue
                except Exception:
                    pass
        except Exception as e:
            print(f"[RuView Bridge] UDP socket warning: {e}. Running in adaptive CSI mode.")

    def _adaptive_sync_loop(self):
        while self.running:
            time.sleep(0.04)
            now = time.time()
            with self.lock:
                if now - self.last_packet_time > 3.0:
                    self.telemetry.timestamp = now
                    self.telemetry.connected = False
                    self.telemetry.source = "WiFi CSI House Mapper"
                    
                    dense_pose = self.pose_engine.estimate_pose(
                        presence=self.telemetry.presence,
                        motion_energy=self.telemetry.motion_energy,
                        vertical_doppler=self.telemetry.vertical_doppler,
                        room_zone=self.telemetry.room_zone,
                        camera_hint_posture=self.camera_hint_posture
                    )
                    self.telemetry.dense_pose = dense_pose
                    if dense_pose:
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
                # User stepped away from camera into room/hallway
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
                connected=self.telemetry.connected,
                source=self.telemetry.source,
                dense_pose=self.telemetry.dense_pose
            )
