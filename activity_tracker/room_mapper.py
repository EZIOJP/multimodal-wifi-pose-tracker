"""
WiFi Multi-Room House Mapping & RF Spatial Floorplan Engine.
Loads auto-scanned real rooms from config/house_rooms.json, or provides baseline rooms.
"""

import os
import json
import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(CURRENT_DIR, "config")
ROOMS_JSON_PATH = os.path.join(CONFIG_DIR, "house_rooms.json")

@dataclass
class RoomZone:
    name: str
    id: str
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    color_hex: int
    bgr_color: Tuple[int, int, int]
    doors: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class SpatialHouseMapResult:
    current_room: str
    room_id: str
    room_confidence: float
    pos_x: float
    pos_z: float
    distance_to_nearest_wall: float
    is_auto_scanned: bool
    occupancy_heatmap: Dict[str, float]

class HouseRoomMapper:
    def __init__(self):
        self.rooms: Dict[str, RoomZone] = {}
        self.is_auto_scanned = False
        self.time_spent = {}
        self.last_update_time = time.time()
        self.current_room_id = "office"
        
        self.load_rooms()

    def load_rooms(self):
        """Loads auto-scanned rooms from JSON if available, otherwise sets default"""
        if os.path.exists(ROOMS_JSON_PATH):
            try:
                with open(ROOMS_JSON_PATH, "r") as f:
                    data = json.load(f)
                self.rooms = {}
                for r_id, r in data.items():
                    doors_list = [tuple(d) for d in r.get("doors", [])]
                    bgr = tuple(r.get("bgr_color", (255, 220, 0)))
                    self.rooms[r_id] = RoomZone(
                        name=r["name"],
                        id=r["id"],
                        x_min=float(r["x_min"]),
                        x_max=float(r["x_max"]),
                        z_min=float(r["z_min"]),
                        z_max=float(r["z_max"]),
                        color_hex=int(r.get("color_hex", 0x00e5ff)),
                        bgr_color=bgr,
                        doors=doors_list
                    )
                self.is_auto_scanned = True
                print(f"[House Mapper] Successfully loaded {len(self.rooms)} auto-scanned rooms from {ROOMS_JSON_PATH}")
            except Exception as e:
                print(f"[House Mapper Warning] Error reading {ROOMS_JSON_PATH}: {e}. Falling back to default layout.")
                self._load_default_rooms()
        else:
            self._load_default_rooms()

        self.time_spent = {r_id: 0.0 for r_id in self.rooms}
        self.current_room_id = list(self.rooms.keys())[0] if self.rooms else "office"

    def _load_default_rooms(self):
        self.is_auto_scanned = False
        self.rooms = {
            "office": RoomZone(
                name="Home Office / Study",
                id="office",
                x_min=-1.6, x_max=1.4,
                z_min=-1.2, z_max=1.5,
                color_hex=0x00e5ff,
                bgr_color=(255, 220, 0),
                doors=[(0.0, 1.5)]
            ),
            "hallway": RoomZone(
                name="Central Hallway",
                id="hallway",
                x_min=-1.6, x_max=1.4,
                z_min=1.5, z_max=3.2,
                color_hex=0x9d4edd,
                bgr_color=(220, 80, 160),
                doors=[(0.0, 1.5), (1.4, 2.3), (-1.6, 2.3)]
            ),
            "living_room": RoomZone(
                name="Living Room",
                id="living_room",
                x_min=1.4, x_max=4.5,
                z_min=1.0, z_max=4.0,
                color_hex=0x00ff88,
                bgr_color=(0, 240, 120),
                doors=[(1.4, 2.3)]
            ),
            "kitchen": RoomZone(
                name="Kitchen / Dining",
                id="kitchen",
                x_min=-4.5, x_max=-1.6,
                z_min=1.0, z_max=4.0,
                color_hex=0xff9e00,
                bgr_color=(0, 160, 255),
                doors=[(-1.6, 2.3)]
            )
        }

    def update_location(self, x: float, z: float) -> SpatialHouseMapResult:
        now = time.time()
        dt = max(0.001, now - self.last_update_time)
        self.last_update_time = now

        detected_room_id = list(self.rooms.keys())[0] if self.rooms else "office"
        matched = False

        for r_id, r in self.rooms.items():
            if r.x_min <= x <= r.x_max and r.z_min <= z <= r.z_max:
                detected_room_id = r_id
                matched = True
                break

        if not matched and self.rooms:
            min_dist = 999.0
            for r_id, r in self.rooms.items():
                cx = (r.x_min + r.x_max) / 2.0
                cz = (r.z_min + r.z_max) / 2.0
                d = math.hypot(x - cx, z - cz)
                if d < min_dist:
                    min_dist = d
                    detected_room_id = r_id

        self.current_room_id = detected_room_id
        if detected_room_id not in self.time_spent:
            self.time_spent[detected_room_id] = 0.0
        self.time_spent[detected_room_id] += dt

        cur_room = self.rooms[detected_room_id]
        dist_wall = min(
            abs(x - cur_room.x_min),
            abs(x - cur_room.x_max),
            abs(z - cur_room.z_min),
            abs(z - cur_room.z_max)
        )

        total_t = max(1.0, sum(self.time_spent.values()))
        heatmap = {r_id: round((t / total_t) * 100.0, 1) for r_id, t in self.time_spent.items()}

        return SpatialHouseMapResult(
            current_room=cur_room.name,
            room_id=cur_room.id,
            room_confidence=0.94 if self.is_auto_scanned else 0.85,
            pos_x=round(x, 2),
            pos_z=round(z, 2),
            distance_to_nearest_wall=round(dist_wall, 2),
            is_auto_scanned=self.is_auto_scanned,
            occupancy_heatmap=heatmap
        )

    def get_floorplan_walls(self) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        walls = []
        floor_y = -0.75
        wall_h = 0.85
        
        for r in self.rooms.values():
            corners = [
                (r.x_min, floor_y, r.z_min),
                (r.x_max, floor_y, r.z_min),
                (r.x_max, floor_y, r.z_max),
                (r.x_min, floor_y, r.z_max)
            ]
            for i in range(4):
                c1 = corners[i]
                c2 = corners[(i + 1) % 4]
                walls.append((c1, c2))
                t1 = (c1[0], floor_y + wall_h, c1[2])
                t2 = (c2[0], floor_y + wall_h, c2[2])
                walls.append((t1, t2))
                walls.append((c1, t1))
                
        return walls
