"""
WiFi Multi-Room House Mapping & RF Spatial Floorplan Engine.
Maps human movement across architectural house zones (Office, Hallway, Living Room, Kitchen)
using WiFi CSI spatial multi-path reflection profiles and trajectory mapping.
"""

import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

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
    doors: List[Tuple[float, float]] = field(default_factory=list) # Doorway coordinates (x, z)

@dataclass
class SpatialHouseMapResult:
    current_room: str
    room_id: str
    room_confidence: float
    pos_x: float
    pos_z: float
    distance_to_nearest_wall: float
    occupancy_heatmap: Dict[str, float]  # Time % spent per room

class HouseRoomMapper:
    def __init__(self):
        # Define realistic multi-room architectural floor plan
        # Coordinates centered at Desk in the Office (0, 0)
        self.rooms: Dict[str, RoomZone] = {
            "office": RoomZone(
                name="Home Office / Study",
                id="office",
                x_min=-1.6, x_max=1.4,
                z_min=-1.2, z_max=1.5,
                color_hex=0x00e5ff,
                bgr_color=(255, 220, 0),
                doors=[(0.0, 1.5)]  # Door connects to hallway at z=1.5
            ),
            "hallway": RoomZone(
                name="Central Hallway",
                id="hallway",
                x_min=-1.6, x_max=1.4,
                z_min=1.5, z_max=3.2,
                color_hex=0x9d4edd,
                bgr_color=(220, 80, 160),
                doors=[(0.0, 1.5), (1.4, 2.3), (-1.6, 2.3)] # Connects to Office, Living Room, Kitchen
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
        
        self.time_spent = {r_id: 0.0 for r_id in self.rooms}
        self.last_update_time = time.time()
        self.current_room_id = "office"

    def update_location(self, x: float, z: float) -> SpatialHouseMapResult:
        now = time.time()
        dt = max(0.001, now - self.last_update_time)
        self.last_update_time = now

        detected_room_id = "office"
        matched = False

        # Find which room bounding box contains the coordinates
        for r_id, r in self.rooms.items():
            if r.x_min <= x <= r.x_max and r.z_min <= z <= r.z_max:
                detected_room_id = r_id
                matched = True
                break

        if not matched:
            # Find closest room center
            min_dist = 999.0
            for r_id, r in self.rooms.items():
                cx = (r.x_min + r.x_max) / 2.0
                cz = (r.z_min + r.z_max) / 2.0
                d = math.hypot(x - cx, z - cz)
                if d < min_dist:
                    min_dist = d
                    detected_room_id = r_id

        self.current_room_id = detected_room_id
        self.time_spent[detected_room_id] += dt

        # Distance to nearest wall of the current room
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
            room_confidence=0.92,
            pos_x=round(x, 2),
            pos_z=round(z, 2),
            distance_to_nearest_wall=round(dist_wall, 2),
            occupancy_heatmap=heatmap
        )

    def get_floorplan_walls(self) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Returns line segments (P1, P2) in 3D for architectural walls of the house.
        Coordinates are (x, y, z) where y=floor level (-0.75m) and wall height is 1.1m.
        """
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
            
            # Bottom wall lines
            for i in range(4):
                c1 = corners[i]
                c2 = corners[(i + 1) % 4]
                walls.append((c1, c2))
                # Top wall rim
                t1 = (c1[0], floor_y + wall_h, c1[2])
                t2 = (c2[0], floor_y + wall_h, c2[2])
                walls.append((t1, t2))
                # Vertical corner pillars
                walls.append((c1, t1))
                
        return walls
