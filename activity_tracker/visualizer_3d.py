"""
3D Spatial Environment & Multi-Room House Floor Plan Visualizer.
Renders 3D architectural floor plans (Office, Hallway, Living Room, Kitchen)
with real-time human avatar localization and wall geometry using 3D perspective projection.
"""

import math
import time
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any

from activity_tracker.room_mapper import HouseRoomMapper

class Spatial3DVisualizer:
    def __init__(self, width=640, height=720):
        self.width = width
        self.height = height
        
        # 3D Camera viewpoint parameters
        self.cam_yaw = 28.0       # Rotation around Y axis in degrees
        self.cam_pitch = 24.0     # Pitch down looking at multi-room house
        self.cam_dist = 4.2       # Zoomed out to view the full multi-room house floorplan
        self.focal_scale = 380.0
        
        self.rf_pulse_phase = 0.0
        self.room_mapper = HouseRoomMapper()
        self.cached_walls = self.room_mapper.get_floorplan_walls()
        
        # Bones connection map
        self.bones = [
            (11, 12),           # Shoulder to shoulder
            (11, 13), (13, 15), # Left arm
            (12, 14), (14, 16), # Right arm
            (11, 23), (12, 24), # Torso sides
            (23, 24),           # Pelvis / Hips
            (23, 25), (25, 27), # Left leg
            (24, 26), (26, 28), # Right leg
            (0, 11), (0, 12)    # Head to shoulders
        ]

    def set_camera_rotation(self, delta_yaw=0.0, delta_pitch=0.0):
        self.cam_yaw = (self.cam_yaw + delta_yaw) % 360.0
        self.cam_pitch = np.clip(self.cam_pitch + delta_pitch, -10.0, 80.0)

    def project_3d(self, pt_3d: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        rad_yaw = math.radians(self.cam_yaw)
        rad_pitch = math.radians(self.cam_pitch)
        
        x, y, z = pt_3d[0], pt_3d[1], pt_3d[2]
        
        # Camera orbit center shifted slightly forward toward center of house (0, 1.5)
        z_offset = 1.2
        z_adj = z - z_offset
        
        x1 = x * math.cos(rad_yaw) + z_adj * math.sin(rad_yaw)
        y1 = y
        z1 = -x * math.sin(rad_yaw) + z_adj * math.cos(rad_yaw)
        
        x2 = x1
        y2 = y1 * math.cos(rad_pitch) - z1 * math.sin(rad_pitch)
        z2 = y1 * math.sin(rad_pitch) + z1 * math.cos(rad_pitch)
        
        z_cam = z2 + self.cam_dist
        if z_cam <= 0.2:
            return None, z_cam
            
        proj_x = int(self.width / 2.0 + (x2 / z_cam) * self.focal_scale)
        proj_y = int(self.height / 2.0 - (y2 / z_cam) * self.focal_scale)
        
        return (proj_x, proj_y), z_cam

    def render(self, optical_landmarks, wifi_dense_pose, posture: str, activity: str, 
               attention_score: float, is_camera_active: bool) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Sci-Fi architectural blueprint background
        for row in range(self.height):
            ratio = row / self.height
            b = int(22 * (1.0 - ratio) + 8 * ratio)
            g = int(26 * (1.0 - ratio) + 10 * ratio)
            r = int(32 * (1.0 - ratio) + 14 * ratio)
            canvas[row, :] = (b, g, r)

        self.rf_pulse_phase += 0.08

        # 1. Render Multi-Room Floor Grid & Architectural Walls
        self._draw_house_floorplan(canvas)

        # 2. Render Virtual Desk & Chair in Office
        self._draw_office_furniture(canvas)

        # 3. Render WiFi CSI Wavefield
        self._draw_wifi_nodes(canvas)

        # 4. Avatar Rendering (Optical vs WiFi CSI DensePose)
        tracking_source = "NONE"
        cur_room_label = "Home Office / Study"
        coords_str = "(0.0m, 0.0m)"

        if is_camera_active and optical_landmarks is not None and len(optical_landmarks) > 0:
            tracking_source = "OPTICAL CAMERA (In FOV)"
            self._draw_optical_skeleton(canvas, optical_landmarks, posture, activity)
        elif wifi_dense_pose is not None and wifi_dense_pose.landmarks:
            tracking_source = f"WiFi CSI House Map: {wifi_dense_pose.current_room}"
            cur_room_label = wifi_dense_pose.current_room
            coords_str = f"({wifi_dense_pose.room_x:+.1f}m, {wifi_dense_pose.room_z:+.1f}m)"
            self._draw_wifi_holographic_skeleton(canvas, wifi_dense_pose, posture, activity)
        else:
            cv2.putText(canvas, "NO PERSON DETECTED IN HOUSE", (self.width // 2 - 150, self.height // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.60, (100, 110, 130), 1)

        # 5. 3D Viewport HUD with House Room Location
        self._draw_3d_hud(canvas, posture, activity, cur_room_label, coords_str, tracking_source)

        return canvas

    def _draw_house_floorplan(self, canvas):
        floor_y = -0.75
        
        # Ground grid
        for x_val in np.linspace(-4.5, 4.5, 19):
            p1, _ = self.project_3d(np.array([x_val, floor_y, -1.2]))
            p2, _ = self.project_3d(np.array([x_val, floor_y, 4.0]))
            if p1 and p2:
                cv2.line(canvas, p1, p2, (38, 44, 55), 1)

        for z_val in np.linspace(-1.2, 4.0, 14):
            p1, _ = self.project_3d(np.array([-4.5, floor_y, z_val]))
            p2, _ = self.project_3d(np.array([4.5, floor_y, z_val]))
            if p1 and p2:
                cv2.line(canvas, p1, p2, (38, 44, 55), 1)

        # Architectural walls
        for c1, c2 in self.cached_walls:
            p1, _ = self.project_3d(np.array(c1))
            p2, _ = self.project_3d(np.array(c2))
            if p1 and p2:
                cv2.line(canvas, p1, p2, (75, 90, 115), 1)

        # Room floating labels in 3D
        for r in self.room_mapper.rooms.values():
            cx = (r.x_min + r.x_max) / 2.0
            cz = (r.z_min + r.z_max) / 2.0
            p_center, _ = self.project_3d(np.array([cx, floor_y + 0.05, cz]))
            if p_center:
                cv2.putText(canvas, r.name.upper(), (p_center[0] - 50, p_center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, r.bgr_color, 1)

    def _draw_office_furniture(self, canvas):
        floor_y = -0.75
        desk_color = (60, 75, 100)
        chair_color = (55, 65, 88)
        
        dx, dy, dz = 0.60, -0.15, 0.35
        desk_top = [
            np.array([-dx, dy, dz]), np.array([dx, dy, dz]),
            np.array([dx, dy, dz + 0.35]), np.array([-dx, dy, dz + 0.35])
        ]
        proj_top = [self.project_3d(pt)[0] for pt in desk_top]
        if all(p is not None for p in proj_top):
            pts = np.array(proj_top, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], True, desk_color, 2)
            for pt in desk_top:
                leg_bot = np.array([pt[0], floor_y, pt[2]])
                p_top, _ = self.project_3d(pt)
                p_bot, _ = self.project_3d(leg_bot)
                if p_top and p_bot:
                    cv2.line(canvas, p_top, p_bot, desk_color, 1)

        cx, cy, cz = 0.22, -0.42, -0.05
        chair_seat = [
            np.array([-cx, cy, cz - 0.20]), np.array([cx, cy, cz - 0.20]),
            np.array([cx, cy, cz + 0.20]), np.array([-cx, cy, cz + 0.20])
        ]
        proj_seat = [self.project_3d(pt)[0] for pt in chair_seat]
        if all(p is not None for p in proj_seat):
            pts_c = np.array(proj_seat, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_c], True, chair_color, 2)

    def _draw_wifi_nodes(self, canvas):
        nodes = [
            (np.array([1.3, 0.5, 0.5]), "WiFi Router (Office)"),
            (np.array([4.2, 0.5, 2.5]), "Mesh Node (Living Room)"),
            (np.array([-4.2, 0.5, 2.5]), "Mesh Node (Kitchen)")
        ]
        for pos, name in nodes:
            p_node, _ = self.project_3d(pos)
            if p_node:
                cv2.circle(canvas, p_node, 5, (0, 220, 255), -1)

    def _draw_optical_skeleton(self, canvas, landmarks, posture: str, activity: str):
        pts_3d = {}
        bone_color = (0, 240, 120) if "WORKING" in activity else ((0, 215, 255) if "EATING" in activity else (255, 190, 0))
        joint_color = (255, 255, 255)
        y_offset = 0.25 if "STAND" in posture else -0.10

        for idx, lm in enumerate(landmarks):
            pos = np.array([lm.x, -lm.y + y_offset, -lm.z])
            p_proj, z_c = self.project_3d(pos)
            if p_proj is not None:
                pts_3d[idx] = (p_proj, z_c, pos)

        for i1, i2 in self.bones:
            if i1 in pts_3d and i2 in pts_3d:
                cv2.line(canvas, pts_3d[i1][0], pts_3d[i2][0], bone_color, 2)

        if 0 in pts_3d:
            p_nose, _, _ = pts_3d[0]
            cv2.circle(canvas, p_nose, 9, joint_color, -1)

        for idx, (p_2d, _, _) in pts_3d.items():
            if idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                cv2.circle(canvas, p_2d, 3, joint_color, -1)

    def _draw_wifi_holographic_skeleton(self, canvas, wifi_pose, posture: str, activity: str):
        # 1. Draw volumetric RF points
        for pt in wifi_pose.rf_point_cloud:
            p_proj, _ = self.project_3d(np.array(pt))
            if p_proj:
                cv2.circle(canvas, p_proj, 2, (255, 220, 0), -1)

        # 2. Draw skeleton
        pts_3d = {}
        rf_bone_color = (255, 220, 0)      # Cyan / Neon Blue
        rf_joint_color = (0, 255, 255)     # Yellow glowing joint
        
        for idx, lm in enumerate(wifi_pose.landmarks):
            pos = np.array([lm.x, lm.y, lm.z])
            p_proj, z_c = self.project_3d(pos)
            if p_proj is not None:
                pts_3d[idx] = (p_proj, z_c, pos)

        for i1, i2 in self.bones:
            if i1 in pts_3d and i2 in pts_3d:
                cv2.line(canvas, pts_3d[i1][0], pts_3d[i2][0], rf_bone_color, 2)

        if 0 in pts_3d:
            p_nose, _, _ = pts_3d[0]
            cv2.circle(canvas, p_nose, 9, rf_joint_color, -1)

        for idx, (p_2d, _, _) in pts_3d.items():
            if idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                cv2.circle(canvas, p_2d, 3, rf_joint_color, -1)

    def _draw_3d_hud(self, canvas, posture: str, activity: str, room_label: str, coords_str: str, source: str):
        cv2.putText(canvas, "3D MULTI-ROOM HOUSE MAP", (20, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 230, 220), 1)
        
        loc_str = f"ROOM: {room_label} | POS: {coords_str}"
        cv2.putText(canvas, loc_str, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180), 1)

        src_color = (0, 240, 120) if "OPTICAL" in source else (255, 215, 0)
        cv2.putText(canvas, f"SENSING: {source}", (20, self.height - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, src_color, 1)
