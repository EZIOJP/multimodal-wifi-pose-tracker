"""
3D Spatial Environment & Digital Twin Visualizer.
Renders real-time 3D skeletal avatars from BOTH Optical Camera AND WiFi CSI DensePose,
along with 3D room stage, virtual desk & chair, and volumetric RF reflection point clouds.
"""

import math
import time
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any

class Spatial3DVisualizer:
    def __init__(self, width=640, height=720):
        self.width = width
        self.height = height
        
        # 3D Camera viewpoint parameters
        self.cam_yaw = 25.0       # Rotation around Y axis in degrees
        self.cam_pitch = 18.0     # Pitch down looking at avatar
        self.cam_dist = 2.4       # Distance from origin in meters
        self.focal_scale = 480.0
        
        # Animation states
        self.rf_pulse_phase = 0.0
        self.resp_pulse_phase = 0.0
        
        # Bones connection map for MediaPipe / Kinematic 3D Pose
        self.bones = [
            (11, 12),           # Shoulder to shoulder
            (11, 13), (13, 15), # Left arm
            (12, 14), (14, 16), # Right arm
            (11, 23), (12, 24), # Torso sides
            (23, 24),           # Pelvis / Hips
            (23, 25), (25, 27), # Left leg
            (24, 26), (26, 28), # Right leg
            (0, 11), (0, 12)    # Neck / Head to shoulders
        ]

    def set_camera_rotation(self, delta_yaw=0.0, delta_pitch=0.0):
        self.cam_yaw = (self.cam_yaw + delta_yaw) % 360.0
        self.cam_pitch = np.clip(self.cam_pitch + delta_pitch, -20.0, 75.0)

    def project_3d(self, pt_3d: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Projects a 3D coordinate (x, y, z) into 2D viewport coordinates.
        Coordinates: X right (+), Y up (+), Z into depth (+)
        """
        rad_yaw = math.radians(self.cam_yaw)
        rad_pitch = math.radians(self.cam_pitch)
        
        x, y, z = pt_3d[0], pt_3d[1], pt_3d[2]
        
        # Yaw rotation (around Y)
        x1 = x * math.cos(rad_yaw) + z * math.sin(rad_yaw)
        y1 = y
        z1 = -x * math.sin(rad_yaw) + z * math.cos(rad_yaw)
        
        # Pitch rotation (around X)
        x2 = x1
        y2 = y1 * math.cos(rad_pitch) - z1 * math.sin(rad_pitch)
        z2 = y1 * math.sin(rad_pitch) + z1 * math.cos(rad_pitch)
        
        # Translate by camera distance
        z_cam = z2 + self.cam_dist
        if z_cam <= 0.2:
            return None, z_cam
            
        proj_x = int(self.width / 2.0 + (x2 / z_cam) * self.focal_scale)
        proj_y = int(self.height / 2.0 - (y2 / z_cam) * self.focal_scale)
        
        return (proj_x, proj_y), z_cam

    def render(self, optical_landmarks, wifi_dense_pose, posture: str, activity: str, 
               resp_rpm: float, attention_score: float, is_camera_active: bool) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Deep sci-fi studio background
        for row in range(self.height):
            ratio = row / self.height
            b = int(24 * (1.0 - ratio) + 10 * ratio)
            g = int(28 * (1.0 - ratio) + 12 * ratio)
            r = int(36 * (1.0 - ratio) + 16 * ratio)
            canvas[row, :] = (b, g, r)

        self.rf_pulse_phase += 0.08
        self.resp_pulse_phase += (resp_rpm / 60.0) * 0.2

        # 1. Ground Floor Grid
        self._draw_3d_grid(canvas)

        # 2. Virtual 3D Desk & Chair
        self._draw_3d_furniture(canvas, posture)

        # 3. Virtual WiFi CSI Antenna Node on Wall
        self._draw_3d_wifi_wavefield(canvas, resp_rpm)

        # 4. Avatar Rendering (Optical vs WiFi CSI)
        tracking_source = "NONE"
        if is_camera_active and optical_landmarks is not None and len(optical_landmarks) > 0:
            # High-precision Optical Skeleton
            tracking_source = "OPTICAL CAMERA (In FOV)"
            self._draw_optical_skeleton(canvas, optical_landmarks, posture, activity)
        elif wifi_dense_pose is not None and wifi_dense_pose.landmarks:
            # WiFi CSI Holographic Avatar
            tracking_source = "WiFi CSI DensePose (Camera: Out of FOV)"
            self._draw_wifi_holographic_skeleton(canvas, wifi_dense_pose, posture, activity)
        else:
            cv2.putText(canvas, "NO PERSON DETECTED IN ROOM", (self.width // 2 - 150, self.height // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (100, 110, 130), 1)

        # 5. 3D Viewport HUD
        self._draw_3d_hud(canvas, posture, activity, resp_rpm, attention_score, tracking_source)

        return canvas

    def _draw_3d_grid(self, canvas):
        grid_color = (48, 55, 70)
        axis_color = (0, 180, 255)
        floor_y = -0.75
        
        for x_val in np.linspace(-1.5, 1.5, 11):
            p1, _ = self.project_3d(np.array([x_val, floor_y, -1.5]))
            p2, _ = self.project_3d(np.array([x_val, floor_y, 1.5]))
            if p1 and p2:
                cv2.line(canvas, p1, p2, grid_color, 1)

        for z_val in np.linspace(-1.5, 1.5, 11):
            p1, _ = self.project_3d(np.array([-1.5, floor_y, z_val]))
            p2, _ = self.project_3d(np.array([1.5, floor_y, z_val]))
            if p1 and p2:
                cv2.line(canvas, p1, p2, grid_color, 1)

        c0, _ = self.project_3d(np.array([0.0, floor_y, 0.0]))
        if c0:
            cv2.circle(canvas, c0, 4, axis_color, -1)

    def _draw_3d_furniture(self, canvas, posture: str):
        floor_y = -0.75
        desk_color = (60, 75, 100)
        chair_color = (55, 65, 88)
        
        dx, dy, dz = 0.65, -0.15, 0.45
        desk_top = [
            np.array([-dx, dy, dz]), np.array([dx, dy, dz]),
            np.array([dx, dy, dz + 0.4]), np.array([-dx, dy, dz + 0.4])
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

        cx, cy, cz = 0.26, -0.42, -0.05
        chair_seat = [
            np.array([-cx, cy, cz - 0.22]), np.array([cx, cy, cz - 0.22]),
            np.array([cx, cy, cz + 0.22]), np.array([-cx, cy, cz + 0.22])
        ]
        proj_seat = [self.project_3d(pt)[0] for pt in chair_seat]
        if all(p is not None for p in proj_seat):
            pts_c = np.array(proj_seat, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_c], True, chair_color, 2)
            cb_top_l, _ = self.project_3d(np.array([-cx, cy + 0.35, cz - 0.22]))
            cb_top_r, _ = self.project_3d(np.array([cx, cy + 0.35, cz - 0.22]))
            if cb_top_l and cb_top_r and proj_seat[0] and proj_seat[1]:
                cv2.line(canvas, proj_seat[0], cb_top_l, chair_color, 2)
                cv2.line(canvas, proj_seat[1], cb_top_r, chair_color, 2)
                cv2.line(canvas, cb_top_l, cb_top_r, chair_color, 2)

    def _draw_3d_wifi_wavefield(self, canvas, resp_rpm: float):
        node_pos = np.array([1.2, 0.6, 0.8])
        p_node, _ = self.project_3d(node_pos)
        if p_node:
            cv2.circle(canvas, p_node, 6, (0, 220, 255), -1)
            cv2.putText(canvas, "WiFi CSI Node", (p_node[0] + 10, p_node[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1)
            
            num_rings = 4
            for i in range(num_rings):
                ring_phase = (self.rf_pulse_phase + i * (math.pi * 2 / num_rings)) % (math.pi * 2)
                radius = 0.3 + 0.8 * (ring_phase / (math.pi * 2))
                alpha = max(0.1, 1.0 - (radius / 1.1))
                
                circle_pts = []
                for a_deg in range(0, 360, 30):
                    rad = math.radians(a_deg)
                    pt = node_pos + np.array([-radius * math.cos(rad), radius * math.sin(rad), 0.0])
                    p_proj, _ = self.project_3d(pt)
                    if p_proj:
                        circle_pts.append(p_proj)
                if len(circle_pts) > 6:
                    pts_arr = np.array(circle_pts, np.int32).reshape((-1, 1, 2))
                    col = (int(0 * alpha), int(160 * alpha), int(220 * alpha))
                    cv2.polylines(canvas, [pts_arr], True, col, 1)

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
                cv2.line(canvas, pts_3d[i1][0], pts_3d[i2][0], bone_color, 3)

        if 0 in pts_3d:
            p_nose, _, pos_nose = pts_3d[0]
            cv2.circle(canvas, p_nose, 12, joint_color, -1)
            cv2.circle(canvas, p_nose, 14, bone_color, 2)
            gaze_target = pos_nose + np.array([0.0, -0.05, 0.35])
            p_gaze, _ = self.project_3d(gaze_target)
            if p_gaze:
                cv2.arrowedLine(canvas, p_nose, p_gaze, (0, 255, 255), 2, tipLength=0.3)

        for idx, (p_2d, _, _) in pts_3d.items():
            if idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                cv2.circle(canvas, p_2d, 4, joint_color, -1)

    def _draw_wifi_holographic_skeleton(self, canvas, wifi_pose, posture: str, activity: str):
        """
        Renders the WiFi CSI reconstructed avatar + volumetric RF Doppler point cloud.
        """
        # 1. Render Volumetric RF Reflection Cloud
        for pt in wifi_pose.rf_point_cloud:
            p_proj, _ = self.project_3d(np.array(pt))
            if p_proj:
                # Scintillating cyan RF particle
                cv2.circle(canvas, p_proj, 2, (255, 220, 0), -1)

        # 2. Render Holographic Skeleton
        pts_3d = {}
        rf_bone_color = (255, 220, 0)      # Cyan / Neon Blue RF signature
        rf_joint_color = (0, 255, 255)     # Glowing Yellow Joint
        
        for idx, lm in enumerate(wifi_pose.landmarks):
            pos = np.array([lm.x, lm.y, lm.z])
            p_proj, z_c = self.project_3d(pos)
            if p_proj is not None:
                pts_3d[idx] = (p_proj, z_c, pos)

        for i1, i2 in self.bones:
            if i1 in pts_3d and i2 in pts_3d:
                # Pulsing RF connection line
                cv2.line(canvas, pts_3d[i1][0], pts_3d[i2][0], rf_bone_color, 2)

        if 0 in pts_3d:
            p_nose, _, _ = pts_3d[0]
            cv2.circle(canvas, p_nose, 11, rf_joint_color, -1)
            cv2.circle(canvas, p_nose, 15, rf_bone_color, 1)

        for idx, (p_2d, _, _) in pts_3d.items():
            if idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                cv2.circle(canvas, p_2d, 4, rf_joint_color, -1)

        # Draw Pulsing WiFi Chest Resonance Sphere
        if 11 in pts_3d and 12 in pts_3d:
            sh_l = pts_3d[11][0]
            sh_r = pts_3d[12][0]
            chest_center = ((sh_l[0] + sh_r[0]) // 2, (sh_l[1] + sh_r[1]) // 2)
            pulse_rad = int(9 + 5 * math.sin(self.resp_pulse_phase))
            cv2.circle(canvas, chest_center, pulse_rad, (0, 240, 255), 1)

    def _draw_3d_hud(self, canvas, posture: str, activity: str, resp_rpm: float, attention: float, source: str):
        cv2.putText(canvas, "3D SPATIAL DIGITAL TWIN", (20, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 230, 220), 1)
        
        info_str = f"Orbit Yaw: {self.cam_yaw:.0f}° | Pitch: {self.cam_pitch:.0f}° | [A/D/W/S] Rotate"
        cv2.putText(canvas, info_str, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 175, 190), 1)

        # Active sensing source pill
        src_color = (0, 240, 120) if "OPTICAL" in source else (255, 215, 0)
        cv2.putText(canvas, f"SENSING: {source}", (20, self.height - 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, src_color, 1)
