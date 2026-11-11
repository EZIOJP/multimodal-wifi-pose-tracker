"""
Desktop Graphical Launcher & Control Center.
Provides 1-click 'RUN EVERYTHING', Auto-Scan Real House Rooms,
Camera selection, and Calibration.
"""

import os
import sys
import subprocess
import threading
import webbrowser
import time
import tkinter as tk
from tkinter import ttk, messagebox

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable

class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multimodal Sensing & Multi-Room House SLAM Control Center")
        self.root.geometry("640x590")
        self.root.resizable(False, False)
        self.root.configure(bg="#0f141c")

        self.processes = []
        self._build_ui()
        self._check_status()

    def _build_ui(self):
        # 1. Title Banner
        title_frame = tk.Frame(self.root, bg="#161f2e", pady=15, padx=20)
        title_frame.pack(fill="x")

        lbl_title = tk.Label(
            title_frame,
            text="MULTIMODAL SENSING & HOUSE SLAM",
            font=("Segoe UI", 16, "bold"),
            fg="#00e5ff",
            bg="#161f2e"
        )
        lbl_title.pack(anchor="w")

        lbl_sub = tk.Label(
            title_frame,
            text="Webcam Vision + WiFi CSI Multi-Room DensePose & House Auto-Mapper",
            font=("Segoe UI", 10),
            fg="#8fa0b5",
            bg="#161f2e"
        )
        lbl_sub.pack(anchor="w", pady=(3, 0))

        # 2. Main Content Frame
        main_frame = tk.Frame(self.root, bg="#0f141c", padx=25, pady=18)
        main_frame.pack(fill="both", expand=True)

        # Status Box
        status_box = tk.LabelFrame(
            main_frame,
            text=" System Readiness Diagnostics ",
            font=("Segoe UI", 10, "bold"),
            fg="#00e5ff",
            bg="#141c28",
            padx=15,
            pady=8
        )
        status_box.pack(fill="x", pady=(0, 12))

        self.lbl_models = tk.Label(status_box, text="• AI Models: Checking...", font=("Segoe UI", 10), fg="#e0e6ed", bg="#141c28")
        self.lbl_models.pack(anchor="w", pady=2)

        self.lbl_rooms = tk.Label(status_box, text="• House Floorplan: Checking...", font=("Segoe UI", 10), fg="#e0e6ed", bg="#141c28")
        self.lbl_rooms.pack(anchor="w", pady=2)

        self.lbl_cal = tk.Label(status_box, text="• Calibration Profile: Checking...", font=("Segoe UI", 10), fg="#e0e6ed", bg="#141c28")
        self.lbl_cal.pack(anchor="w", pady=2)

        # Camera selection row
        cam_frame = tk.Frame(main_frame, bg="#0f141c")
        cam_frame.pack(fill="x", pady=(0, 12))
        
        lbl_cam = tk.Label(cam_frame, text="Select Webcam Device:", font=("Segoe UI", 10, "bold"), fg="#e0e6ed", bg="#0f141c")
        lbl_cam.pack(side="left")

        self.cam_var = tk.StringVar(value="0")
        cam_combo = ttk.Combobox(cam_frame, textvariable=self.cam_var, values=["0", "1", "2"], width=8, state="readonly")
        cam_combo.pack(side="left", padx=10)

        # --- ACTION BUTTONS ---
        
        # 1. Big Glowing 'RUN EVERYTHING' Button
        btn_run_all = tk.Button(
            main_frame,
            text="🚀  RUN EVERYTHING TOGETHER  (Desktop 3D + WebGL)",
            font=("Segoe UI", 12, "bold"),
            bg="#00e5ff",
            fg="#000000",
            activebackground="#33ecff",
            activeforeground="#000000",
            relief="flat",
            pady=11,
            cursor="hand2",
            command=self.run_everything
        )
        btn_run_all.pack(fill="x", pady=4)

        # 2. Auto-Scan Real House (Walk-to-Map)
        btn_scan = tk.Button(
            main_frame,
            text="🗺️  Auto-Scan Real House Rooms (Walk-to-Map WiFi SLAM)",
            font=("Segoe UI", 10, "bold"),
            bg="#1c2a3d",
            fg="#00e5ff",
            activebackground="#293d56",
            activeforeground="#00e5ff",
            relief="flat",
            pady=7,
            cursor="hand2",
            command=self.run_room_scanner
        )
        btn_scan.pack(fill="x", pady=3)

        # 3. Desktop 3D Split-Screen App Only
        btn_run_desktop = tk.Button(
            main_frame,
            text="🖥️  Launch Desktop 3D Tracker (Camera + Multi-Room Stage)",
            font=("Segoe UI", 10, "bold"),
            bg="#1c2a3d",
            fg="#00ff88",
            activebackground="#293d56",
            activeforeground="#00ff88",
            relief="flat",
            pady=7,
            cursor="hand2",
            command=self.run_desktop_app
        )
        btn_run_desktop.pack(fill="x", pady=3)

        # 4. 3-Step Calibration Wizard
        btn_cal = tk.Button(
            main_frame,
            text="🎯  Launch 3-Step Calibration Wizard (Sit / Stand / Gaze)",
            font=("Segoe UI", 10),
            bg="#1c2a3d",
            fg="#ffd000",
            activebackground="#293d56",
            activeforeground="#ffd000",
            relief="flat",
            pady=7,
            cursor="hand2",
            command=self.run_calibration
        )
        btn_cal.pack(fill="x", pady=3)

        # 5. WebGL Browser 3D Twin Only
        btn_web = tk.Button(
            main_frame,
            text="🌐  Launch Web 3D Digital Twin Server (Browser at localhost:8000)",
            font=("Segoe UI", 10),
            bg="#1c2a3d",
            fg="#e0e6ed",
            activebackground="#293d56",
            activeforeground="#ffffff",
            relief="flat",
            pady=7,
            cursor="hand2",
            command=self.run_web_server
        )
        btn_web.pack(fill="x", pady=3)

        # Status text footer
        self.lbl_footer = tk.Label(
            self.root,
            text="Status: Ready to launch. Press 'Q' anytime in tracker to quit.",
            font=("Segoe UI", 9),
            fg="#708298",
            bg="#0f141c",
            pady=6
        )
        self.lbl_footer.pack(side="bottom")

    def _check_status(self):
        models_dir = os.path.join(CURRENT_DIR, "models")
        pose_m = os.path.join(models_dir, "pose_landmarker.task")
        face_m = os.path.join(models_dir, "face_landmarker.task")
        gest_m = os.path.join(models_dir, "gesture_recognizer.task")

        if os.path.exists(pose_m) and os.path.exists(face_m) and os.path.exists(gest_m):
            self.lbl_models.config(text="• AI Models: [OK] Google MediaPipe Models Installed", fg="#00ff88")
        else:
            self.lbl_models.config(text="• AI Models: [!] Missing models (Auto-downloads on run)", fg="#ffd000")

        rooms_json = os.path.join(CURRENT_DIR, "config", "house_rooms.json")
        if os.path.exists(rooms_json):
            self.lbl_rooms.config(text="• House Floorplan: [OK] Auto-Scanned Real Rooms Active", fg="#00ff88")
        else:
            self.lbl_rooms.config(text="• House Floorplan: Default [Click 'Auto-Scan' to map real home]", fg="#ffd000")

        cal_json = os.path.join(CURRENT_DIR, "config", "calibration.json")
        if os.path.exists(cal_json):
            self.lbl_cal.config(text="• Calibration Profile: [OK] Custom Profile Found", fg="#00ff88")
        else:
            self.lbl_cal.config(text="• Calibration Profile: Default [Click Calibration Wizard below]", fg="#ffd000")

    def run_room_scanner(self):
        scan_script = os.path.join(CURRENT_DIR, "room_scanner.py")
        self.lbl_footer.config(text="Status: Running Auto-Scan Walkthrough...")
        def launch():
            subprocess.run([PYTHON_EXE, scan_script])
            self._check_status()
            self.lbl_footer.config(text="Status: House Scan finished & saved to house_rooms.json.")
        threading.Thread(target=launch, daemon=True).start()

    def run_everything(self):
        self.lbl_footer.config(text="Status: Starting 3D Web Server & Desktop Tracker...")
        server_script = os.path.join(CURRENT_DIR, "server_3d.py")
        p_server = subprocess.Popen([PYTHON_EXE, server_script])
        self.processes.append(p_server)
        
        def open_browser():
            time.sleep(2.0)
            webbrowser.open("http://localhost:8000")
        threading.Thread(target=open_browser, daemon=True).start()

        self.run_desktop_app()

    def run_desktop_app(self):
        cam = self.cam_var.get()
        app_script = os.path.join(CURRENT_DIR, "app.py")
        self.lbl_footer.config(text=f"Status: Running Desktop 3D Tracker (Camera {cam})...")
        
        def launch():
            subprocess.run([PYTHON_EXE, app_script, cam])
            self.lbl_footer.config(text="Status: Desktop Tracker closed.")
        threading.Thread(target=launch, daemon=True).start()

    def run_calibration(self):
        cam = self.cam_var.get()
        cal_script = os.path.join(CURRENT_DIR, "calibrator.py")
        self.lbl_footer.config(text="Status: Running 3-Step Calibration Wizard...")
        
        def launch():
            subprocess.run([PYTHON_EXE, cal_script, cam])
            self._check_status()
            self.lbl_footer.config(text="Status: Calibration Wizard finished.")
        threading.Thread(target=launch, daemon=True).start()

    def run_web_server(self):
        server_script = os.path.join(CURRENT_DIR, "server_3d.py")
        self.lbl_footer.config(text="Status: Launching Web 3D Server at http://localhost:8000...")
        p_server = subprocess.Popen([PYTHON_EXE, server_script])
        self.processes.append(p_server)
        
        def open_browser():
            time.sleep(1.8)
            webbrowser.open("http://localhost:8000")
        threading.Thread(target=open_browser, daemon=True).start()

def main():
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
