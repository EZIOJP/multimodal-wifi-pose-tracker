# 🌐 Multimodal Spatial Human Activity & Multi-Room WiFi SLAM Tracker

A multimodal sensing pipeline that fuses **Laptop Webcam Vision (MediaPipe Tasks)** with **Real WiFi Hardware Signal Sensing (RuView / WiFi CSI & RSSI Fingerprinting)** to recognize human activities, solve sit/stand desk tracking, and map indoor multi-room house positions — complete with an interactive **3D Spatial Digital Twin Room Environment**.

---

## 🌟 Key Innovations & Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   MULTIMODAL SENSING & 3D DIGITAL TWIN                 │
├───────────────────────────────────┬────────────────────────────────────┤
│         1. VISION TIER            │          2. WiFi SENSING           │
│  • Multi-Factor Sit vs Stand      │  • Real MediaTek Wi-Fi 6 RSSI      │
│  • Hand-to-Mouth Eating/Drinking  │  • Physical Multi-Room SLAM        │
│  • Eye Closure & Gaze Attention   │  • Through-Wall 3D DensePose       │
├───────────────────────────────────┴────────────────────────────────────┤
│                       3. FUSION & ANALYTICS                            │
│  • 45-Frame Temporal Trajectory Buffer (PoseSense / Aditi1607)         │
│  • Cognitive Burnout & Sedentary Fatigue Gauge                         │
├────────────────────────────────────────────────────────────────────────┤
│                     4. 3D SPATIAL DIGITAL TWIN                         │
│  • Desktop Dual Viewport & WebGL (Three.js at localhost:8000)          │
│  • Multi-Room House Floor Plan (Office, Hallway, Living Room, Kitchen) │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Multi-Factor Sit vs. Stand Classifier
Solves the fundamental limitation of laptop webcams where legs and knees are hidden under the desk:
* **Auto-Rolling Baseline**: Learns your natural sitting height within 15 frames.
* **Perspective Look-Up Angle**: Measures camera upward pitch tilt ($> +14^\circ$).
* **Top-Boundary Exit Tracking**: Maintains `STANDING` state even if your head moves above the camera frame.
* **Scale Ratio**: Detects when you stand up and step back.

### 2. Real Laptop WiFi Hardware RSSI & Multi-Room SLAM
* Directly queries your laptop's **MediaTek Wi-Fi 6 card** via Windows `netsh` to read physical signal attenuation (dBm).
* Automatically estimates your location across house zones:
  * **Office / Desk:** $\approx -50\text{ dBm}$ (Signal 85–95%)
  * **Central Hallway:** $\approx -62\text{ dBm}$ (Signal 60–75%)
  * **Living Room:** $\approx -72\text{ dBm}$ (Signal 45–60%)
  * **Kitchen / Balcony:** $\approx -82\text{ dBm}$ (Signal 20–40%)
* Press **`R`** to instantly calibrate any room's signal fingerprint.

### 3. Automatic WiFi Room Scanner ("Walk-to-Map")
* Run `room_scanner.py` to auto-map your real house layout.
* Uses **DBSCAN machine learning spatial clustering** on your walking trajectory to discover distinct rooms, doorways, and wall boundaries, saving directly to `config/house_rooms.json`.

### 4. Interactive 3D Spatial Digital Twin
* **Desktop App**: Side-by-side split screen showing real camera feed on the left and the 3D room stage on the right.
* **WebGL Browser (Three.js)**: Full 3D orbit controls (mouse drag to rotate, scroll to zoom) running on `http://localhost:8000`.

---

## 📂 Project Structure

```
├── run_all.bat               # 1-Click launcher (starts web server + desktop HUD)
├── run_tracker.bat           # Interactive menu launcher
├── activity_tracker/
│   ├── app.py                # Main desktop app with live 3D Split-Screen HUD
│   ├── launcher_gui.py       # Graphical Control Center with clickable buttons
│   ├── visualizer_3d.py      # 3D Perspective Projection Engine (Rooms, Skeleton)
│   ├── server_3d.py          # FastAPI + WebSocket server for WebGL 3D browser viewing
│   ├── room_scanner.py       # "Walk-to-Map" RF SLAM automatic room discovery
│   ├── room_mapper.py        # Multi-room architectural house floor plan manager
│   ├── real_wifi_scanner.py  # Live Windows Wi-Fi 6 RSSI hardware poller
│   ├── calibrator.py         # 3-step interactive calibration wizard & profile manager
│   ├── vision_engine.py      # MediaPipe pose, face blendshapes, gaze, and hand tracker
│   ├── ruview_bridge.py      # RuView UDP listener (:8888) & DensePose generator
│   ├── temporal_buffer.py    # 45-frame sliding window trajectory engine
│   ├── fatigue_analyzer.py   # Sedentary timer & student cognitive burnout engine
│   ├── fusion_engine.py      # Multimodal Bayesian/temporal fusion state machine
│   ├── download_models.py    # Google MediaPipe task model downloader
│   ├── web/
│   │   └── index.html        # Three.js 3D WebGL Digital Twin with OrbitControls
│   └── config/               # Calibration profiles and auto-scanned room layouts
```

---

## 🚀 Quick Start

### 1. Launch Everything (1-Click)
Double-click `run_all.bat` or run:
```cmd
run_all.bat
```

### 2. Launch Graphical Control Center
```powershell
python activity_tracker/launcher_gui.py
```

### 3. Hotkeys Inside the Desktop App
* **`V`**: Cycle view modes (Split-Screen Dual View $\leftrightarrow$ Fullscreen 3D $\leftrightarrow$ Camera Only).
* **`A` / `D`**: Orbit the 3D camera left / right around yourself (360°).
* **`W` / `S`**: Orbit the 3D camera up / down (tilt angle).
* **`R`**: Calibrate current room's live WiFi RSSI signal.
* **`C`**: Quick-calibrate sitting desk height.
* **`K`**: Launch 3-Step Calibration Wizard.
* **`Q`**: Quit application.

---

## 📚 Open-Source Foundations & Acknowledgments
* [RuView](https://github.com/ruvnet/ruview) — Commodity WiFi spatial intelligence & DensePose.
* [SenseFi](https://github.com/YangZhao123/SenseFi) — Deep learning library for WiFi sensing.
* [PoseSense](https://github.com/Prith-ML/PoseSense) — 3D joint keypoint extraction and temporal sequence classification.
* [Google MediaPipe](https://github.com/google/mediapipe) — On-device vision and face blendshape models.
