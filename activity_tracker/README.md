# Multimodal Human Activity, Posture (Sit/Stand) & Attention Tracker

A multimodal sensing pipeline that fuses **Laptop Webcam Vision (MediaPipe Tasks)** with **RuView WiFi CSI RF Intelligence** to recognize user activities, improve sit/stand accuracy, and track attentiveness and vitals — complete with an interactive **3D Spatial Digital Twin Room Environment**.

---

## 🌟 Key Capabilities & Detection Mechanics

### 1. 3D Spatial Digital Twin Viewer (`visualizer_3d.py` & `web/index.html`)
Watch yourself in a real-time virtual 3D room to instantly verify whether tracking is working:
* **3D Skeletal Avatar**: Reconstructs your full 3D body pose using metric world coordinates ($X, Y, Z$) in meters.
* **3D Virtual Desk & Chair**: The 3D avatar physically sits down into the virtual chair or stands up when you rise.
* **3D Gaze Ray**: A 3D pointer shows the exact direction you are looking in 3D space.
* **3D WiFi CSI Wavefield**: Animated RF antenna ripples on the wall that react to body movement, with an expanding chest pulse illustrating real-time breathing Doppler.
* **Dual Viewport**: Split-screen display shows your real camera feed on the left and your 3D digital twin on the right.

### 2. Dynamic Calibration Engine (`calibrator.py`)
Customizes thresholds to your exact desk and torso proportions:
* **Sitting Baseline Height**: Captures your natural working shoulder Y-coordinate as $0\%$ elevation.
* **Standing Baseline Height**: Measures your standing shoulder Y-coordinate as $100\%$ elevation and computes your exact elevation span $\Delta Y$.
* **Neutral Gaze & Face Orientation**: Centers head yaw/pitch when looking directly at the screen.
* **Profile Persistence**: Saves to `activity_tracker/config/calibration.json`.

### 3. Activity & Attention Recognition
* **Eating & Drinking**: Verified by a 3-phase temporal trajectory (hand approaches mouth $\rightarrow$ dwells for $0.2\text{s} - 2\text{s}$ with jaw mastication $\rightarrow$ departs).
* **Sleeping & Drowsiness**: Continuous eye closure ($> 2.5\text{s}$), drooping head pitch, and low-frequency resting respiration ($12 - 16\text{ RPM}$).
* **Working / Typing**: Screen attention index ($> 60\%$), erect posture, hands at keyboard zone.
* **Out-of-Camera Room Presence (RuView Continuity)**: Distinguishes whether you are pacing in the room, resting nearby, or have left the room.

---

## 📂 Project Structure

```
activity_tracker/
├── app.py               # Main desktop app with live 3D Split-Screen HUD
├── visualizer_3d.py     # 3D Perspective Projection Engine (Room, Desk, Skeleton)
├── server_3d.py         # FastAPI + WebSocket server for WebGL 3D browser viewing
├── web/
│   └── index.html       # Three.js 3D WebGL Digital Twin with OrbitControls
├── calibrator.py        # 3-step interactive calibration wizard & profile manager
├── vision_engine.py     # MediaPipe pose, face blendshapes, gaze, and hand tracker
├── ruview_bridge.py     # RuView UDP listener (:8888) & adaptive RF synchronizer
├── csi_sanitizer.py     # SenseFi Hampel filter & respiration cleaner
├── temporal_buffer.py   # 45-frame sliding window trajectory engine (PoseSense style)
├── fatigue_analyzer.py  # Sedentary timer & student cognitive burnout engine
├── fusion_engine.py     # Multimodal Bayesian/temporal fusion state machine
└── config/              # Persistent calibration profile (calibration.json)
```

---

## 🛠️ Quick Start

### 1. Run the Desktop 3D Split-Screen App
```powershell
python activity_tracker/app.py
```
* **`V`**: Toggle 3D Split-Screen Dual View vs. Fullscreen 3D vs. Camera Only.
* **`A` / `D`**: Orbit the 3D camera left/right around your avatar (360° rotation).
* **`W` / `S`**: Adjust 3D view pitch up/down.
* **`K`**: Launch the 3-Step Calibration Wizard anytime.
* **`C`**: Quick-calibrate sitting height (locks current height as $0\%$).
* **`Q`**: Quit the application.

### 2. (Optional) Run the WebGL 3D Browser Digital Twin (Three.js)
If you want to view your 3D digital twin in your browser with full mouse drag/orbit controls:
```powershell
python activity_tracker/server_3d.py
```
Then open: **`http://localhost:8000`** in your browser.
