# VisionX — Real-Time AI Vision, Facial Expression, Hand Gesture & Air Interaction System

VisionX is a real-time Computer Vision, Machine Learning, and Deep Learning web laboratory powered by on-device **MediaPipe Tasks Vision** (`FaceLandmarker` + `HandLandmarker`), custom geometric CV shape classification, and an in-browser neural network training workbench.

## Features

1. **Real-Time Facial Landmarks & Expression Patterns (`Overview`)**
   - 478-point facial contour & iris mesh, Eye Aspect Ratio (EAR) blink counter, Mouth Aspect Ratio, and 3D Head Pose (Yaw / Pitch / Roll).
   - 7-class facial expression pattern prediction (`Neutral`, `Happy`, `Surprised`, `Angry`, `Sad`, `Fearful`, `Disgusted`) smoothed over 52 ARKit blendshapes.
2. **Dual-Hand Tracking & Multi-Model Gesture Recognition (`Overview` & `ML Lab`)**
   - Tracks up to 2 hands (21 landmarks per hand), handedness, finger extension states, pinch distance, and dynamic swipe motion.
   - Switch live between **Rule-Based Geometry**, **K-Nearest Neighbors (KNN, k=5)**, and a **3-Layer MLP Neural Network (`28 -> 32 -> 24 -> K`)** trained directly in the browser on 28-D engineered hand features.
3. **Air Drawing & Custom Shape Naming (`Air Studio`)**
   - Smooth fingertip air drawing (`Pointing Index` or `Thumb-Index Pinch`), brush/eraser controls, and PNG export.
   - Geometric classification (`Circle`, `Ellipse`, `Triangle`, `Square`, `Rectangle`, `Pentagon`, `Hexagon`, `Star`, `Arrow`, `Line`, `Freeform`) plus persistent **Custom Shape Naming** (`localStorage`).
4. **Touchless YouTube Control (`YouTube`)**
   - Gesture-driven playback, volume, mute, ±10s seeking, theater mode, and video queue scrolling with safety arming, cooldown guardrails, and `ESC` Emergency Stop.
5. **Touchless Virtual Keyboard & Google Search (`Virtual Search`)**
   - Index-finger aimed QWERTY keyboard with configurable **Hover Dwell Timer** or **Pinch-to-Type**, live Wikipedia search previews, and direct Google Search launch.

## Run Locally in VS Code

### Prerequisites
- **Node.js 18+ or 20 LTS** ([nodejs.org](https://nodejs.org))
- **VS Code** and **Google Chrome** or **Microsoft Edge**

### Steps
1. Extract the downloaded project ZIP and open the folder in **VS Code** (`File -> Open Folder...`).
2. Open the integrated terminal (`Ctrl + ~`).
3. Install dependencies and start the local server:
   ```bash
   npm install --legacy-peer-deps
npm run dev
   ```
   *(Or press **`Ctrl + Shift + B`** in VS Code to start the default dev server task.)*
4. Open **`http://localhost:3000`** in Chrome/Edge and click **Start Camera** (allow webcam permissions).

*(All MediaPipe WASM binaries and `.task` model weights are already included locally inside `public/mediapipe/`—no Python or external model downloads required.)*

### Windows PowerShell Note
If PowerShell blocks `npm`, switch the VS Code terminal profile to **Command Prompt (`cmd`)** or run:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
