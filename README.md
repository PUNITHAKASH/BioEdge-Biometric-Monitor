# BioEdge: Multi-Modal Biometric & Ergonomic Monitoring

## Project Overview
**BioEdge** is an advanced Edge AI application developed for the NVIDIA Jetson Nano. It focuses on Human-Computer Interaction (HCI) by monitoring user attention, fatigue, and posture in real-time. By utilizing deep learning landmark detection and sensor fusion, the system provides a comprehensive "Focus Score" and critical safety alerts for workstation environments.

---

## The "Heart" of BioEdge (Decision Logic)
Unlike simple detection scripts, BioEdge uses **Sensor Fusion** to analyze multiple biometric markers simultaneously:

| Feature | Data Source | Logic / Threshold |
| :--- | :--- | :--- |
| **Drowsiness** | Face Mesh | EAR (Eye Aspect Ratio) < 0.23 for 8+ frames |
| **Distraction** | Iris Tracking | Gaze Vector deviation > 35% from center |
| **Poor Posture**| Skeletal Pose | Ear-to-Shoulder horizontal inclination > 70px |
| **Sleep Alert** | Fusion | (Poor Posture) + (Drowsiness) = **CRITICAL ALERT** |
| **Blink Reminder**| Timer | Alert triggered if no blink detected for 10s |

---

## Mathematical Foundations

### Eye Aspect Ratio (EAR)
To determine eyelid closure accurately, we calculate the EAR using the Euclidean distance between vertical eye landmarks and horizontal landmarks:

$$EAR = \frac{||p2 - p6|| + ||p3 - p5||}{2||p1 - p4||}$$

### Postural Inclination
The system monitors the horizontal displacement ($\Delta x$) between the ear (Landmark 7) and shoulder (Landmark 11). An increase in $\Delta x$ signifies a "forward head poke," which is the primary indicator of ergonomic strain and slouching.

---

## Tech Stack & Requirements
- **Hardware:** NVIDIA Jetson Nano, USB Webcam.
- **AI Models:** MediaPipe BlazeFace, FaceMesh (with Iris Refinement), and BlazePose.
- **Language:** Python 3.x
- **Core Libraries:** OpenCV, NumPy, MediaPipe.

---



###  Upcoming Phase: UI & UX Enhancement
The next development sprint focuses on transitioning from a developer-centric HUD to a production-ready dashboard:
- **Alert Visuals:** Implementing full-screen color overlays (Red/Amber) for critical safety alerts.
- **Toggle Controls:** Adding a GUI sidebar to adjust EAR and Posture sensitivity thresholds without restarting the script.
- **Telemetry Dashboard:** Real-time graphing of blink frequency and attention scores.


---

##  Project Roadmap (WIP Status)

### Phase 1: Core Logic (Current) - COMPLETED
- [x] Multi-modal sensor fusion (Face Mesh + Pose Exoskeleton).
- [x] Persistent ID tracking logic for multi-user stability.
- [x] Mathematical modeling for EAR and Neck Inclination.

### Phase 2: UI & Data Integration (Due: May 5)
- **Advanced HUD:** Transitioning to a human-centric dashboard with semi-transparent overlays.
- **Alert Visuals:** Full-screen color-coded alerts for sleep and distraction.
- **CSV Data Logging:** Automated background logging for session analytics.
- "Development of ui_overlay.py for a modular Frontend/Backend separation."

### Phase 3: Final Testing & Thesis (Due: May 15)
- **Edge Optimization:** Maximizing FPS on Jetson Nano hardware.
- **Stress Testing:** Validating system accuracy over long-duration monitoring.
- **Final Submission:** Finalizing documentation and video demonstration.


## Development Progress (WIP Screenshots)

| Feature | System Visualization | Description |
| :--- | :--- | :--- |
| **Ocular Tracking** | ![EAR Tracking](./media/ear_demo.png) | Real-time Eye Aspect Ratio (EAR) calculation for drowsiness detection. |
| **Skeletal Pose** | ![Posture Demo](./media/pose_demo.png) | Exoskeleton mapping for neck inclination and ergonomic analysis. |
| **Gaze Logic** | ![Gaze Demo](./media/gaze_demo.png) | Iris landmark isolation for distraction monitoring. |
| **Persistent ID** | ![ID Tracking](./media/id_demo.png) | Centroid-based tracking to maintain user identity across frames. |
| **Sensor Fusion** | ![Sleep Alert](./media/sleep_demo.png) | Logic-gate trigger: Combining EAR and Posture for 'SLEEP!!' alerts. |
| **Blink Monitor** | ![Blink Demo](./media/blink_demo.png) | Temporal tracking: Triggers 'BLINK ALERT' if no blink is detected for 10s. |


 ---
## Installation & Usage

### 1. Prerequisites
Ensure your Jetson Nano is running JetPack 4.6+ and has a camera connected.

### 2. Setup
```bash
# Clone the repository
git clone [https://github.com/PUNITHAKASH/BioEdge-Biometric-Monitor.git](https://github.com/PUNITHAKASH/BioEdge-Biometric-Monitor.git)
cd BioEdge-Biometric-Monitor





# Install dependencies
pip3 install -r requirements.txt





