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
