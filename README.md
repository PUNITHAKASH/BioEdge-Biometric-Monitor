# BioEdge: Multi-Modal Biometric & Ergonomic Monitoring

##  Project Overview
BioEdge is an advanced Edge AI application developed for the NVIDIA Jetson Nano. It focuses on Human-Computer Interaction (HCI) by monitoring user attention, fatigue, and posture in real-time using deep learning landmark detection.

##  The "Heart" of BioEdge (Decision Logic)
Unlike simple detection scripts, BioEdge uses **Sensor Fusion** to make high-level decisions:

| Feature | Data Source | Logic |
| :--- | :--- | :--- |
| **Drowsiness** | Face Mesh | EAR (Eye Aspect Ratio) < 0.23 for 8+ frames |
| **Distraction** | Iris Tracking | Gaze Vector deviation > 35% from center |
| **Poor Posture**| Skeletal Pose | Ear-to-Shoulder horizontal inclination > 70px |
| **Sleep Alert** | Fusion | (Poor Posture) + (Drowsiness) = **CRITICAL ALERT** |

##  Tech Stack
- **Hardware:** NVIDIA Jetson Nano, USB Camera
- **AI Models:** MediaPipe BlazeFace, FaceMesh (Refined Iris), and BlazePose
- **Language:** Python 3.x
- **Libraries:** OpenCV, NumPy

- ##  Mathematical Foundations

### Eye Aspect Ratio (EAR)
To determine drowsiness, we calculate the EAR using the Euclidean distance between vertical eye landmarks ($p_2, p_6, p_3, p_5$) and horizontal landmarks ($p_1, p_4$):

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2||p_1 - p_4||}$$



### Postural Inclination
The system monitors the horizontal displacement ($\Delta x$) between the ear and shoulder landmarks to detect a "forward head poke" (text neck), a primary indicator of poor ergonomic posture.
