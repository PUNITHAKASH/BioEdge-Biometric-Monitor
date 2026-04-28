import cv2
import mediapipe as mp
import numpy as np
import time
import math

# AI INITIALIZATION
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=4, refine_landmarks=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Landmarks for Eyes and Iris
L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33, 160, 158, 133, 153, 144]
L_IRIS = [473] 
R_IRIS = [468]

def get_ear(landmarks, eye_indices, w, h):
    """Calculate Eye Aspect Ratio (EAR) for blink detection."""
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h1 = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h1)

# Persistent Tracking Data
user_db = {} 
NEXT_ID = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process both Face and Pose (Sensor Fusion)
    face_res = face_mesh.process(rgb_frame)
    pose_res = pose.process(rgb_frame)

    if face_res.multi_face_landmarks:
        # Draw Pose Exoskeleton first
        if pose_res.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for face_lms in face_res.multi_face_landmarks:
            lms = face_lms.landmark
            
            # --- PERSISTENT ID LOGIC (Centroid Tracking) ---
            cx, cy = int(lms[1].x * w), int(lms[1].y * h)
            assigned_id = None
            for uid, data in user_db.items():
                dist = math.sqrt((cx - data['last_pos'][0])**2 + (cy - data['last_pos'][1])**2)
                if dist < 150: # Threshold for ID lock
                    assigned_id = uid
                    break
            
            if assigned_id is None:
                assigned_id = NEXT_ID
                user_db[assigned_id] = {'last_blink': time.time(), 'drowsy_f': 0, 'last_pos': (cx, cy)}
                NEXT_ID += 1
            
            user_db[assigned_id]['last_pos'] = (cx, cy)

            # SKELETAL POSTURE (NECK INCLINATION)
            is_poor_posture = False
            if pose_res.pose_landmarks:
                plm = pose_res.pose_landmarks.landmark
                # Check distance between ear (7) and shoulder (11)
                ear_l = np.array([plm[7].x * w, plm[7].y * h])
                sh_l = np.array([plm[11].x * w, plm[11].y * h])
                neck_dist = abs(ear_l[0] - sh_l[0])
                if neck_dist > 70: # Threshold for slouching
                    is_poor_posture = True

            # OCULAR ANALYSIS (EAR) 
            ear = (get_ear(lms, L_EYE, w, h) + get_ear(lms, R_EYE, w, h)) / 2.0
            
            # GAZE TRACKING 
            iris_x = lms[473].x
            # Ratio of iris position between inner and outer corners
            gaze_ratio = (iris_x - lms[362].x) / (lms[263].x - lms[362].x + 1e-6)
            is_distracted = (gaze_ratio < 0.35 or gaze_ratio > 0.65)

            # STATE MANAGEMENT
            if ear < 0.23: # Eyes Closed
                user_db[assigned_id]['drowsy_f'] += 1
            else:
                user_db[assigned_id]['last_blink'] = time.time()
                user_db[assigned_id]['drowsy_f'] = 0

            #  DECISION ENGINE
            t_since_blink = time.time() - user_db[assigned_id]['last_blink']
            status, color = "FOCUSED", (0, 255, 0)

            if is_poor_posture and user_db[assigned_id]['drowsy_f'] > 8:
                status, color = "SLEEP!!", (0, 0, 255)
            elif user_db[assigned_id]['drowsy_f'] > 8:
                status, color = "DROWSY", (0, 165, 255)
            elif is_distracted:
                status, color = "DISTRACTED", (0, 255, 255)
            elif is_poor_posture:
                status, color = "POOR POSTURE", (0, 120, 255)
            elif t_since_blink > 10:
                status, color = "BLINK ALERT", (255, 0, 255)

            #  VISUAL HUD 
            # Display ID and Status
            cv2.putText(frame, f"ID {assigned_id}: {status}", (cx-80, cy-110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # Display Metrics for Thesis Proof
            cv2.putText(frame, f"EAR: {ear:.2f} Gaze: {gaze_ratio:.2f}", (cx-80, cy-85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow('BioEdge Master Suite', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
