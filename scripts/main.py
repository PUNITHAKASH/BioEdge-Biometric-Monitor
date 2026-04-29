import cv2
import mediapipe as mp
import numpy as np
import time
import math

# 1. INITIALIZE SENSORS
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5)
# refine_landmarks=True is REQUIRED for Iris/Gaze tracking
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=4, refine_landmarks=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Eye and Iris Indices
L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33, 160, 158, 133, 153, 144]
L_IRIS = [474, 475, 476, 477] # Left Iris
R_IRIS = [469, 470, 471, 472] # Right Iris

def get_ear(landmarks, eye_indices, w, h):
pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
v1 = np.linalg.norm(pts[1] - pts[5])
v2 = np.linalg.norm(pts[2] - pts[4])
h1 = np.linalg.norm(pts[0] - pts[3])
return (v1 + v2) / (2.0 * h1)

user_db = {}
NEXT_ID = 0
cap = cv2.VideoCapture(0)

while cap.isOpened():
ret, frame = cap.read()
if not ret: break
frame = cv2.flip(frame, 1)
h, w, _ = frame.shape
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

pose_res = pose.process(rgb_frame)
face_res = face_mesh.process(rgb_frame)

if not face_res.multi_face_landmarks:
cv2.putText(frame, "NO PERSON DETECTED", (20, 60), 1, 2, (100, 100, 100), 3)
else:
if pose_res.pose_landmarks:
mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

for face_lms in face_res.multi_face_landmarks:
lms = face_lms.landmark

# --- 1. PERSISTENT ID ---
cx, cy = int(lms[1].x * w), int(lms[1].y * h)
assigned_id = None
for uid, data in user_db.items():
dist = math.sqrt((cx - data['last_pos'][0])**2 + (cy - data['last_pos'][1])**2)
if dist < 150: assigned_id = uid; break
if assigned_id is None:
assigned_id = NEXT_ID
user_db[assigned_id] = {'last_blink': time.time(), 'drowsy_f': 0, 'last_pos': (cx, cy)}
NEXT_ID += 1
user_db[assigned_id]['last_pos'] = (cx, cy)

# --- 2. SKELETAL POSTURE (EXO) ---
is_poor_posture = False
neck_inclination = 0
if pose_res.pose_landmarks:
plm = pose_res.pose_landmarks.landmark
# Bilateral check: use ears and shoulders from the skeleton
ear_l = np.array([plm[7].x * w, plm[7].y * h])
sh_l = np.array([plm[11].x * w, plm[11].y * h])
neck_inclination = abs(ear_l[0] - sh_l[0])
if neck_inclination > 70: is_poor_posture = True

# --- 3. EYE STATUS (DROWSY & BLINK) ---
ear = (get_ear(lms, L_EYE, w, h) + get_ear(lms, R_EYE, w, h)) / 2.0
is_eyes_closed = False
if ear < 0.23:
is_eyes_closed = True
user_db[assigned_id]['last_blink'] = time.time()
user_db[assigned_id]['drowsy_f'] += 1
else: user_db[assigned_id]['drowsy_f'] = 0

# --- 4. GAZE TRACKING (EYE-BASED DISTRACTION) ---
# Compare iris center to eye corners
iris_center = lms[473].x # Center of left iris
eye_inner = lms[362].x
eye_outer = lms[263].x
# Ratio of 0.5 is centered. < 0.35 or > 0.65 is looking away.
gaze_ratio = (iris_center - eye_inner) / (eye_outer - eye_inner + 1e-6)
is_distracted = True if (gaze_ratio < 0.35 or gaze_ratio > 0.65) else False

# --- 5. LOGIC MERGE ---
t_since_blink = time.time() - user_db[assigned_id]['last_blink']
status, color = "FOCUSED", (0, 255, 0)

if is_poor_posture and user_db[assigned_id]['drowsy_f'] > 8:
status, color = "SLEEP!!", (0, 0, 255)
elif is_eyes_closed and user_db[assigned_id]['drowsy_f'] > 8:
status, color = "DROWSY", (0, 165, 255)
elif is_distracted:
status, color = "DISTRACTED", (0, 255, 255)
elif is_poor_posture:
status, color = "POOR POSTURE", (0, 120, 255)
elif t_since_blink > 10:
status, color = "BLINK ALERT", (255, 0, 255)

# --- 6. EYE DEBUG DIALOG ---
bx, by = int(lms[168].x * w), int(lms[168].y * h)
y1, y2, x1, x2 = max(0, by-35), min(h, by+35), max(0, bx-70), min(w, bx+70)
eye_roi = frame[y1:y2, x1:x2].copy()
if eye_roi.size > 0:
eye_roi = cv2.resize(eye_roi, (140, 70))
# Display box above head
dy1, dy2, dx1, dx2 = max(0, cy-190), max(70, cy-120), max(0, cx-70), min(w, cx+70)
if (dy2-dy1) == 70 and (dx2-dx1) == 140:
frame[dy1:dy2, dx1:dx2] = eye_roi
cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (255, 255, 255), 1)

cv2.putText(frame, f"ID {assigned_id}: {status}", (cx-80, cy-100), 1, 1.5, color, 2)
cv2.putText(frame, f"Gaze: {gaze_ratio:.2f}", (cx-80, cy-80), 1, 0.8, (255,255,255), 1)

cv2.imshow('BioEdge Multi-Person Final', frame)
if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
