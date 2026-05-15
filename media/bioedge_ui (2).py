import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from itertools import combinations

import cv2
import mediapipe as mp
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
    QWidget, QDialog, QFrame, QGridLayout, QPushButton,
    QComboBox, QListWidget, QListWidgetItem
)
from PyQt5.QtGui  import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker


UI_SCALE = 0.75

def S(px: int) -> int: return max(1, int(px * UI_SCALE))
def F(pt: int) -> int: return max(6, int(pt * UI_SCALE))


MAX_FACES            = 5
FACE_DETECT_CONF     = 0.35
FACE_TRACK_CONF      = 0.35

FACE_SIZE_NEAR       = 0.06
FACE_SIZE_MED        = 0.02
FACE_SIZE_FAR        = 0.006

EAR_THRESH           = 0.23
DROWSY_FRAME_THRESH  = 12
RECOVERY_FRAMES      = 3
BLINK_MIN_DUR        = 0.04
BLINK_MAX_DUR        = 0.40
BLINK_COUNTDOWN      = 10.0

SPINE_ANGLE_THRESH   = 0.12
FWD_HEAD_THRESH      = 0.06
SHOULDER_TILT_THRESH = 0.04
HIP_TILT_THRESH      = 0.03

GAZE_RATIO_LOW       = 0.30
GAZE_RATIO_HIGH      = 0.75
GAZE_FRAME_THRESH    = 8

FACE_ID_THRESHOLD    = 0.82
SIGNATURE_BANK_SIZE  = 6
MAX_HISTORY          = 30

HUD_W = 224
HUD_H = 96


L_EYE       = [362, 385, 387, 263, 373, 380]
R_EYE       = [33,  160, 158, 133, 153, 144]
L_EYE_OUTER = 362;  L_EYE_INNER = 263
R_EYE_INNER = 133;  R_EYE_OUTER = 33
IRIS_L = 468;  IRIS_R = 473

POSE_IDX = {
    "nose":0,  "l_ear":7,  "r_ear":8,
    "l_sho":11,"r_sho":12, "l_elb":13,"r_elb":14,
    "l_wri":15,"r_wri":16, "l_hip":23,"r_hip":24,
    "l_kne":25,"r_kne":26, "l_ank":27,"r_ank":28,
}

SKELETON_SEGMENTS = {
    "head":     ((160, 240, 255), (0, 165, 255)),
    "neck":     ((160, 240, 255), (0, 165, 255)),
    "shoulder": ((80,  200, 255), (0, 165, 255)),
    "l_arm":    ((60,  255, 180), (0, 165, 255)),
    "r_arm":    ((255, 160,  80), (0, 165, 255)),
    "torso":    ((180, 100, 255), (0, 165, 255)),
    "l_leg":    ((60,  255, 180), (0, 165, 255)),
    "r_leg":    ((255, 160,  80), (0, 165, 255)),
}

SKELETON_CONNECTIONS = [
    (7,  0,  "head"),
    (8,  0,  "head"),
    (11, 0,  "neck"),
    (12, 0,  "neck"),
    (11, 12, "shoulder"),
    (11, 13, "l_arm"),
    (13, 15, "l_arm"),
    (12, 14, "r_arm"),
    (14, 16, "r_arm"),
    (11, 23, "torso"),
    (12, 24, "torso"),
    (23, 24, "torso"),
    (23, 25, "l_leg"),
    (25, 27, "l_leg"),
    (24, 26, "r_leg"),
    (26, 28, "r_leg"),
]

SKELETON_CONNECTIONS_PLAIN = [(a,b) for a,b,_ in SKELETON_CONNECTIONS]

PID_COLORS_BGR = [
    (60, 200, 255),
    (60, 255, 180),
    (255,  80, 180),
    (80,  180, 255),
    (80,  255,  80),
    (40,  140, 255),
    (180,  80, 255),
    (220, 220,  40),
    (100, 255, 255),
    (255, 200, 100),
]

STATUS_CSS = {
    "SLEEP!!":      "#FF2020",
    "DROWSY":       "#FF8C00",
    "POOR POSTURE": "#FFA500",
    "BLINK ALERT":  "#FF00FF",
    "DISTRACTED":   "#00DCDC",
    "FOCUSED":      "#50DC50",
    "AWAY":         "#606060",
    "TOO FAR":      "#444466",
}

COL_SKEL_WARN = (0,  165, 255)
COL_JOINT     = (255,255, 255)
COL_NO_PERSON = (100,100, 100)
COL_IRIS_OK   = (100,255, 150)
COL_IRIS_WARN = (0,  220, 220)
COL_MONITOR   = (0,  255, 128)

DIST_COLORS = {
    "NEAR":    (80,  220,  80),
    "MED":     (80,  200, 255),
    "FAR":     (0,   165, 255),
    "TOO FAR": (80,   80, 140),
}


@dataclass
class AlertEvent:
    timestamp: float
    status:    str

@dataclass
class PersonState:
    pid:   str
    color: Tuple[int,int,int]

    last_blink_time: float = field(default_factory=time.time)
    drowsy_frames:   int   = 0
    recovery_f:      int   = 0
    eye_close_start: Optional[float] = None
    eye_was_closed:  bool  = False
    gaze_off_frames: int   = 0
    posture_bad:     bool  = False
    posture_note:    str   = ""

    ear:          float = 0.0
    l_ear:        float = 0.0
    r_ear:        float = 0.0
    gaze_l:       float = 0.5
    gaze_r:       float = 0.5
    status:       str   = "FOCUSED"
    countdown:    float = BLINK_COUNTDOWN
    eye_crop:     Optional[np.ndarray] = None

    face_size_pct: float = 0.0
    dist_zone:     str   = "NEAR"
    reid_score:    float = 1.0

    is_live:      bool  = False
    is_monitored: bool  = False

    last_seen:    float = field(default_factory=time.time)
    first_seen:   float = field(default_factory=time.time)
    total_frames: int   = 0

    history: List[AlertEvent] = field(default_factory=list)
    _prev_status: str = field(default="", repr=False)

    def record_status(self, now: float):
        if self.status != self._prev_status:
            self.history.append(AlertEvent(now, self.status))
            if len(self.history) > MAX_HISTORY:
                self.history = self.history[-MAX_HISTORY:]
            self._prev_status = self.status

    @property
    def css_color(self) -> str:
        b, g, r = self.color
        return f"rgb({r},{g},{b})"

    @property
    def status_css(self) -> str:
        return STATUS_CSS.get(self.status, "#E2E8F0")

    def status_bgr(self) -> Tuple[int,int,int]:
        c = self.status_css.lstrip("#")
        r,g,b = int(c[0:2],16), int(c[2:4],16), int(c[4:6],16)
        return (b,g,r)

    @property
    def metrics_reliable(self) -> bool:
        return self.dist_zone in ("NEAR", "MED")


def get_ear(lms, idx, w, h) -> float:
    pts = [np.array([lms[i].x*w, lms[i].y*h]) for i in idx]
    v1 = np.linalg.norm(pts[1]-pts[5])
    v2 = np.linalg.norm(pts[2]-pts[4])
    h1 = np.linalg.norm(pts[0]-pts[3])
    return (v1+v2)/(2.0*h1) if h1>1e-6 else 0.3

def gaze_ratio(lms, iris_id, outer_id, inner_id, w, h) -> float:
    ix = lms[iris_id].x*w
    ox = lms[outer_id].x*w
    nx = lms[inner_id].x*w
    ew = abs(nx-ox)
    return (ix-min(ox,nx))/ew if ew>1 else 0.5

def face_bbox(lms, w, h):
    xs = [lms[i].x*w for i in range(468)]
    ys = [lms[i].y*h for i in range(468)]
    return int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))

def face_area(lms, w, h) -> float:
    x1,y1,x2,y2 = face_bbox(lms,w,h)
    return float((x2-x1)*(y2-y1))

def dist_zone_from_size(pct: float) -> str:
    if pct >= FACE_SIZE_NEAR: return "NEAR"
    if pct >= FACE_SIZE_MED:  return "MED"
    if pct >= FACE_SIZE_FAR:  return "FAR"
    return "TOO FAR"


def face_sig(lms) -> np.ndarray:
    anchor_ids = [10, 152, 234, 454, 33, 263, 133, 362, 1, 61, 291, 168, 199, 4]
    pts = np.array([[lms[i].x, lms[i].y] for i in anchor_ids], dtype=np.float32)
    eye_l = pts[4]
    eye_r = pts[5]
    iod   = np.linalg.norm(eye_l - eye_r)
    if iod < 1e-6:
        iod = 1.0
    n = len(pts)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(np.linalg.norm(pts[i] - pts[j]) / iod)
    return np.array(dists, dtype=np.float32)


def cosine_sim(a, b) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a,b)/d) if d > 1e-8 else 0.0

def cv_to_pixmap(img, tw, th) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ih,iw,ch = rgb.shape
    qi = QImage(rgb.data, iw, ih, ch*iw, QImage.Format_RGB888)
    return QPixmap.fromImage(qi).scaled(
        tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def hud_position(hx: int, hy: int, w: int, h: int,
                 placed: List[Tuple[int,int,int,int]]) -> Tuple[int,int]:
    bw, bh = HUD_W, HUD_H
    bx = max(0, min(hx + 44, w - bw - 2))
    by = max(2, min(hy - 100, h - bh - 2))
    max_tries = 8
    for _ in range(max_tries):
        overlap = False
        for (px1,py1,px2,py2) in placed:
            if not (bx+bw < px1 or bx > px2 or by+bh < py1 or by > py2):
                overlap = True
                by = py2 + 4
                break
        if not overlap:
            break
    bx = max(0, min(bx, w - bw - 2))
    by = max(0, min(by, h - bh - 2))
    return bx, by


class PersonRegistry:
    def __init__(self):
        self._entries: List[Dict] = []
        self._next = 1
        self._mutex = QMutex()
        self.monitored_pid: Optional[str] = None

    def set_monitor(self, pid: Optional[str]):
        with QMutexLocker(self._mutex):
            self.monitored_pid = pid
            for e in self._entries:
                e["state"].is_monitored = (e["state"].pid == pid)

    def begin_frame(self):
        with QMutexLocker(self._mutex):
            for e in self._entries:
                e["state"].is_live = False

    def assign_all(self, sigs: List[np.ndarray], dist_zones: List[str],
                   now: float) -> List[PersonState]:
        with QMutexLocker(self._mutex):
            n_faces   = len(sigs)
            n_entries = len(self._entries)
            results   = [None] * n_faces

            if n_entries > 0:
                sim_matrix = np.zeros((n_faces, n_entries), dtype=np.float32)
                for f, sig in enumerate(sigs):
                    for e_idx, entry in enumerate(self._entries):
                        best = 0.0
                        for stored in entry["sigs"]:
                            s = cosine_sim(sig, stored)
                            if s > best:
                                best = s
                        sim_matrix[f, e_idx] = best

                assigned_faces   = set()
                assigned_entries = set()

                pairs = [(sim_matrix[f,e], f, e)
                         for f in range(n_faces)
                         for e in range(n_entries)]
                pairs.sort(reverse=True)

                for sim_val, f_idx, e_idx in pairs:
                    if sim_val < FACE_ID_THRESHOLD:
                        break
                    if f_idx in assigned_faces or e_idx in assigned_entries:
                        continue
                    entry = self._entries[e_idx]
                    bank  = entry["sigs"]
                    if len(bank) < SIGNATURE_BANK_SIZE:
                        bank.append(sigs[f_idx].copy())
                    else:
                        idx = entry["bank_idx"] = (entry["bank_idx"]+1) % SIGNATURE_BANK_SIZE
                        bank[idx] = sigs[f_idx].copy()
                    st = entry["state"]
                    st.last_seen    = now
                    st.is_live      = True
                    st.total_frames += 1
                    st.reid_score   = float(sim_val)
                    results[f_idx]  = st
                    assigned_faces.add(f_idx)
                    assigned_entries.add(e_idx)

            for f_idx in range(n_faces):
                if results[f_idx] is None:
                    pid   = f"P{self._next:03d}"
                    color = PID_COLORS_BGR[(self._next-1) % len(PID_COLORS_BGR)]
                    self._next += 1
                    st = PersonState(pid=pid, color=color,
                                     last_seen=now, first_seen=now,
                                     is_live=True, total_frames=1,
                                     reid_score=1.0)
                    st.is_monitored = (pid == self.monitored_pid)
                    self._entries.append({
                        "sigs":     [sigs[f_idx].copy()],
                        "bank_idx": 0,
                        "state":    st,
                    })
                    results[f_idx] = st

            return results

    def all_states(self) -> List[PersonState]:
        with QMutexLocker(self._mutex):
            return sorted([e["state"] for e in self._entries],
                          key=lambda s: s.pid)

    def get_monitored_state(self) -> Optional[PersonState]:
        with QMutexLocker(self._mutex):
            if self.monitored_pid is None:
                return None
            for e in self._entries:
                if e["state"].pid == self.monitored_pid:
                    return e["state"]
            return None


class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, list)

    def __init__(self, registry: PersonRegistry):
        super().__init__()
        self._running  = True
        self._registry = registry

    def stop(self):
        self._running = False

    @staticmethod
    def _draw_skeleton(frame, pose_lm, w, h, bad_joints, base_col):
        lm = pose_lm.landmark
        VIS = 0.35

        def pt(idx):
            return (int(lm[idx].x*w), int(lm[idx].y*h))

        def visible(idx):
            return lm[idx].visibility > VIS

        for (a, b, seg) in SKELETON_CONNECTIONS:
            if not (visible(a) and visible(b)):
                continue
            pa, pb = pt(a), pt(b)
            bad = (a in bad_joints or b in bad_joints)
            norm_col, warn_col = SKELETON_SEGMENTS[seg]
            col = warn_col if bad else norm_col
            shadow = (max(0,col[0]//4), max(0,col[1]//4), max(0,col[2]//4))
            cv2.line(frame, pa, pb, shadow, 5, cv2.LINE_AA)
            cv2.line(frame, pa, pb, col, 2, cv2.LINE_AA)
            blen = np.hypot(pb[0]-pa[0], pb[1]-pa[1])
            if blen > 30:
                bright = tuple(min(255, int(c*1.6)) for c in col)
                dx, dy = (pb[0]-pa[0])/blen, (pb[1]-pa[1])/blen
                pa2 = (int(pa[0]+dx*4), int(pa[1]+dy*4))
                pb2 = (int(pb[0]-dx*4), int(pb[1]-dy*4))
                cv2.line(frame, pa2, pb2, bright, 1, cv2.LINE_AA)

        vis_ids = {idx for (a,b,_) in SKELETON_CONNECTIONS for idx in (a,b)}
        ENDPOINTS = {15, 16, 27, 28}

        for idx in vis_ids:
            if not visible(idx):
                continue
            px, py = pt(idx)
            bad = idx in bad_joints
            seg_col = (80, 80, 80)
            for (a, b, seg) in SKELETON_CONNECTIONS:
                if idx in (a, b):
                    seg_col = SKELETON_SEGMENTS[seg][1 if bad else 0]
                    break
            warn = (0, 165, 255) if bad else None
            if idx in ENDPOINTS:
                r = 6
                pts_d = np.array([
                    [px,    py-r],
                    [px+r,  py  ],
                    [px,    py+r],
                    [px-r,  py  ],
                ], dtype=np.int32)
                cv2.fillPoly(frame, [pts_d], warn if warn else seg_col)
                cv2.polylines(frame, [pts_d], True, (255,255,255), 1, cv2.LINE_AA)
            else:
                glow = tuple(min(255, int(c*0.5)) for c in seg_col)
                cv2.circle(frame,(px,py), 9, glow, -1, cv2.LINE_AA)
                cv2.circle(frame,(px,py), 6, warn if warn else seg_col, -1, cv2.LINE_AA)
                cv2.circle(frame,(px,py), 3, (255,255,255), -1, cv2.LINE_AA)
                cv2.circle(frame,(px,py), 9, warn if warn else seg_col, 1, cv2.LINE_AA)

        if visible(0) and visible(7) and visible(8):
            nose  = pt(0)
            lear  = pt(7)
            rear  = pt(8)
            ear_dist = int(np.hypot(rear[0]-lear[0], rear[1]-lear[1]) * 0.65)
            ear_dist = max(ear_dist, 18)
            hcol  = SKELETON_SEGMENTS["head"][0]
            cv2.circle(frame, nose, ear_dist+2,
                       (hcol[0]//4, hcol[1]//4, hcol[2]//4), 4, cv2.LINE_AA)
            cv2.circle(frame, nose, ear_dist, hcol, 2, cv2.LINE_AA)
            cv2.circle(frame, nose, ear_dist-4,
                       tuple(min(255,int(c*0.5)) for c in hcol), 1, cv2.LINE_AA)

        if all(visible(i) for i in [11,12,23,24]):
            ls, rs = pt(11), pt(12)
            lh, rh = pt(23), pt(24)
            mid_sho = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
            mid_hip = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
            tcol = SKELETON_SEGMENTS["torso"][0]
            cv2.line(frame, mid_sho, mid_hip,
                     (tcol[0]//4, tcol[1]//4, tcol[2]//4), 5, cv2.LINE_AA)
            cv2.line(frame, mid_sho, mid_hip, tcol, 2, cv2.LINE_AA)
            cv2.line(frame, mid_sho, mid_hip,
                     tuple(min(255,int(c*1.6)) for c in tcol), 1, cv2.LINE_AA)
            msp = ((mid_sho[0]+mid_hip[0])//2, (mid_sho[1]+mid_hip[1])//2)
            cv2.circle(frame, msp, 7, (tcol[0]//2, tcol[1]//2, tcol[2]//2),
                       -1, cv2.LINE_AA)
            cv2.circle(frame, msp, 4, tcol, -1, cv2.LINE_AA)
            cv2.circle(frame, msp, 2, (255,255,255), -1, cv2.LINE_AA)

    @staticmethod
    def _eye_crop(frame, lms, w, h,
                  gaze_l, gaze_r, distracted,
                  l_ear, r_ear, reliable) -> np.ndarray:
        try:
            ex  = sorted([int(lms[i].x*w) for i in [L_EYE_OUTER,R_EYE_OUTER]])
            ey  = sorted([int(lms[i].y*h) for i in [159,145]])
            pad = 55
            y1,y2 = max(0,ey[0]-pad), min(h,ey[1]+pad)
            x1,x2 = max(0,ex[0]-pad), min(w,ex[1]+pad)
            crop = frame[y1:y2,x1:x2].copy()
            if crop.size==0: raise ValueError
            ch2,cw2 = crop.shape[:2]
            sx = cw2/max(1,x2-x1)
            def tc(i):
                return (int((lms[i].x*w-x1)*sx),
                        int((lms[i].y*h-y1)*sx))
            ic = COL_IRIS_WARN if distracted else COL_IRIS_OK
            for iid in [IRIS_L,IRIS_R]:
                ix,iy = tc(iid)
                cv2.circle(crop,(ix,iy),max(3,int(7*sx)), ic,-1,cv2.LINE_AA)
                cv2.circle(crop,(ix,iy),max(5,int(12*sx)),ic, 1,cv2.LINE_AA)
            for cid in [L_EYE_OUTER,L_EYE_INNER,R_EYE_INNER,R_EYE_OUTER]:
                cx2,cy2 = tc(cid)
                cv2.circle(crop,(cx2,cy2),3,(200,200,100),-1,cv2.LINE_AA)
            bary,barw = ch2-12,cw2-20
            cv2.rectangle(crop,(10,bary-5),(10+barw,bary+5),(30,30,45),-1)
            lo = int(10+GAZE_RATIO_LOW*barw)
            hi = int(10+GAZE_RATIO_HIGH*barw)
            cv2.rectangle(crop,(lo,bary-3),(hi,bary+3),(25,55,35),-1)
            cv2.line(crop,(lo,bary-6),(lo,bary+6),(80,80,80),1)
            cv2.line(crop,(hi,bary-6),(hi,bary+6),(80,80,80),1)
            for g in (gaze_l,gaze_r):
                gx = max(12,min(cw2-12,int(10+g*barw)))
                cv2.circle(crop,(gx,bary),5,ic,-1,cv2.LINE_AA)
            lc = (80,80,255) if l_ear<EAR_THRESH else (100,255,150)
            rc = (80,80,255) if r_ear<EAR_THRESH else (100,255,150)
            cv2.putText(crop,f"L:{l_ear:.2f}",(3,14),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,lc,1,cv2.LINE_AA)
            cv2.putText(crop,f"R:{r_ear:.2f}",(cw2-48,14),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,rc,1,cv2.LINE_AA)
            if not reliable:
                cv2.putText(crop,"TOO FAR — metrics unreliable",
                            (3,ch2-26),cv2.FONT_HERSHEY_SIMPLEX,
                            0.28,(80,80,180),1,cv2.LINE_AA)
            return crop
        except Exception:
            return np.zeros((70,160,3),dtype=np.uint8)

    @staticmethod
    def _draw_hud(frame, lms, w, h, st: PersonState,
                  placed_huds: List[Tuple[int,int,int,int]]):
        monitored = st.is_monitored
        hx = int(lms[10].x*w)
        hy = int(lms[10].y*h)
        bx, by = hud_position(hx, hy, w, h, placed_huds)
        bw, bh = HUD_W, HUD_H
        col = COL_MONITOR if monitored else st.color
        placed_huds.append((bx, by, bx+bw, by+bh))
        ov = frame.copy()
        cv2.rectangle(ov,(bx,by),(bx+bw,by+bh),(6,6,16),-1)
        cv2.addWeighted(ov,0.68,frame,0.32,0,frame)
        thick = 3 if monitored else 1
        cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),col,thick,cv2.LINE_AA)
        sc = st.status_bgr() if st.metrics_reliable else (80,80,130)
        cv2.putText(frame, st.status,
                    (bx+8,by+24), cv2.FONT_HERSHEY_DUPLEX,0.56,sc,2,cv2.LINE_AA)
        dz_col = DIST_COLORS.get(st.dist_zone,(80,80,80))
        cv2.putText(frame, f"[{st.dist_zone}] {st.face_size_pct*100:.1f}%",
                    (bx+8,by+40), cv2.FONT_HERSHEY_SIMPLEX,0.30,dz_col,1,cv2.LINE_AA)
        if st.metrics_reliable:
            cv2.putText(frame,f"Blink {st.countdown:.1f}s  EAR {st.ear:.2f}",
                        (bx+8,by+56),cv2.FONT_HERSHEY_SIMPLEX,0.30,(180,180,180),1,cv2.LINE_AA)
        else:
            cv2.putText(frame,"metrics N/A (too far)",
                        (bx+8,by+56),cv2.FONT_HERSHEY_SIMPLEX,0.28,(80,80,140),1,cv2.LINE_AA)
        reid_txt = "new" if st.reid_score >= 0.999 else f"{int(st.reid_score*100)}%"
        id_line = f"ID:{st.pid}{'[MON]' if monitored else ''}  conf:{reid_txt}"
        cv2.putText(frame, id_line,
                    (bx+8,by+70), cv2.FONT_HERSHEY_SIMPLEX,0.28,col,1,cv2.LINE_AA)
        if st.posture_note:
            cv2.putText(frame, st.posture_note.upper(),
                        (bx+8,by+84), cv2.FONT_HERSHEY_SIMPLEX,0.26,
                        COL_SKEL_WARN,1,cv2.LINE_AA)
        cv2.line(frame,(hx,hy-4),(bx,by+bh),col,1,cv2.LINE_AA)
        x1,y1,x2,y2 = face_bbox(lms,w,h)
        cv2.rectangle(frame,(x1,y1),(x2,y2),col,thick,cv2.LINE_AA)
        if monitored:
            cv2.rectangle(frame,(x1-3,y1-3),(x2+3,y2+3),col,1,cv2.LINE_AA)
            cv2.rectangle(frame,(x1-6,y1-6),(x2+6,y2+6),col,1,cv2.LINE_AA)
        dz_t = 1
        cv2.rectangle(frame,(x1-dz_t-2,y1-dz_t-2),(x2+dz_t+2,y2+dz_t+2),
                      dz_col,dz_t,cv2.LINE_AA)
        cv2.putText(frame, st.pid,(x1+4,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1,cv2.LINE_AA)

    @staticmethod
    def _update_person(st: PersonState, lms, w, h, now: float):
        st.l_ear = get_ear(lms,L_EYE,w,h)
        st.r_ear = get_ear(lms,R_EYE,w,h)
        st.ear   = (st.l_ear+st.r_ear)/2.0
        st.gaze_l = gaze_ratio(lms,IRIS_L,L_EYE_OUTER,L_EYE_INNER,w,h)
        st.gaze_r = gaze_ratio(lms,IRIS_R,R_EYE_INNER,R_EYE_OUTER,w,h)
        avg_g = (st.gaze_l+st.gaze_r)/2.0
        if avg_g<GAZE_RATIO_LOW or avg_g>GAZE_RATIO_HIGH:
            st.gaze_off_frames += 1
        else:
            st.gaze_off_frames = max(0,st.gaze_off_frames-1)
        if st.metrics_reliable:
            closed = st.ear < EAR_THRESH
            if closed:
                st.recovery_f = 0
                if st.eye_close_start is None: st.eye_close_start = now
                st.drowsy_frames += 1
                st.eye_was_closed = True
            else:
                if st.eye_was_closed and st.eye_close_start:
                    dur = now-st.eye_close_start
                    if BLINK_MIN_DUR<=dur<=BLINK_MAX_DUR:
                        st.last_blink_time = now
                st.recovery_f += 1
                if st.recovery_f>=RECOVERY_FRAMES:
                    st.drowsy_frames=0; st.recovery_f=0
                st.eye_close_start=None; st.eye_was_closed=False
        else:
            st.drowsy_frames=0; st.gaze_off_frames=0
            st.eye_close_start=None
        st.countdown = max(0.0, BLINK_COUNTDOWN-(now-st.last_blink_time))
        is_drowsy    = st.drowsy_frames > DROWSY_FRAME_THRESH
        is_dist      = st.gaze_off_frames >= GAZE_FRAME_THRESH
        is_blink_due = st.countdown<=0 and st.metrics_reliable
        if   not st.metrics_reliable:      st.status="TOO FAR"
        elif is_drowsy and st.posture_bad: st.status="SLEEP!!"
        elif is_drowsy:                    st.status="DROWSY"
        elif st.posture_bad:               st.status="POOR POSTURE"
        elif is_blink_due:                 st.status="BLINK ALERT"
        elif is_dist:                      st.status="DISTRACTED"
        else:                              st.status="FOCUSED"
        st.record_status(now)

    @staticmethod
    def _eval_posture(st: PersonState, pose_lm, w, h) -> set:
        lm = pose_lm.landmark
        def g(n): i=POSE_IDX[n]; return lm[i],i
        def v(p): return p.visibility>0.40
        bad=set(); st.posture_bad=False; st.posture_note=""
        nos,nos_i=g("nose"); ls,ls_i=g("l_sho"); rs,rs_i=g("r_sho")
        le,le_i=g("l_ear");  re,re_i=g("r_ear")
        lh,lh_i=g("l_hip");  rh,rh_i=g("r_hip")
        if v(lh) and v(rh) and v(nos):
            mhx=(lh.x+rh.x)/2.0
            if abs(nos.x-mhx)>SPINE_ANGLE_THRESH:
                st.posture_bad=True; st.posture_note="spine drift"
                bad.update([nos_i,lh_i,rh_i])
        if v(ls) and v(rs):
            asy=(ls.y+rs.y)/2.0
            for em,ei in [(le,le_i),(re,re_i)]:
                if v(em) and (em.y-asy)>FWD_HEAD_THRESH:
                    st.posture_bad=True
                    st.posture_note=st.posture_note or "fwd head"
                    bad.update([ei,ls_i,rs_i])
            if abs(ls.y-rs.y)>SHOULDER_TILT_THRESH:
                st.posture_bad=True
                st.posture_note=st.posture_note or "sho tilt"
                bad.update([ls_i,rs_i])
        if v(lh) and v(rh) and abs(lh.y-rh.y)>HIP_TILT_THRESH:
            st.posture_bad=True
            st.posture_note=st.posture_note or "hip lean"
            bad.update([lh_i,rh_i])
        return bad

    @staticmethod
    def _dim_face(frame, lms, w, h):
        x1,y1,x2,y2 = face_bbox(lms,w,h)
        pad=20
        x1=max(0,x1-pad); y1=max(0,y1-pad)
        x2=min(w,x2+pad); y2=min(h,y2+pad)
        roi = frame[y1:y2,x1:x2]
        if roi.size>0:
            frame[y1:y2,x1:x2]=(roi*0.35).astype(np.uint8)

    @staticmethod
    def _pose_face_index(pose_lm, states_lms, w, h) -> Optional[int]:
        nose = pose_lm.landmark[0]
        nx, ny = int(nose.x*w), int(nose.y*h)
        for idx, (st, lms) in enumerate(states_lms):
            x1,y1,x2,y2 = face_bbox(lms, w, h)
            pad = 30
            if x1-pad<=nx<=x2+pad and y1-pad<=ny<=y2+pad:
                return idx
        return None

    def run(self):
        mp_face = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose
        face_mesh = mp_face.FaceMesh(
            max_num_faces=MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=FACE_DETECT_CONF,
            min_tracking_confidence=FACE_TRACK_CONF,
        )
        pose = mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=FACE_DETECT_CONF,
            min_tracking_confidence=FACE_TRACK_CONF,
        )
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self._running:
            t0  = time.time()
            ret, frame = cap.read()
            if not ret: break
            frame   = cv2.flip(frame,1)
            h,w,_   = frame.shape
            fa      = float(w*h)
            rgb     = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            now     = time.time()
            face_res = face_mesh.process(rgb)
            pose_res = pose.process(rgb)
            self._registry.begin_frame()
            monitored_pid = self._registry.monitored_pid

            if face_res.multi_face_landmarks:
                all_lms   = face_res.multi_face_landmarks
                areas     = [face_area(lm.landmark,w,h) for lm in all_lms]
                size_pcts = [a/fa for a in areas]
                dzones    = [dist_zone_from_size(p) for p in size_pcts]
                sigs      = [face_sig(lm.landmark) for lm in all_lms]
                states    = self._registry.assign_all(sigs, dzones, now)
                states_lms = []
                for fi, (face_lm, st) in enumerate(zip(all_lms, states)):
                    lms = face_lm.landmark
                    st.face_size_pct = size_pcts[fi]
                    st.dist_zone     = dzones[fi]
                    self._update_person(st, lms, w, h, now)
                    dist = st.gaze_off_frames >= GAZE_FRAME_THRESH
                    st.eye_crop = self._eye_crop(
                        frame, lms, w, h,
                        st.gaze_l, st.gaze_r, dist,
                        st.l_ear, st.r_ear, st.metrics_reliable)
                    if st.metrics_reliable:
                        ic = COL_IRIS_WARN if dist else COL_IRIS_OK
                        for iid in [IRIS_L,IRIS_R]:
                            ix=int(lms[iid].x*w); iy=int(lms[iid].y*h)
                            cv2.circle(frame,(ix,iy),4,ic,-1,cv2.LINE_AA)
                            cv2.circle(frame,(ix,iy),9,ic, 1,cv2.LINE_AA)
                    states_lms.append((st, lms))

                if pose_res.pose_landmarks:
                    pose_fi = self._pose_face_index(
                        pose_res.pose_landmarks, states_lms, w, h)
                    if monitored_pid and pose_fi is None:
                        for fi,(st,lms) in enumerate(states_lms):
                            if st.pid==monitored_pid:
                                pose_fi=fi; break
                    if pose_fi is None:
                        pose_fi = int(np.argmax(areas))
                    posture_st, posture_lms = states_lms[pose_fi]
                    bad_j = self._eval_posture(
                        posture_st, pose_res.pose_landmarks, w, h)
                    self._draw_skeleton(
                        frame, pose_res.pose_landmarks, w, h,
                        bad_j, posture_st.color)
                    self._update_person(posture_st, posture_lms, w, h, now)

                if monitored_pid:
                    for st,lms in states_lms:
                        if st.pid!=monitored_pid:
                            self._dim_face(frame,lms,w,h)

                placed_huds: List[Tuple[int,int,int,int]] = []
                for st,lms in states_lms:
                    self._draw_hud(frame,lms,w,h,st,placed_huds)

            else:
                cv2.putText(frame,"NO PERSON DETECTED",
                            (int(w*0.12),int(h*0.50)),
                            cv2.FONT_HERSHEY_DUPLEX,0.9,
                            COL_NO_PERSON,2,cv2.LINE_AA)

            self.frame_ready.emit(frame, self._registry.all_states())
            time.sleep(max(0.0, 0.033-(time.time()-t0)))

        cap.release()


class MetricCard(QFrame):
    def __init__(self, label, value="--"):
        super().__init__()
        self.setFixedSize(S(108), S(54))
        self.setStyleSheet(
            "QFrame{background:#0C1E30;border:1px solid #1E3A5F;border-radius:6px;}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(S(6),S(3),S(6),S(3))
        lay.setSpacing(S(1))
        self._l = QLabel(label)
        self._l.setFont(QFont("Courier New", F(6)))
        self._l.setStyleSheet("color:#334155;letter-spacing:1px;")
        lay.addWidget(self._l)
        self._v = QLabel(value)
        self._v.setFont(QFont("Courier New", F(11), QFont.Bold))
        self._v.setStyleSheet("color:#E2E8F0;")
        lay.addWidget(self._v)

    def set_value(self, text, color="#E2E8F0"):
        self._v.setText(text)
        self._v.setStyleSheet(
            f"color:{color};font-family:Courier New;"
            f"font-size:{F(11)}pt;font-weight:bold;")


class PersonInspectorDialog(QDialog):
    person_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Person Inspector")
        self.setFixedWidth(S(380))
        self.setMinimumHeight(S(620))
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setStyleSheet("QDialog{background:#050E1C;} QLabel{color:#CBD5E1;}")

        self._selected_pid: Optional[str] = None
        self._all_states:   List[PersonState] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(S(8),S(8),S(8),S(6))
        lay.setSpacing(S(5))

        hdr = QLabel("PERSON  INSPECTOR")
        hdr.setFont(QFont("Courier New", F(9), QFont.Bold))
        hdr.setStyleSheet("color:#38BDF8;letter-spacing:2px;")
        hdr.setAlignment(Qt.AlignCenter)
        lay.addWidget(hdr)

        dd_row = QHBoxLayout()
        dd_lbl = QLabel("MONITOR:")
        dd_lbl.setFont(QFont("Courier New", F(7)))
        dd_lbl.setStyleSheet("color:#64748B;")
        dd_row.addWidget(dd_lbl)
        self._dropdown = QComboBox()
        self._dropdown.setFont(QFont("Courier New", F(8), QFont.Bold))
        self._dropdown.setFixedHeight(S(26))
        self._dropdown.setStyleSheet(f"""
            QComboBox{{background:#0C1E30;border:2px solid #00FF80;
                border-radius:5px;color:#00FF80;padding:1px 6px;font-size:{F(8)}pt;}}
            QComboBox::drop-down{{border:none;}}
            QComboBox QAbstractItemView{{background:#0C1E30;color:#E2E8F0;
                selection-background-color:#1E3A5F;border:1px solid #38BDF8;
                font-size:{F(8)}pt;}}
        """)
        self._dropdown.addItem("-- AUTO (closest person) --", None)
        self._dropdown.currentIndexChanged.connect(self._on_dropdown_changed)
        dd_row.addWidget(self._dropdown, 1)
        lay.addLayout(dd_row)

        self._lock_lbl = QLabel("MODE: AUTO")
        self._lock_lbl.setFont(QFont("Courier New", F(7), QFont.Bold))
        self._lock_lbl.setAlignment(Qt.AlignCenter)
        self._lock_lbl.setFixedHeight(S(22))
        self._lock_lbl.setStyleSheet(
            "background:#0C2010;border:1px solid #50DC50;"
            "border-radius:5px;color:#50DC50;padding:1px;")
        lay.addWidget(self._lock_lbl)

        self._status_box = QLabel("---")
        self._status_box.setFixedHeight(S(44))
        self._status_box.setAlignment(Qt.AlignCenter)
        self._status_box.setFont(QFont("Impact", F(14)))
        self._status_box.setStyleSheet(
            "background:#0C1E30;border-radius:7px;"
            "border:2px solid #1E3A5F;color:#94A3B8;")
        lay.addWidget(self._status_box)

        info_top = QHBoxLayout()
        self._dist_lbl = QLabel("DIST: ---")
        self._dist_lbl.setFont(QFont("Courier New", F(7), QFont.Bold))
        info_top.addWidget(self._dist_lbl)
        self._reid_lbl = QLabel("Re-ID: ---")
        self._reid_lbl.setFont(QFont("Courier New", F(7)))
        self._reid_lbl.setStyleSheet("color:#64748B;")
        self._reid_lbl.setAlignment(Qt.AlignRight)
        info_top.addWidget(self._reid_lbl, 1)
        lay.addLayout(info_top)

        ew,eh = S(362),S(135)
        self._eye_lbl = QLabel()
        self._eye_lbl.setFixedSize(ew, eh)
        self._eye_lbl.setAlignment(Qt.AlignCenter)
        self._eye_lbl.setStyleSheet(
            "background:#010C18;border:1px solid #1E3A5F;border-radius:4px;")
        lay.addWidget(self._eye_lbl)
        self._ew, self._eh = ew, eh

        gc = QGridLayout(); gc.setSpacing(S(4))
        self._c_ear    = MetricCard("EAR")
        self._c_lear   = MetricCard("L-EAR")
        self._c_rear   = MetricCard("R-EAR")
        self._c_blink  = MetricCard("BLINK")
        self._c_drowsy = MetricCard("DROWSY")
        self._c_gaze   = MetricCard("GAZE")
        self._c_post   = MetricCard("POSTURE")
        self._c_frames = MetricCard("FRAMES")
        cards = [self._c_ear,  self._c_lear, self._c_rear,
                 self._c_blink,self._c_drowsy,self._c_gaze,
                 self._c_post, self._c_frames]
        for i,card in enumerate(cards):
            gc.addWidget(card, i//3, i%3)
        lay.addLayout(gc)

        pr = QHBoxLayout()
        self._post_note = QLabel("")
        self._post_note.setFont(QFont("Courier New", F(6)))
        self._post_note.setStyleSheet("color:#FFA500;")
        pr.addWidget(self._post_note)
        self._presence_lbl = QLabel("")
        self._presence_lbl.setFont(QFont("Courier New", F(6)))
        self._presence_lbl.setAlignment(Qt.AlignRight)
        pr.addWidget(self._presence_lbl, 1)
        lay.addLayout(pr)

        hist_hdr = QLabel("ALERT HISTORY (newest first):")
        hist_hdr.setFont(QFont("Courier New", F(6)))
        hist_hdr.setStyleSheet("color:#475569;")
        lay.addWidget(hist_hdr)

        self._history = QListWidget()
        self._history.setFont(QFont("Courier New", F(7)))
        self._history.setStyleSheet(f"""
            QListWidget{{background:#020D1A;border:1px solid #1E3A5F;
                border-radius:4px;color:#94A3B8;}}
            QListWidget::item{{padding:1px {S(5)}px;}}
            QListWidget::item:selected{{background:#1E3A5F;}}
        """)
        lay.addWidget(self._history, 1)

    def _on_dropdown_changed(self, idx):
        if idx<0: return
        pid = self._dropdown.itemData(idx)
        self._selected_pid = pid
        self.person_selected.emit(pid if pid else "")
        self._update_lock_indicator(pid)
        self._refresh_detail()

    def _update_lock_indicator(self, pid):
        if pid:
            st  = next((s for s in self._all_states if s.pid==pid), None)
            col = st.css_color if st else "#00FF80"
            self._lock_lbl.setText(f"LOCKED: {pid}")
            self._lock_lbl.setStyleSheet(
                f"background:#1A0A20;border:2px solid {col};"
                f"border-radius:5px;color:{col};padding:1px;"
                f"font-family:Courier New;font-size:{F(7)}pt;font-weight:bold;")
        else:
            self._lock_lbl.setText("MODE: AUTO  (closest person)")
            self._lock_lbl.setStyleSheet(
                "background:#0C2010;border:1px solid #50DC50;"
                f"border-radius:5px;color:#50DC50;padding:1px;"
                f"font-family:Courier New;font-size:{F(7)}pt;font-weight:bold;")

    def update_states(self, all_states):
        self._all_states = all_states
        self._sync_dropdown(all_states)
        self._refresh_detail()

    def _sync_dropdown(self, states):
        current_pid = self._selected_pid
        existing = [self._dropdown.itemData(i)
                    for i in range(self._dropdown.count())]
        self._dropdown.blockSignals(True)
        for st in states:
            if st.pid not in existing:
                self._dropdown.addItem("", st.pid)
        for i in range(self._dropdown.count()):
            pid = self._dropdown.itemData(i)
            if pid is None:
                self._dropdown.setItemText(i,"-- AUTO --"); continue
            st = next((s for s in states if s.pid==pid), None)
            if st:
                tag = "LIVE" if st.is_live else "AWAY"
                dz  = st.dist_zone if st.is_live else "--"
                self._dropdown.setItemText(
                    i, f"{st.pid} [{tag}] {dz}  {st.status}")
        if current_pid:
            for i in range(self._dropdown.count()):
                if self._dropdown.itemData(i)==current_pid:
                    self._dropdown.setCurrentIndex(i); break
        else:
            self._dropdown.setCurrentIndex(0)
        self._dropdown.blockSignals(False)

    def _refresh_detail(self):
        if self._selected_pid:
            st = next((s for s in self._all_states
                       if s.pid==self._selected_pid), None)
        else:
            st = next((s for s in self._all_states
                       if s.is_live and s.is_monitored), None)
            if st is None:
                live = [s for s in self._all_states if s.is_live]
                st = live[0] if live else None
        if st is None:
            self._status_box.setText("NO DATA"); return

        sc   = st.status_css
        disp = st.status if st.is_live else f"{st.status} [AWAY]"
        self._status_box.setText(disp)
        self._status_box.setStyleSheet(
            f"background:#0C1E30;border-radius:7px;border:2px solid {sc};"
            f"color:{sc};font-family:Impact;font-size:{F(14)}pt;")

        dz_css={"NEAR":"#50DC50","MED":"#38BDF8","FAR":"#FFA500","TOO FAR":"#444466"}
        if st.is_live:
            dc = dz_css.get(st.dist_zone,"#E2E8F0")
            self._dist_lbl.setText(
                f"DIST: {st.dist_zone}  ({st.face_size_pct*100:.1f}%)")
            self._dist_lbl.setStyleSheet(
                f"color:{dc};font-family:Courier New;font-size:{F(7)}pt;font-weight:bold;")
            if st.reid_score >= 0.999:
                self._reid_lbl.setText("NEW person")
                self._reid_lbl.setStyleSheet(
                    f"color:#64748B;font-family:Courier New;font-size:{F(6)}pt;")
            else:
                pct=int(st.reid_score*100)
                rc="#50DC50" if pct>=80 else "#FFA500"
                self._reid_lbl.setText(f"Re-ID: {pct}% match — same ID")
                self._reid_lbl.setStyleSheet(
                    f"color:{rc};font-family:Courier New;font-size:{F(6)}pt;")
        else:
            self._dist_lbl.setText("DIST: AWAY")
            self._dist_lbl.setStyleSheet(
                f"color:#475569;font-family:Courier New;font-size:{F(7)}pt;")
            self._reid_lbl.setText("ID locked — will reconnect on return")
            self._reid_lbl.setStyleSheet(
                f"color:#38BDF8;font-family:Courier New;font-size:{F(6)}pt;")

        if st.eye_crop is not None and st.eye_crop.size>100:
            self._eye_lbl.setPixmap(cv_to_pixmap(st.eye_crop,self._ew,self._eh))
        else:
            self._eye_lbl.setText("No eye data")

        reliable=st.metrics_reliable; no_data="#444466"
        ec="#FF4444" if st.ear  <EAR_THRESH else "#50DC50"
        lc="#FF4444" if st.l_ear<EAR_THRESH else "#50DC50"
        rc="#FF4444" if st.r_ear<EAR_THRESH else "#50DC50"
        self._c_ear.set_value(f"{st.ear:.3f}"   if reliable else "N/A", ec if reliable else no_data)
        self._c_lear.set_value(f"{st.l_ear:.3f}" if reliable else "N/A", lc if reliable else no_data)
        self._c_rear.set_value(f"{st.r_ear:.3f}" if reliable else "N/A", rc if reliable else no_data)
        ct=st.countdown
        bc="#FF00FF" if ct<=0 else ("#FFA500" if ct<4 else "#E2E8F0")
        self._c_blink.set_value(f"{ct:.1f}s" if reliable else "N/A", bc if reliable else no_data)
        df=st.drowsy_frames
        dc2="#FF2020" if df>DROWSY_FRAME_THRESH else ("#FFA500" if df>6 else "#E2E8F0")
        self._c_drowsy.set_value(str(df) if reliable else "N/A", dc2 if reliable else no_data)
        avg_g=(st.gaze_l+st.gaze_r)/2.0
        gc2="#00DCDC" if st.gaze_off_frames>=GAZE_FRAME_THRESH else "#E2E8F0"
        self._c_gaze.set_value(f"{avg_g:.2f}" if reliable else "N/A", gc2 if reliable else no_data)
        self._c_post.set_value("BAD" if st.posture_bad else "OK",
                               "#FFA500" if st.posture_bad else "#50DC50")
        self._c_frames.set_value(str(st.total_frames))
        self._post_note.setText(st.posture_note.upper() if st.posture_note else "")

        if st.is_live:
            dur=int(time.time()-st.first_seen)
            self._presence_lbl.setText(f"ON SCREEN {dur}s")
            self._presence_lbl.setStyleSheet(
                f"color:#50DC50;font-family:Courier New;font-size:{F(6)}pt;")
        else:
            ago=int(time.time()-st.last_seen)
            self._presence_lbl.setText(f"AWAY {ago}s ago")
            self._presence_lbl.setStyleSheet(
                f"color:#64748B;font-family:Courier New;font-size:{F(6)}pt;")

        self._history.clear()
        for ev in reversed(st.history):
            ts  = time.strftime("%H:%M:%S", time.localtime(ev.timestamp))
            css = STATUS_CSS.get(ev.status,"#E2E8F0")
            item= QListWidgetItem(f" {ts}  {ev.status}")
            item.setForeground(QColor(css))
            self._history.addItem(item)

    def select_person(self, pid):
        self._selected_pid = pid
        self._sync_dropdown(self._all_states)
        self._update_lock_indicator(pid)
        self._refresh_detail()

    def closeEvent(self, event):
        event.ignore(); self.hide()


class BioEdgeDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            f"BioEdge AI  (scale={UI_SCALE}  MAX_FACES={MAX_FACES})")
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(screen.width(), S(1280)),
                    min(screen.height(), S(760)))
        self.setStyleSheet("background-color:#040D1A;color:#E2E8F0;")

        self._registry = PersonRegistry()
        self._build_ui()

        self._inspector = PersonInspectorDialog(self)
        self._inspector.person_selected.connect(self._on_person_selected)
        self._reposition_inspector()
        self._inspector.show()

        self._thread = VideoThread(self._registry)
        self._thread.frame_ready.connect(self._update_ui)
        self._thread.start()

    def _reposition_inspector(self):
        mw = self.geometry()
        x  = max(0, mw.x() - self._inspector.width() - 6)
        self._inspector.move(x, mw.y())

    def moveEvent(self, e):
        super().moveEvent(e)
        if hasattr(self,"_inspector"): self._reposition_inspector()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self,"_inspector"): self._reposition_inspector()

    def _on_person_selected(self, pid: str):
        self._registry.set_monitor(pid if pid else None)

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(S(10),S(10),S(10),S(10))
        root.setSpacing(S(8))

        sb = QVBoxLayout(); sb.setSpacing(S(5))

        title = QLabel("BIOEDGE AI")
        title.setFont(QFont("Impact", F(22)))
        title.setStyleSheet("color:#38BDF8;letter-spacing:3px;")
        sb.addWidget(title)

        sub = QLabel(f"MULTI-PERSON | MAX {MAX_FACES} | SCALE {UI_SCALE}")
        sub.setFont(QFont("Courier New", F(6)))
        sub.setStyleSheet("color:#334155;letter-spacing:2px;margin-top:-4px;")
        sb.addWidget(sub)

        self._count_lbl = QLabel("ACTIVE: 0 / TOTAL: 0")
        self._count_lbl.setFont(QFont("Courier New", F(7), QFont.Bold))
        self._count_lbl.setStyleSheet("color:#38BDF8;")
        sb.addWidget(self._count_lbl)

        self._mode_lbl = QLabel("MODE: AUTO")
        self._mode_lbl.setFont(QFont("Courier New", F(7), QFont.Bold))
        self._mode_lbl.setAlignment(Qt.AlignCenter)
        self._mode_lbl.setFixedHeight(S(24))
        self._mode_lbl.setStyleSheet(
            "background:#0C2010;border:1px solid #50DC50;"
            "border-radius:5px;color:#50DC50;padding:1px;")
        sb.addWidget(self._mode_lbl)

        self._monitor_hdr = QLabel("MONITORING: AUTO")
        self._monitor_hdr.setFont(QFont("Courier New", F(6)))
        self._monitor_hdr.setStyleSheet("color:#475569;")
        sb.addWidget(self._monitor_hdr)

        self._status_box = QLabel("---")
        self._status_box.setFixedSize(S(240), S(60))
        self._status_box.setAlignment(Qt.AlignCenter)
        self._status_box.setFont(QFont("Impact", F(15)))
        self._status_box.setStyleSheet(
            "background:#0C1E30;border-radius:9px;"
            "border:2px solid #1E3A5F;color:#94A3B8;")
        sb.addWidget(self._status_box)

        self._dist_sidebar = QLabel("DIST: ---")
        self._dist_sidebar.setFont(QFont("Courier New", F(7), QFont.Bold))
        self._dist_sidebar.setStyleSheet("color:#64748B;")
        sb.addWidget(self._dist_sidebar)

        self._pid_lbl = QLabel("ID: ---")
        self._pid_lbl.setFont(QFont("Courier New", F(8), QFont.Bold))
        self._pid_lbl.setStyleSheet("color:#38BDF8;")
        sb.addWidget(self._pid_lbl)

        self._reid_sidebar = QLabel("")
        self._reid_sidebar.setFont(QFont("Courier New", F(6)))
        self._reid_sidebar.setStyleSheet("color:#334155;")
        self._reid_sidebar.setWordWrap(True)
        sb.addWidget(self._reid_sidebar)

        gc = QGridLayout(); gc.setSpacing(S(4))
        self._c_ear    = MetricCard("EAR")
        self._c_blink  = MetricCard("BLINK")
        self._c_drowsy = MetricCard("DROWSY")
        self._c_post   = MetricCard("POSTURE")
        gc.addWidget(self._c_ear,   0,0); gc.addWidget(self._c_blink, 0,1)
        gc.addWidget(self._c_drowsy,1,0); gc.addWidget(self._c_post,  1,1)
        sb.addLayout(gc)

        self._post_note = QLabel("")
        self._post_note.setFont(QFont("Courier New", F(6)))
        self._post_note.setStyleSheet("color:#FFA500;")
        sb.addWidget(self._post_note)

        rl = QLabel("ALL PERSONS — click to monitor")
        rl.setFont(QFont("Courier New", F(6)))
        rl.setStyleSheet("color:#475569;letter-spacing:1px;")
        sb.addWidget(rl)

        self._roster = QListWidget()
        self._roster.setFixedHeight(S(110))
        self._roster.setFont(QFont("Courier New", F(7)))
        self._roster.setStyleSheet(f"""
            QListWidget{{background:#020D1A;border:1px solid #1E3A5F;border-radius:4px;}}
            QListWidget::item{{padding:{S(2)}px {S(4)}px;}}
            QListWidget::item:selected{{background:#1E3A5F;}}
            QListWidget::item:hover{{background:#0F2237;}}
        """)
        self._roster.itemClicked.connect(self._roster_clicked)
        sb.addWidget(self._roster)

        btn_row = QHBoxLayout(); btn_row.setSpacing(S(4))
        auto_btn = QPushButton("⟳ AUTO")
        auto_btn.setFont(QFont("Courier New", F(7)))
        auto_btn.setFixedHeight(S(26))
        auto_btn.setStyleSheet("""
            QPushButton{background:#0C2010;border:1px solid #50DC50;
                border-radius:5px;color:#50DC50;padding:1px;}
            QPushButton:hover{background:#1A4020;}""")
        auto_btn.clicked.connect(self._release_lock)
        btn_row.addWidget(auto_btn)
        insp_btn = QPushButton("Inspector")
        insp_btn.setFont(QFont("Courier New", F(7)))
        insp_btn.setFixedHeight(S(26))
        insp_btn.setStyleSheet("""
            QPushButton{background:#0C1E30;border:1px solid #38BDF8;
                border-radius:5px;color:#38BDF8;padding:1px;}
            QPushButton:hover{background:#1E3A5F;}""")
        insp_btn.clicked.connect(self._toggle_inspector)
        btn_row.addWidget(insp_btn)
        sb.addLayout(btn_row)

        sep = QLabel("─"*20)
        sep.setStyleSheet("color:#1E3A5F;font-size:6pt;")
        sb.addWidget(sep)

        for lbl,col in [
            ("SLEEP!!","#FF2020"),("DROWSY","#FF8C00"),
            ("POOR POSTURE","#FFA500"),("BLINK ALERT","#FF00FF"),
            ("DISTRACTED","#00DCDC"),("FOCUSED","#50DC50"),
            ("TOO FAR","#444466"),("NO PERSON","#606060")
        ]:
            row=QHBoxLayout(); row.setSpacing(S(3))
            dot=QLabel("●"); dot.setFixedWidth(S(12))
            dot.setStyleSheet(f"color:{col};font-size:{F(8)}px;")
            txt=QLabel(lbl); txt.setFont(QFont("Courier New",F(6)))
            txt.setStyleSheet("color:#94A3B8;")
            row.addWidget(dot); row.addWidget(txt); row.addStretch()
            sb.addLayout(row)

        sb.addStretch()
        root.addLayout(sb, 0)

        self._vid = QLabel()
        self._vid.setAlignment(Qt.AlignCenter)
        self._vid.setStyleSheet(
            "background:#06101E;border-radius:10px;border:2px solid #1E3A5F;")
        root.addWidget(self._vid, 1)

    def _roster_clicked(self, item):
        pid = item.data(Qt.UserRole)
        if pid:
            self._registry.set_monitor(pid)
            self._inspector.select_person(pid)

    def _release_lock(self):
        self._registry.set_monitor(None)
        self._inspector.select_person(None)

    def _toggle_inspector(self):
        if self._inspector.isVisible():
            self._inspector.hide()
        else:
            self._reposition_inspector(); self._inspector.show()

    def _update_ui(self, cv_img, all_states: list):
        if cv_img is not None:
            self._vid.setPixmap(cv_to_pixmap(cv_img, S(960), S(720)))

        self._inspector.update_states(all_states)

        live  = sum(1 for s in all_states if s.is_live)
        total = len(all_states)
        self._count_lbl.setText(f"ACTIVE: {live}  /  TOTAL: {total}")

        mon_pid = self._registry.monitored_pid
        if mon_pid:
            self._mode_lbl.setText(f"LOCKED: {mon_pid}")
            self._mode_lbl.setStyleSheet(
                "background:#1A0A20;border:2px solid #00FF80;"
                f"border-radius:5px;color:#00FF80;padding:1px;"
                f"font-family:Courier New;font-size:{F(7)}pt;font-weight:bold;")
            self._monitor_hdr.setText(f"MONITORING: {mon_pid}")
        else:
            self._mode_lbl.setText("MODE: AUTO  (closest)")
            self._mode_lbl.setStyleSheet(
                "background:#0C2010;border:1px solid #50DC50;"
                f"border-radius:5px;color:#50DC50;padding:1px;"
                f"font-family:Courier New;font-size:{F(7)}pt;font-weight:bold;")
            self._monitor_hdr.setText("MONITORING: AUTO")

        existing = {self._roster.item(i).data(Qt.UserRole)
                    for i in range(self._roster.count())}
        for st in all_states:
            dot  = "●" if st.is_live else "○"
            mon  = " ★" if st.is_monitored else ""
            dist = f" {st.dist_zone}" if st.is_live else ""
            txt  = f"{dot} {st.pid}{mon}{dist} [{st.status}]"
            if st.pid not in existing:
                item = QListWidgetItem(txt)
                item.setData(Qt.UserRole, st.pid)
                item.setForeground(QColor(st.css_color))
                self._roster.addItem(item)
                existing.add(st.pid)
            else:
                for i in range(self._roster.count()):
                    it = self._roster.item(i)
                    if it.data(Qt.UserRole)==st.pid:
                        it.setText(txt)
                        fg = "#00FF80" if st.is_monitored else (
                             st.status_css if st.is_live else "#475569")
                        it.setForeground(QColor(fg))
                        break

        target = None
        if mon_pid:
            target = next((s for s in all_states if s.pid==mon_pid), None)
        if target is None:
            target = next((s for s in all_states if s.is_live), None)

        if target:
            sc=target.status_css; pc=target.css_color
            reliable=target.metrics_reliable; no_data="#444466"
            disp = target.status if target.is_live else f"{target.status} [AWAY]"
            self._status_box.setText(disp)
            self._status_box.setStyleSheet(
                f"background:#0C1E30;border-radius:9px;border:2px solid {sc};"
                f"color:{sc};font-family:Impact;font-size:{F(15)}pt;")
            dz_css={"NEAR":"#50DC50","MED":"#38BDF8","FAR":"#FFA500","TOO FAR":"#444466"}
            if target.is_live:
                dc=dz_css.get(target.dist_zone,"#E2E8F0")
                self._dist_sidebar.setText(
                    f"DIST: {target.dist_zone}  ({target.face_size_pct*100:.1f}%)")
                self._dist_sidebar.setStyleSheet(
                    f"color:{dc};font-family:Courier New;font-size:{F(7)}pt;font-weight:bold;")
                if target.reid_score>=0.999:
                    self._reid_sidebar.setText("New person")
                else:
                    pct=int(target.reid_score*100)
                    rc="#50DC50" if pct>=80 else "#FFA500"
                    self._reid_sidebar.setText(f"Re-ID: {pct}% — same ID kept")
                    self._reid_sidebar.setStyleSheet(
                        f"color:{rc};font-family:Courier New;font-size:{F(6)}pt;")
            else:
                self._dist_sidebar.setText("AWAY — last known values")
                self._dist_sidebar.setStyleSheet(
                    f"color:#475569;font-family:Courier New;font-size:{F(7)}pt;")
                self._reid_sidebar.setText("ID locked — auto-reconnect on return")
                self._reid_sidebar.setStyleSheet(
                    f"color:#38BDF8;font-family:Courier New;font-size:{F(6)}pt;")

            tag="[MON]" if mon_pid else "[AUTO]"
            self._pid_lbl.setText(f"ID: {target.pid}  {tag}")
            self._pid_lbl.setStyleSheet(
                f"color:{pc};font-family:Courier New;"
                f"font-size:{F(8)}pt;font-weight:bold;")
            ec="#FF4444" if target.ear<EAR_THRESH else "#50DC50"
            self._c_ear.set_value(
                f"{target.ear:.3f}" if reliable else "FAR", ec if reliable else no_data)
            ct=target.countdown
            bc="#FF00FF" if ct<=0 else ("#FFA500" if ct<4 else "#E2E8F0")
            self._c_blink.set_value(f"{ct:.1f}s" if reliable else "FAR",
                                    bc if reliable else no_data)
            df=target.drowsy_frames
            dc2="#FF2020" if df>DROWSY_FRAME_THRESH else ("#FFA500" if df>6 else "#E2E8F0")
            self._c_drowsy.set_value(str(df) if reliable else "FAR",
                                     dc2 if reliable else no_data)
            self._c_post.set_value("BAD" if target.posture_bad else "OK",
                                   "#FFA500" if target.posture_bad else "#50DC50")
            self._post_note.setText(
                target.posture_note.upper() if target.posture_note else "")
        else:
            self._status_box.setText("NO PERSON")
            self._status_box.setStyleSheet(
                "background:#0C1E30;border-radius:9px;"
                "border:2px solid #606060;color:#606060;"
                f"font-family:Impact;font-size:{F(15)}pt;")
            self._dist_sidebar.setText("DIST: ---")
            self._reid_sidebar.setText("")

    def closeEvent(self, event):
        self._thread.stop(); self._thread.wait(2000)
        self._inspector.deleteLater(); event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = BioEdgeDashboard()
    win.show()
    sys.exit(app.exec_())
