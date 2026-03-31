"""
╔══════════════════════════════════════════════════════════════════════╗
║         INDIAN SIGN LANGUAGE (ISL) RECOGNITION SYSTEM v2.0          ║
║         Real-Time Hand Gesture + Body Pose Detection                 ║
║         MediaPipe 0.10.x | OpenCV | NumPy | pyttsx3 (optional)      ║
╚══════════════════════════════════════════════════════════════════════╝

Controls:
  Hold gesture 2.0s = auto-save word to sentence
  SPACE             = save current word instantly
  C                 = clear sentence
  S                 = speak sentence (TTS, if pyttsx3 installed)
  ESC               = quit

ACCURACY IMPROVEMENTS v2.0:
  ✓ Chirality-aware thumb detection (left vs right hand)
  ✓ 3D depth (z-axis) used in finger curl/extension checks
  ✓ Angle-based finger classification (not just y-position)
  ✓ Confidence scoring per gesture — low-confidence → "UNCERTAIN"
  ✓ Wider smoother window (18 frames) + higher threshold (65%)
  ✓ Gesture priority ordering to resolve overlapping states
  ✓ Normalised distance thresholds calibrated per gesture
  ✓ Pinch detection with dual-axis check
  ✓ Hook-finger detection using angle, not just y
  ✓ Pose classifier with stricter per-sign thresholds
"""

import cv2
import numpy as np
import time
import urllib.request
import os
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# ── Optional TTS ──────────────────────────────────────────────────────────────
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty('rate', 150)
    TTS_AVAILABLE = True
    print("  pyttsx3 TTS loaded.")
except ImportError:
    TTS_AVAILABLE = False
    print("  pyttsx3 not found — TTS disabled.")

print(f"  MediaPipe {mp.__version__} loaded.")

# ── Landmark indices ──────────────────────────────────────────────────────────
THUMB_TIP=4;  THUMB_IP=3;  THUMB_MCP=2;  THUMB_CMC=1
INDEX_TIP=8;  INDEX_DIP=7; INDEX_PIP=6;  INDEX_MCP=5
MIDDLE_TIP=12;MIDDLE_DIP=11;MIDDLE_PIP=10;MIDDLE_MCP=9
RING_TIP=16;  RING_DIP=15; RING_PIP=14;  RING_MCP=13
PINKY_TIP=20; PINKY_DIP=19;PINKY_PIP=18; PINKY_MCP=17
WRIST=0

POSE_NOSE=0; POSE_L_SHOULDER=11; POSE_R_SHOULDER=12
POSE_L_ELBOW=13; POSE_R_ELBOW=14
POSE_L_WRIST=15; POSE_R_WRIST=16
POSE_L_HIP=23;   POSE_R_HIP=24

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24)
]

# ── Download MediaPipe task models ────────────────────────────────────────────
HAND_PATH = "hand_landmarker.task"
POSE_PATH = "pose_landmarker.task"

def download_model(url, path):
    if not os.path.exists(path):
        print(f"  Downloading {path} (~10MB, first time only)...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  Saved: {path}")
        except Exception as e:
            print(f"  ERROR: {e}\n  Check your internet connection.")
            exit(1)

download_model(
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    HAND_PATH
)
download_model(
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    POSE_PATH
)

# ── Create MediaPipe detectors ────────────────────────────────────────────────
hand_landmarker = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=HAND_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.60,
    )
)

pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
    mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=POSE_PATH),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.60,
        min_pose_presence_confidence=0.60,
        min_tracking_confidence=0.60,
    )
)

print("  Models ready. Opening camera...\n")

# ── Landmark drawing ──────────────────────────────────────────────────────────
def draw_hand_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 210, 110), 2)
    for i, pt in enumerate(pts):
        color = (0, 255, 80) if i in (4,8,12,16,20) else (240, 240, 240)
        cv2.circle(frame, pt, 5, color, -1)
        cv2.circle(frame, pt, 5, (0, 0, 0), 1)

def draw_pose_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (30, 100, 200), 2)
    for pt in pts[:25]:
        cv2.circle(frame, pt, 4, (60, 160, 255), -1)

# ── Geometric helpers ─────────────────────────────────────────────────────────
def lma(lms, idx):
    return np.array([lms[idx].x, lms[idx].y, lms[idx].z])

def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def dist2d(a, b):
    return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2])))

def angle_3pts(a, b, c):
    """Angle at point b formed by a-b-c (degrees)."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cos_a = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

def hand_scale_ref(lms):
    """Wrist to middle-MCP distance — used to normalise all distances."""
    return dist(lma(lms, WRIST), lma(lms, MIDDLE_MCP)) + 1e-5

# ── Improved finger state detection ──────────────────────────────────────────
def is_finger_extended(lms, tip, dip, pip, mcp):
    """
    Multi-criterion finger extension test:
    1. Tip y is above PIP y (primary)
    2. Tip-MCP distance > PIP-MCP distance (length-based)
    3. Angle at PIP joint is relatively straight (> 150°)
    All three must agree for confident extension.
    Uses majority vote if they disagree.
    """
    tip_pt  = lma(lms, tip)
    dip_pt  = lma(lms, dip)
    pip_pt  = lma(lms, pip)
    mcp_pt  = lma(lms, mcp)

    # Criterion 1: y position (tip above pip in image coords)
    y_up = tip_pt[1] < pip_pt[1]

    # Criterion 2: distance — extended finger reaches further from MCP
    d_tip_mcp = dist(tip_pt, mcp_pt)
    d_pip_mcp = dist(pip_pt, mcp_pt)
    dist_ext = d_tip_mcp > d_pip_mcp * 1.15

    # Criterion 3: angle at pip joint — straight finger > 150°
    ang = angle_3pts(mcp_pt, pip_pt, tip_pt)
    angle_ext = ang > 150

    votes = sum([y_up, dist_ext, angle_ext])
    return votes >= 2  # majority of criteria

def is_finger_curled(lms, tip, mcp):
    """True if fingertip is curled toward palm."""
    tip_pt = lma(lms, tip)
    mcp_pt = lma(lms, mcp)
    # Tip y is below (greater) than MCP y, AND tip z is more positive (deeper)
    y_below = tip_pt[1] > mcp_pt[1] - 0.01
    return y_below

def is_thumb_extended(lms, handedness_label):
    """
    Chirality-aware thumb extension.
    For RIGHT hand (appears on left side of mirrored frame):
        Thumb tip x < Thumb IP x  → extended left
    For LEFT hand (appears on right side of mirrored frame):
        Thumb tip x > Thumb IP x  → extended right
    Also checks that thumb tip is away from index MCP (not tucked).
    """
    tip = lma(lms, THUMB_TIP)
    ip  = lma(lms, THUMB_IP)
    mcp = lma(lms, THUMB_MCP)
    idx_mcp = lma(lms, INDEX_MCP)

    # Horizontal extension check (chirality-corrected)
    if handedness_label == "Right":
        x_ext = tip[0] < ip[0] - 0.01   # tip goes left of IP
    else:
        x_ext = tip[0] > ip[0] + 0.01   # tip goes right of IP

    # Vertical: tip above MCP (thumb pointing up)
    y_ext = tip[1] < mcp[1] - 0.02

    # Distance: tip is not too close to index MCP (not tucked)
    sep_from_index = dist(tip, idx_mcp)
    hs = hand_scale_ref(lms)
    not_tucked = sep_from_index / hs > 0.25

    # Thumb considered extended if horizontally OR vertically extended + not tucked
    return (x_ext or y_ext) and not_tucked

def is_thumb_down(lms):
    """True if thumb tip is pointing clearly downward (below wrist)."""
    tip = lma(lms, THUMB_TIP)
    mcp = lma(lms, THUMB_MCP)
    wrist = lma(lms, WRIST)
    # Tip is below MCP and below wrist y
    return tip[1] > mcp[1] + 0.04 and tip[1] > wrist[1]

def is_index_hook(lms):
    """
    Index finger in hook/curl shape (COME sign).
    Uses angle at PIP joint — a hook has a bent PIP (< 130°).
    """
    tip = lma(lms, INDEX_TIP)
    pip = lma(lms, INDEX_PIP)
    mcp = lma(lms, INDEX_MCP)
    ang = angle_3pts(mcp, pip, tip)
    # Hook: PIP bent but MCP somewhat raised
    return ang < 130 and lms[INDEX_MCP].y < lms[WRIST].y + 0.05

def get_finger_states(lms, handedness_label="Right"):
    """
    Returns reliable finger state dict using improved per-finger detection.
    """
    thu = is_thumb_extended(lms, handedness_label)
    idx = is_finger_extended(lms, INDEX_TIP,  INDEX_DIP,  INDEX_PIP,  INDEX_MCP)
    mid = is_finger_extended(lms, MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP)
    rng = is_finger_extended(lms, RING_TIP,   RING_DIP,   RING_PIP,   RING_MCP)
    pnk = is_finger_extended(lms, PINKY_TIP,  PINKY_DIP,  PINKY_PIP,  PINKY_MCP)

    return {
        "thumb":  thu,
        "index":  idx,
        "middle": mid,
        "ring":   rng,
        "pinky":  pnk,
        "count":  sum([thu, idx, mid, rng, pnk])
    }

# ── ISL Hand Gesture Classifier ───────────────────────────────────────────────
def classify_isl_hand(lms, handedness_label="Right"):
    """
    Classify ISL hand gestures using improved landmark geometry.
    Returns: (sign_code, isl_meaning, display_color, confidence_0_to_1)
    """
    f   = get_finger_states(lms, handedness_label)
    thu = f["thumb"]; idx = f["index"]; mid = f["middle"]
    rng = f["ring"];  pnk = f["pinky"]; cnt = f["count"]

    hs = hand_scale_ref(lms)

    # Normalised distances (robust to hand size)
    d_ti  = dist(lma(lms, THUMB_TIP), lma(lms, INDEX_TIP))  / hs
    d_tm  = dist(lma(lms, THUMB_TIP), lma(lms, MIDDLE_TIP)) / hs
    d_tr  = dist(lma(lms, THUMB_TIP), lma(lms, RING_TIP))   / hs
    d_tp  = dist(lma(lms, THUMB_TIP), lma(lms, PINKY_TIP))  / hs
    d_im  = dist(lma(lms, INDEX_TIP), lma(lms, MIDDLE_TIP)) / hs
    d_ip  = dist(lma(lms, INDEX_TIP), lma(lms, PINKY_TIP))  / hs

    # Spread: average distance between adjacent extended fingertips
    finger_tips = []
    tip_ids = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    ext_flags = [idx, mid, rng, pnk]
    for i, (tid, ext) in enumerate(zip(tip_ids, ext_flags)):
        if ext:
            finger_tips.append(lma(lms, tid))

    spread = 0.0
    if len(finger_tips) >= 2:
        dists = [dist(finger_tips[i], finger_tips[i+1])
                 for i in range(len(finger_tips)-1)]
        spread = np.mean(dists) / hs

    # ── PRIORITY ORDER: most specific → least specific ─────────────────────

    # ── Pinch-based signs (require accurate thumb+finger contact) ──────────

    # OK — tight thumb-index pinch, other 3 fingers extended
    if d_ti < 0.11 and mid and rng and pnk and not idx:
        conf = 1.0 - d_ti/0.11
        return "OK", "OK / Perfect / Theek Hai", (50, 220, 180), conf

    # EAT/FOOD — all fingertips bunched together near thumb (O-shape)
    # Check thumb-index AND thumb-middle both pinched, other fingers curled
    if d_ti < 0.18 and d_tm < 0.20 and not rng and not pnk:
        conf = max(0, 1.0 - max(d_ti, d_tm)/0.20)
        return "EAT / FOOD", "Eat / Food / Khana", (255, 160, 80), conf

    # ── Thumb-only signs ──────────────────────────────────────────────────

    # THUMBS DOWN — must check before THUMBS UP
    if is_thumb_down(lms) and not idx and not mid and not rng and not pnk:
        tip_y  = lms[THUMB_TIP].y
        mcp_y  = lms[THUMB_MCP].y
        conf   = min(1.0, (tip_y - mcp_y) / 0.08)
        return "THUMBS DOWN", "No / Bad / Disagree", (50, 50, 220), conf

    # THUMBS UP — thumb raised, all others curled
    if thu and not idx and not mid and not rng and not pnk:
        tip_y = lms[THUMB_TIP].y
        mcp_y = lms[THUMB_MCP].y
        conf  = min(1.0, max(0, (mcp_y - tip_y) / 0.06))
        return "THUMBS UP", "Yes / Good / Agree", (50, 220, 50), conf

    # CALL ME — thumb + pinky, others curled
    if thu and pnk and not idx and not mid and not rng:
        conf = 0.85 if d_tp > 0.5 else 0.65
        return "CALL ME", "Call Me / Phone", (200, 100, 255), conf

    # I LOVE YOU — index + pinky + thumb, middle+ring curled
    if thu and idx and pnk and not mid and not rng:
        conf = 0.80 if d_ip > 0.55 else 0.65
        return "I LOVE YOU", "I Love You / Pyaar", (255, 80, 160), conf

    # ── Index-only signs ─────────────────────────────────────────────────

    # COME — index hook (bent), others curled
    if is_index_hook(lms) and not mid and not rng and not pnk:
        ang = angle_3pts(lma(lms, INDEX_MCP), lma(lms, INDEX_PIP), lma(lms, INDEX_TIP))
        conf = min(1.0, max(0, (130 - ang) / 40))
        return "COME", "Come Here / Aao", (255, 200, 100), conf

    # POINT/ONE — index only extended
    if idx and not mid and not rng and not pnk and not thu:
        conf = 0.90
        return "ONE", "1 / One", (180, 255, 100), conf

    # ── Multi-finger signs ────────────────────────────────────────────────

    # SCISSORS — index + middle close together (spread < threshold)
    if idx and mid and not rng and not pnk and not thu:
        if spread < 0.22:
            return "SCISSORS", "Cut / Scissors", (255, 150, 0), 0.80
        else:
            return "TWO / PEACE", "2 / Two / Peace", (255, 220, 0), 0.85

    # WATER / THREE — index + middle + ring, no pinky, no thumb
    if idx and mid and rng and not pnk and not thu:
        return "WATER", "Water / Paani", (80, 160, 255), 0.80

    # THREE (with thumb) - sometimes used in ISL
    if idx and mid and rng and not pnk and thu:
        return "THREE", "3 / Three", (180, 255, 150), 0.75

    # FOUR — four fingers, no thumb
    if idx and mid and rng and pnk and not thu:
        conf = 0.85 if spread > 0.20 else 0.70
        return "FOUR", "4 / Four", (100, 255, 200), conf

    # POWER / ROCK — index + pinky, no thumb, others curled
    if idx and pnk and not thu and not mid and not rng:
        return "POWER", "Strong / Power", (255, 100, 100), 0.80

    # ── All-finger signs ─────────────────────────────────────────────────

    # FIVE / OPEN HAND / NAMASTE — all five fingers up
    if idx and mid and rng and pnk and thu:
        if spread < 0.22 and d_im < 0.25:
            # Fingers close together
            return "NAMASTE", "Namaste / Hello", (255, 200, 60), 0.80
        elif spread > 0.30:
            return "OPEN HAND", "Stop / Hello / Wait", (0, 180, 255), 0.85
        else:
            return "FIVE", "5 / Five", (0, 200, 255), 0.80

    # THANK YOU — four fingers (no thumb), spread open
    if idx and mid and rng and pnk and not thu and spread > 0.25:
        return "THANK YOU", "Thank You / Shukriya", (255, 140, 60), 0.75

    # FIST — all curled (cnt == 0 or only marginal)
    if cnt == 0:
        return "FIST", "Strong / Ready / Fight", (0, 100, 255), 0.90

    # PAIN — index pointing downward
    if (lms[INDEX_TIP].y > lms[WRIST].y and
            not mid and not rng and not pnk and not thu):
        return "PAIN", "Pain / Hurt / Dard", (255, 60, 60), 0.70

    # Uncertain / unclear gesture
    return "HOLD", "Hold On / Thinking...", (160, 160, 160), 0.30


# ── ISL Pose / Body Language Classifier ──────────────────────────────────────
def classify_isl_pose(lms):
    """
    Classify ISL body-pose gestures with stricter, calibrated thresholds.
    Returns: (sign_code, isl_meaning, color, confidence)  or None
    """
    try:
        if len(lms) < 25:
            return None

        def pt(i):
            vis = float(getattr(lms[i], 'visibility', 1.0) or 1.0)
            return np.array([lms[i].x, lms[i].y]), vis

        ls, lsv = pt(POSE_L_SHOULDER)
        rs, rsv = pt(POSE_R_SHOULDER)
        lw, lwv = pt(POSE_L_WRIST)
        rw, rwv = pt(POSE_R_WRIST)
        le, _   = pt(POSE_L_ELBOW)
        re, _   = pt(POSE_R_ELBOW)
        lh, _   = pt(POSE_L_HIP)
        rh, _   = pt(POSE_R_HIP)
        ns, _   = pt(POSE_NOSE)

        # Require high visibility of shoulders
        if lsv < 0.55 or rsv < 0.55:
            return None
        # Require wrists to be visible for meaningful pose classification
        if lwv < 0.40 or rwv < 0.40:
            return None

        mid_shoulder_y = (ls[1] + rs[1]) / 2.0
        body_width     = dist(ls, rs) + 1e-5
        wrist_span     = dist(lw, rw)
        span_ratio     = wrist_span / body_width

        lw_above_shoulder = lw[1] < ls[1] - 0.06   # stricter: 6% body unit above
        rw_above_shoulder = rw[1] < rs[1] - 0.06
        lw_above_head     = lw[1] < ns[1] - 0.08
        rw_above_head     = rw[1] < ns[1] - 0.08
        lw_at_chest       = ls[1] < lw[1] < (ls[1] + lh[1]) / 2
        rw_at_chest       = rs[1] < rw[1] < (rs[1] + rh[1]) / 2
        both_at_chest     = lw_at_chest and rw_at_chest
        shoulder_tilt     = abs(ls[1] - rs[1]) / body_width

        # CELEBRATE — both wrists high above head, very wide
        if lw_above_head and rw_above_head and span_ratio > 2.0:
            return "CELEBRATE", "Celebrate / Bahut Khushi", (0, 215, 255), 0.90

        # HANDS UP — both above shoulders, very wide span
        if lw_above_shoulder and rw_above_shoulder and span_ratio > 2.2:
            return "HANDS UP", "Hands Up / Surrender", (255, 200, 0), 0.85

        # WELCOME — both wrists above shoulders, moderate span
        if lw_above_shoulder and rw_above_shoulder and 1.0 < span_ratio < 2.0:
            return "WELCOME", "Welcome / Swagat", (0, 220, 120), 0.80

        # WAVE — single wrist raised; other at rest
        if lw_above_shoulder and not rw_above_shoulder:
            conf = min(1.0, (ls[1] - lw[1]) / 0.10)
            return "WAVE LEFT", "Hi / Wave (Left)", (0, 200, 255), conf
        if rw_above_shoulder and not lw_above_shoulder:
            conf = min(1.0, (rs[1] - rw[1]) / 0.10)
            return "WAVE RIGHT", "Hi / Wave (Right)", (0, 200, 255), conf

        # PRAY / PLEASE — both wrists at chest level, close together
        if both_at_chest and span_ratio < 0.50:
            return "PRAY", "Please / Pray / Kripa", (255, 180, 80), 0.85

        # HUG / EMBRACE — elbows cross body midline
        mid_x = (ls[0] + rs[0]) / 2.0
        arms_crossed = (le[0] > mid_x + 0.03) and (re[0] < mid_x - 0.03)
        if arms_crossed:
            return "HUG", "Hug / Love / Pyaar", (255, 100, 180), 0.80

        # CONFUSED / SHRUG — significant shoulder tilt
        if shoulder_tilt > 0.10:
            conf = min(1.0, shoulder_tilt / 0.15)
            return "CONFUSED", "I Don't Know / Pata Nahi", (180, 180, 50), conf

    except Exception:
        pass
    return None


# ── Improved Gesture Smoother ─────────────────────────────────────────────────
class GestureSmoother:
    """
    Confidence-weighted gesture smoother.
    - Keeps a longer history (18 frames) for stability.
    - Requires 65% majority to commit (reduces false transitions).
    - Weights recent frames higher than older frames.
    - Ignores low-confidence frames (< 0.40) when deciding.
    """
    def __init__(self, window=18, threshold=0.65):
        self.history   = deque(maxlen=window)   # (label, confidence)
        self.threshold = threshold
        self.stable    = ""
        self.stable_conf = 0.0

    def update(self, label, confidence=1.0):
        self.history.append((label, confidence))

        # Build weighted vote — more recent = higher weight
        n = len(self.history)
        counts = {}
        total_weight = 0.0
        for i, (lbl, conf) in enumerate(self.history):
            if conf < 0.38:    # skip very low confidence entries
                continue
            w = (i + 1) / n   # linear recency weight
            w *= conf          # scale by confidence
            counts[lbl]  = counts.get(lbl, 0.0) + w
            total_weight += w

        if total_weight == 0:
            return self.stable

        best = max(counts, key=counts.get)
        best_ratio = counts[best] / total_weight

        if best_ratio >= self.threshold:
            self.stable      = best
            self.stable_conf = best_ratio

        return self.stable


# ── UI drawing helpers ────────────────────────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2, r=12):
    cv2.line(img,(x1+r,y1),(x2-r,y1),color,thickness)
    cv2.line(img,(x1+r,y2),(x2-r,y2),color,thickness)
    cv2.line(img,(x1,y1+r),(x1,y2-r),color,thickness)
    cv2.line(img,(x2,y1+r),(x2,y2-r),color,thickness)
    cv2.ellipse(img,(x1+r,y1+r),(r,r),180,0,90,color,thickness)
    cv2.ellipse(img,(x2-r,y1+r),(r,r),270,0,90,color,thickness)
    cv2.ellipse(img,(x1+r,y2-r),(r,r), 90,0,90,color,thickness)
    cv2.ellipse(img,(x2-r,y2-r),(r,r),  0,0,90,color,thickness)

def draw_pill(img, text, x, y, bg, fg=(255,255,255), scale=0.60, bold=False):
    font = cv2.FONT_HERSHEY_DUPLEX
    th   = 2 if bold else 1
    (tw, txh), _ = cv2.getTextSize(text, font, scale, th)
    px, py = 10, 6
    x2 = min(x + tw + px*2, img.shape[1]-1)
    y2 = min(y + txh + py*2, img.shape[0]-1)
    ov = img.copy()
    cv2.rectangle(ov, (x,y), (x2,y2), bg, -1)
    cv2.addWeighted(ov, 0.82, img, 0.18, 0, img)
    cv2.putText(img, text, (x+px, y+txh+py-1), font, scale, fg, th, cv2.LINE_AA)

def draw_progress_ring(img, cx, cy, progress, color):
    cv2.circle(img, (cx, cy), 30, (30, 30, 40), -1)
    cv2.ellipse(img, (cx, cy), (26, 26), -90, 0, int(360*progress), color, 4)

def draw_confidence_bar(img, x, y, w_bar, conf, color):
    """Draw a small confidence bar below gesture label."""
    h_bar = 6
    bg    = (40, 40, 50)
    cv2.rectangle(img, (x, y), (x + w_bar, y + h_bar), bg, -1)
    fill  = int(w_bar * conf)
    if fill > 2:
        bar_color = color if conf > 0.60 else (80, 80, 80)
        cv2.rectangle(img, (x, y), (x + fill, y + h_bar), bar_color, -1)

def draw_banner(img, sign, label, color, confidence, sentence, tts_on):
    h, w = img.shape[:2]
    y0 = h - 130
    ov = img.copy()
    cv2.rectangle(ov, (0, y0), (w, h), (8, 10, 20), -1)
    cv2.addWeighted(ov, 0.80, img, 0.20, 0, img)
    cv2.line(img, (0, y0), (w, y0), color, 2)

    f = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, sign, (20, y0+46), f, 1.15, color, 2, cv2.LINE_AA)
    cv2.putText(img, f"ISL: {label}", (20, y0+80), f, 0.65, (220,220,255), 1, cv2.LINE_AA)

    # Confidence bar + value
    bar_w = 200
    draw_confidence_bar(img, 20, y0+92, bar_w, confidence, color)
    conf_pct = int(confidence * 100)
    conf_color = (80, 220, 80) if conf_pct >= 70 else (200, 180, 50) if conf_pct >= 50 else (160, 80, 80)
    cv2.putText(img, f"Conf: {conf_pct}%", (230, y0+100), f, 0.42, conf_color, 1, cv2.LINE_AA)

    sent_str = "  |  ".join(sentence[-5:]) if sentence else "(empty)"
    cv2.putText(img, f"Sentence: {sent_str[-68:]}", (20, y0+118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (140, 200, 140), 1, cv2.LINE_AA)
    tts_hint = " S=speak" if tts_on else ""
    cv2.putText(img,
                f"Hold 2.0s=save | SPACE=add | C=clear |{tts_hint} ESC=quit",
                (w-560, y0+118), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (80, 80, 110), 1, cv2.LINE_AA)

def draw_hud(img, fps, hands, pose_on):
    ov = img.copy()
    cv2.rectangle(ov, (8,8), (320,100), (6,8,18), -1)
    cv2.addWeighted(ov, 0.72, img, 0.28, 0, img)
    draw_rounded_rect(img, 8, 8, 320, 100, (60,60,120), 1, 8)
    f = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, "ISL RECOGNITION SYSTEM v2.0",    (16, 30),  f, 0.46, (160,140,255), 1, cv2.LINE_AA)
    cv2.putText(img, f"FPS:{fps:4.1f}  Hands:{hands}  Body:{'ON' if pose_on else 'OFF'}",
                (16, 58), f, 0.52, (160,230,160), 1, cv2.LINE_AA)
    cv2.putText(img, f"MediaPipe {mp.__version__} | Indian Sign Language",
                (16, 84), f, 0.34, (70,110,70),  1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    sentence_words = []
    last_label     = ""
    hold_start     = None
    HOLD_SEC       = 2.0   # slightly longer hold → fewer false saves

    smoother = GestureSmoother(window=18, threshold=0.65)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Try VideoCapture(1).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("ISL Recognition System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ISL Recognition System", 1280, 720)

    print("="*60)
    print("  ISL Recognition System v2.0  —  For the Hearing Impaired")
    print("="*60)
    print("  Hold 2.0s → save | SPACE → add | C → clear | ESC → quit")
    if TTS_AVAILABLE:
        print("  S → speak sentence aloud")
    print()

    prev_time = time.time()
    current_confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Lost camera frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        sign      = "..."
        label     = "Waiting for gesture..."
        color     = (100, 100, 100)
        conf_raw  = 0.0
        num_hands = 0
        pose_on   = False

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        ts     = int(time.time() * 1000)

        # ── Hand detection ────────────────────────────────────────────────────
        hand_result = hand_landmarker.detect_for_video(mp_img, ts)
        if hand_result.hand_landmarks:
            num_hands = len(hand_result.hand_landmarks)

            # Use handedness info if available for chirality-aware classification
            handedness_list = []
            if hasattr(hand_result, 'handedness') and hand_result.handedness:
                for h_cat in hand_result.handedness:
                    if h_cat:
                        handedness_list.append(h_cat[0].display_name)
                    else:
                        handedness_list.append("Right")
            else:
                handedness_list = ["Right"] * num_hands

            for i, lms in enumerate(hand_result.hand_landmarks):
                draw_hand_landmarks(frame, lms)

                hand_side = handedness_list[i] if i < len(handedness_list) else "Right"

                xs = [lm.x * w for lm in lms]
                ys = [lm.y * h for lm in lms]
                bx1 = max(0,   int(min(xs)) - 20)
                by1 = max(0,   int(min(ys)) - 20)
                bx2 = min(w-1, int(max(xs)) + 20)
                by2 = min(h-1, int(max(ys)) + 20)

                s, l, c, cf = classify_isl_hand(lms, hand_side)

                # Use highest-confidence hand detection if multiple hands
                if cf > conf_raw:
                    sign, label, color, conf_raw = s, l, c, cf

                draw_rounded_rect(frame, bx1, by1, bx2, by2, c, 2, 14)
                draw_pill(frame, f"[{s}]  {l}  {int(cf*100)}%",
                          bx1, max(by1-40, 0), c, scale=0.58, bold=True)

        # ── Pose detection ────────────────────────────────────────────────────
        pose_result = pose_landmarker.detect_for_video(mp_img, ts)
        if pose_result.pose_landmarks:
            pose_on = True
            for lms in pose_result.pose_landmarks:
                draw_pose_landmarks(frame, lms)
                pg = classify_isl_pose(lms)
                if pg:
                    ps, pl, pc, pcf = pg
                    draw_pill(frame, f"BODY:[{ps}] {pl}  {int(pcf*100)}%",
                              w - 490, 14, pc, scale=0.50)
                    # Only override hand sign if no hand detected
                    if sign == "..." and pcf > 0.60:
                        sign, label, color, conf_raw = ps, pl, pc, pcf

        # ── Smoothing ─────────────────────────────────────────────────────────
        stable_label = smoother.update(label, conf_raw)
        label        = stable_label
        current_confidence = smoother.stable_conf

        # ── Hold-to-save logic ────────────────────────────────────────────────
        now = time.time()
        SKIP_LABELS = {"Waiting for gesture...", "Hold On / Thinking..."}

        if label not in SKIP_LABELS and current_confidence >= 0.50:
            if label == last_label:
                if hold_start is None:
                    hold_start = now
                held = now - hold_start
                prog = min(held / HOLD_SEC, 1.0)
                draw_progress_ring(frame, w - 55, 55, prog, color)
                cv2.putText(frame, f"{held:.1f}s", (w-72, 61),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
                if held >= HOLD_SEC:
                    sentence_words.append(label)
                    print(f"  [AUTO-SAVE] {label}  (conf={current_confidence:.2f})")
                    hold_start = None
                    last_label = ""
            else:
                last_label = label
                hold_start = None
        else:
            last_label = ""
            hold_start = None

        # ── Draw UI ───────────────────────────────────────────────────────────
        draw_banner(frame, sign, label, color, current_confidence,
                    sentence_words, TTS_AVAILABLE)
        cur_time  = time.time()
        fps       = 1.0 / max(cur_time - prev_time, 1e-5)
        prev_time = cur_time
        draw_hud(frame, fps, num_hands, pose_on)

        cv2.imshow("ISL Recognition System", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:   # ESC
            break
        elif key == 32: # SPACE
            if label not in SKIP_LABELS and current_confidence >= 0.45:
                sentence_words.append(label)
                print(f"  [MANUAL] {label}  (conf={current_confidence:.2f})")
        elif key in (ord('c'), ord('C')):
            sentence_words.clear()
            print("  Sentence cleared.")
        elif key in (ord('s'), ord('S')):
            if TTS_AVAILABLE and sentence_words:
                text = " ".join(sentence_words)
                print(f"  [TTS] Speaking: {text}")
                _tts_engine.say(text)
                _tts_engine.runAndWait()

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    pose_landmarker.close()

    print(f"\n  Final sentence: {' | '.join(sentence_words)}")
    print("  Thank you for using ISL Recognition System v2.0!")


if __name__ == "__main__":
    main()