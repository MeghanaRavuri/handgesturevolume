import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import math
import platform

st.set_page_config(page_title="Milestone 4 - Gesture UI", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
/* ---- Hide ALL Streamlit chrome ---- */
header[data-testid="stHeader"]          { display: none !important; }
#MainMenu                                { display: none !important; }
[data-testid="stToolbar"]               { display: none !important; }
[data-testid="stDecoration"]            { display: none !important; }
[data-testid="stStatusWidget"]          { display: none !important; }
footer                                   { display: none !important; }
.stDeployButton                          { display: none !important; }

/* Hide the search/curved bar that appears below header */
[data-testid="stAppViewBlockContainer"] > div:first-child > div:first-child {
    display: none !important;
}
div[class*="viewerBadge"]               { display: none !important; }
div[class*="stChatInput"]               { display: none !important; }

/* ---- Base ---- */
html, body, [data-testid="stAppViewContainer"], .stApp {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    background-color: #ffffff !important;
    color: #2d0057 !important;
}
[data-testid="stAppViewContainer"] > section { overflow: hidden !important; }

/* ---- SIDEBAR white/purple theme ---- */
section[data-testid="stSidebar"] {
    background-color: #f8f5ff !important;
    border-right: 2px solid #e0d0ff !important;
    min-width: 220px !important;
    max-width: 240px !important;
}
section[data-testid="stSidebar"] * { color: #2d0057 !important; }
section[data-testid="stSidebar"] label {
    color: #4c1d95 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stSlider > div > div > div { background: #7c3aed !important; }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #7c3aed !important; border-color: #7c3aed !important;
}
/* Volume bar in sidebar */
.sidebar-vol-bar-wrap {
    background: #e9d5ff;
    border-radius: 8px;
    height: 18px;
    width: 100%;
    border: 1px solid #d8b4fe;
    overflow: hidden;
    margin-top: 4px;
}
.sidebar-vol-bar-fill {
    background: #7c3aed;
    height: 100%;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* ---- Main block container ---- */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-height: 100vh !important;
    overflow: hidden !important;
}

/* ---- Text ---- */
label, p, span, div, .stMarkdown p {
    color: #2d0057 !important;
    font-size: 14px !important;
}

/* ---- Buttons ---- */
.stButton > button {
    background-color: #7c3aed !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 5px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 0.3rem 1rem !important;
    width: 100% !important;
    transition: background 0.2s ease;
}
.stButton > button:hover { background-color: #5b21b6 !important; }

.stAlert { margin-bottom: 3px !important; margin-top: 0 !important; }
div[data-testid="column"] { padding-left: 4px !important; padding-right: 4px !important; }
[data-testid="stImage"] { margin: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ===================== MEDIAPIPE SETUP =====================
mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils

def get_finger_distance(hand_landmarks, fw, fh):
    t = hand_landmarks.landmark[4]
    i = hand_landmarks.landmark[8]
    x1, y1 = int(t.x * fw), int(t.y * fh)
    x2, y2 = int(i.x * fw), int(i.y * fh)
    return math.hypot(x2-x1, y2-y1), (x1,y1), (x2,y2)

def classify_gesture(dist):
    if dist > 80:   return "Open Hand"
    elif dist > 20: return "Pinch"
    else:           return "Closed"

def get_quality(active):
    if active == "Open Hand": return "Detected",      (124, 58, 237)
    if active == "Pinch":     return "Good Detection",(34, 197, 94)
    if active == "Closed":    return "Detected",      (239, 68, 68)
    return "No Gesture", (160,160,160)

def draw_overlay(frame, active, quality_text, quality_bgr, tp, ip, dist, vol):
    h, w, _ = frame.shape

    # top strip
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,44), (20,0,40), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "Live Gesture Control", (10,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # thumb-index line
    if tp and ip:
        cv2.line(frame, tp, ip, (180,100,255), 2, cv2.LINE_AA)
        cv2.circle(frame, tp, 7, (124,58,237), -1, cv2.LINE_AA)
        cv2.circle(frame, ip, 7, (124,58,237), -1, cv2.LINE_AA)
        mid = ((tp[0]+ip[0])//2, (tp[1]+ip[1])//2)
        cv2.putText(frame, f"{int(dist)}px", (mid[0]+5, mid[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,220,100), 1, cv2.LINE_AA)

    # gesture pill
    if active != "None":
        pill = f"{active} Gesture"
        (tw,_),_ = cv2.getTextSize(pill, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (8,54), (tw+22,78), (200,80,20), -1, cv2.LINE_AA)
        cv2.putText(frame, pill, (14,71),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    # quality pill
    (qw,_),_ = cv2.getTextSize(quality_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (w-qw-28,54), (w-6,78), quality_bgr, -1, cv2.LINE_AA)
    cv2.putText(frame, quality_text, (w-qw-20,71),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    # volume bar — right edge, clearly visible
    bx1, bx2 = w-46, w-18
    bt, bb    = 90, h-20
    bh        = bb - bt
    filled    = int((vol/100) * bh)
    cv2.rectangle(frame, (bx1, bt), (bx2, bb), (225,210,255), -1)
    cv2.rectangle(frame, (bx1, bt), (bx2, bb), (150, 80,255), 2)
    if filled > 0:
        cv2.rectangle(frame, (bx1, bb-filled), (bx2, bb), (124,58,237), -1)
    cv2.putText(frame, "VOL", (bx1-2, bt-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,40,180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{vol}%", (bx1-2, bb+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,0,140), 1, cv2.LINE_AA)

    return frame

def open_camera_fast(index=0):
    # DirectShow on Windows skips slow MSMF negotiation (~3s saved)
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # minimal buffer = less latency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)  # fixed res = no format negotiation
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(3): cap.read()            # discard blank warm-up frames
    return cap

# ===================== STREAMLIT LAYOUT =====================

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("""<div style="font-size:15px;font-weight:700;color:#7c3aed;
                               padding:6px 0 4px 0;border-bottom:2px solid #e0d0ff;
                               margin-bottom:10px;">
        🎛 Simulation Controls</div>""", unsafe_allow_html=True)

    det_conf  = st.slider("Detection Confidence",  0.0, 1.0, 0.7)
    trk_conf  = st.slider("Tracking Confidence",   0.0, 1.0, 0.7)
    max_hands = st.slider("Max Hands",              1,   4,   2)

    st.markdown("<div style='margin-top:10px;font-weight:600;color:#4c1d95;font-size:13px;'>📊 Camera Info</div>", unsafe_allow_html=True)
    cam_info = st.empty()

    st.markdown("<div style='margin-top:10px;font-weight:600;color:#4c1d95;font-size:13px;'>🔊 Live Volume</div>", unsafe_allow_html=True)
    sidebar_vol = st.empty()

# ---- TOP BANNER (Milestone 4 only) ----
st.markdown("""
<div style="background:linear-gradient(90deg,#7c3aed,#9d4edd);
            padding:10px 16px;">
    <div style="font-size:16px;font-weight:800;color:white;margin-bottom:2px;">
        Milestone 4
    </div>
</div>
""", unsafe_allow_html=True)

# ---- CARD ----
st.markdown("""<div style="background:#ffffff;margin:8px 10px 0 10px;
                           border-radius:10px;border:1px solid #e0d0ff;
                           padding:8px 12px;">""", unsafe_allow_html=True)

# Sub-header + buttons
hc1, hc2 = st.columns([3, 2])
with hc1:
    st.markdown("""<div style="font-size:15px;font-weight:700;color:#2d0057;padding:4px 0;">
        🖐 Gesture Control Interface</div>""", unsafe_allow_html=True)
with hc2:
    b1, b2, b3 = st.columns(3)
    with b1: start    = st.button("▶ Start")
    with b2: stop     = st.button("⏹ Stop")
    with b3: st.button("⚙ Settings")

# Two-column body
lc, rc = st.columns([3, 2])

with lc:
    st.markdown('<div style="font-size:12px;font-weight:600;color:#555;margin-bottom:3px;">📷 Live Gesture Control</div>', unsafe_allow_html=True)
    FRAME_WINDOW = st.image([], use_container_width=True)

with rc:
    st.markdown('<div style="font-size:13px;font-weight:700;color:#2d0057;margin-bottom:5px;">🤚 Gesture Recognition</div>', unsafe_allow_html=True)
    row_open   = st.empty()
    row_pinch  = st.empty()
    row_closed = st.empty()

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;font-weight:700;color:#2d0057;margin-bottom:5px;">📈 Performance Metrics</div>', unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)
    mc3, mc4 = st.columns(2)
    vol_card  = mc1.empty()
    dist_card = mc2.empty()
    acc_card  = mc3.empty()
    resp_card = mc4.empty()

st.markdown("</div>", unsafe_allow_html=True)

# ---- RENDER HELPERS ----
GMETA = {
    "Open Hand": {"dot":"#22c55e", "desc":"Distance > 80mm"},
    "Pinch":     {"dot":"#f59e0b", "desc":"20mm < Distance < 80mm"},
    "Closed":    {"dot":"#ef4444", "desc":"Distance < 20mm"},
}

def render_gesture_row(ph, name, status):
    m = GMETA[name]
    active = status == "Active"
    bg  = "#f3e8ff" if active else "#fafafa"
    stc = "#7c3aed" if active else "#aaa"
    sbg = "#ede9fe" if active else "#f3f3f3"
    ph.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                background:{bg};border-radius:8px;padding:7px 10px;
                margin-bottom:5px;border:1px solid #e0d0ff;">
        <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:13px;height:13px;border-radius:50%;
                        background:{m['dot']};flex-shrink:0;"></div>
            <div>
                <div style="font-weight:700;font-size:13px;color:#2d0057;">{name}</div>
                <div style="font-size:11px;color:#888;">{m['desc']}</div>
            </div>
        </div>
        <div style="font-size:12px;font-weight:600;color:{stc};
                    background:{sbg};padding:2px 9px;border-radius:12px;">{status}</div>
    </div>""", unsafe_allow_html=True)

def render_metric(ph, value, label, color="#7c3aed"):
    ph.markdown(f"""
    <div style="background:#fff;border-radius:10px;border:1px solid #e0d0ff;
                padding:10px 6px;text-align:center;margin-bottom:4px;">
        <div style="font-size:22px;font-weight:800;color:{color};">{value}</div>
        <div style="font-size:11px;color:#888;margin-top:1px;">{label}</div>
    </div>""", unsafe_allow_html=True)

def render_sidebar_vol(ph, vol):
    ph.markdown(f"""
    <div style="background:#e9d5ff;border-radius:8px;height:22px;width:100%;
                border:1px solid #d8b4fe;overflow:hidden;margin-top:4px;">
        <div style="background:#7c3aed;width:{vol}%;height:100%;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;">
            <span style="color:white;font-size:12px;font-weight:700;">{vol}%</span>
        </div>
    </div>""", unsafe_allow_html=True)

# ---- DEFAULT STATE — dashes, NOT numbers ----
render_gesture_row(row_open,   "Open Hand", "Inactive")
render_gesture_row(row_pinch,  "Pinch",     "Inactive")
render_gesture_row(row_closed, "Closed",    "Inactive")
render_metric(vol_card,  "—",   "Current Volume")
render_metric(dist_card, "—",   "Finger Distance", "#f59e0b")
render_metric(acc_card,  "—",   "Camera FPS",      "#22c55e")
render_metric(resp_card, "—",   "Response Time",   "#7c3aed")
render_sidebar_vol(sidebar_vol, 0)

# ===================== CAMERA LOOP =====================
if start and not stop:

    cap = open_camera_fast(0)

    if not cap.isOpened():
        lc.error("❌ Cannot open camera")
    else:
        hands_model = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,           # lite = fastest
            min_detection_confidence=det_conf,
            min_tracking_confidence=trk_conf
        )

        prev_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands_model.process(rgb)
            rgb.flags.writeable = True

            dist, tp, ip, active = 0, None, None, "None"

            if results.multi_hand_landmarks:
                for hlm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hlm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(180,100,255), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(124,58,237),  thickness=2)
                    )
                    dist, tp, ip = get_finger_distance(hlm, w, h)
                active = classify_gesture(dist)

            # Volume: pinch controls it live; otherwise stays at last known
            vol_live = int(np.clip((dist - 20) / 80 * 100, 0, 100)) if active == "Pinch" and dist > 0 else (int(np.clip(dist / 160 * 100, 0, 100)) if dist > 0 else 0)

            qt, qbgr = get_quality(active)
            frame = draw_overlay(frame, active, qt, qbgr, tp, ip, dist, vol_live)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            rms = round(1000/fps, 1) if fps > 0 else 0

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Gesture rows
            sts = {"Open Hand":"Inactive","Pinch":"Inactive","Closed":"Inactive"}
            if active != "None": sts[active] = "Active"
            render_gesture_row(row_open,   "Open Hand", sts["Open Hand"])
            render_gesture_row(row_pinch,  "Pinch",     sts["Pinch"])
            render_gesture_row(row_closed, "Closed",    sts["Closed"])

            # Metrics
            render_metric(vol_card,  f"{vol_live}%",      "Current Volume")
            render_metric(dist_card, f"{int(dist)}px",    "Finger Distance", "#f59e0b")
            render_metric(acc_card,  f"{int(fps)} FPS",   "Camera FPS",      "#22c55e")
            render_metric(resp_card, f"{rms}ms",          "Response Time",   "#7c3aed")

            # Sidebar
            render_sidebar_vol(sidebar_vol, vol_live)
            cam_info.markdown(f"""
            <div style="background:#ede9fe;border-radius:8px;padding:8px;
                        font-size:12px;color:#4c1d95;margin-top:4px;">
                📐 {w}×{h}<br>
                🎯 <b>{active}</b><br>
                📏 {int(dist)}px &nbsp;|&nbsp; ⚡ {int(fps)} FPS
            </div>""", unsafe_allow_html=True)

            if stop:
                break

        cap.release()
        hands_model.close()