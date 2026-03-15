import streamlit as st
import cv2
import mediapipe as mp
import time
import platform

st.set_page_config(page_title="Milestone 1", layout="wide")

# ---------------- WHITE & PURPLE THEME CSS ----------------
st.markdown("""
<style>
/* ---- Hide Streamlit default black header/toolbar ---- */
header[data-testid="stHeader"] {
    display: none !important;
    height: 0 !important;
    visibility: hidden !important;
}
#MainMenu { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
footer { display: none !important; }
.stDeployButton { display: none !important; }

/* ---- Remove scroll, fix full viewport ---- */
html, body, [data-testid="stAppViewContainer"], .stApp {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    background-color: #ffffff !important;
    color: #2d0057 !important;
}
[data-testid="stAppViewContainer"] > section { overflow: hidden !important; }

/* ---- Main block container ---- */
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0.2rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    background-color: #ffffff !important;
    max-height: 100vh !important;
    overflow: hidden !important;
}

/* ---- Headings ---- */
h1, h2, h3, h4, h5, h6 {
    color: #6a0dad !important;
    margin-bottom: 4px !important;
    margin-top: 4px !important;
}
h2 { font-size: 20px !important; }
h3 { font-size: 16px !important; }
h4 { font-size: 15px !important; }

/* ---- Labels & text ---- */
label, p, span, div, .stMarkdown p {
    color: #2d0057 !important;
    font-size: 15px !important;
}
.stSlider label { font-size: 14px !important; font-weight: 600 !important; }
hr { margin-top: 4px; margin-bottom: 4px; border-color: #d8b4fe; }

/* ---- Buttons ---- */
.stButton > button {
    background-color: #7c3aed !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 0.35rem 1rem !important;
    width: 100% !important;
    transition: background 0.2s ease;
}
.stButton > button:hover {
    background-color: #5b21b6 !important;
    color: #ffffff !important;
}

/* ---- Sliders ---- */
.stSlider > div > div > div { background: #7c3aed !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #7c3aed !important;
    border-color: #7c3aed !important;
}

/* ---- Alert boxes ---- */
[data-testid="stNotificationContentSuccess"] {
    background-color: #f3e8ff !important; color: #5b21b6 !important;
    border-left: 4px solid #7c3aed !important; font-size: 15px !important; padding: 6px 10px !important;
}
[data-testid="stNotificationContentInfo"] {
    background-color: #ede9fe !important; color: #4c1d95 !important;
    border-left: 4px solid #8b5cf6 !important; font-size: 15px !important; padding: 6px 10px !important;
}
[data-testid="stNotificationContentWarning"] {
    background-color: #fdf4ff !important; color: #7e22ce !important;
    border-left: 4px solid #c084fc !important; font-size: 15px !important; padding: 6px 10px !important;
}
[data-testid="stNotificationContentError"] {
    background-color: #fce7f3 !important; color: #831843 !important;
    border-left: 4px solid #e879f9 !important; font-size: 15px !important; padding: 6px 10px !important;
}

.stAlert { margin-bottom: 4px !important; margin-top: 0px !important; }
div[data-testid="column"] { padding-left: 6px !important; padding-right: 6px !important; }
[data-testid="stImage"] { margin-top: 0 !important; margin-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
    <div style="background-color:#7c3aed;padding:8px 12px;border-radius:5px;margin-bottom:6px;">
        <h2 style="color:white !important;margin:0 0 2px 0;font-size:68px !important;">
        Milestone 1: Webcam Input and Hand Detection Module
        </h2>
    </div>
""", unsafe_allow_html=True)

# ---------------- BUTTON ROW ----------------
col_btn1, col_btn2, col_btn3 = st.columns(3)
with col_btn1:
    start = st.button("▶ Start Camera")
with col_btn2:
    stop = st.button("⏹ Stop Camera")
with col_btn3:
    capture = st.button("📸 Capture")

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([3, 2])
FRAME_WINDOW = col1.image([], use_container_width=True)

with col2:
    status_col, param_col, info_col = st.columns(3)

    with status_col:
        st.markdown("### 👁 Status")
        camera_status = st.empty()
        hands_status = st.empty()
        fps_status = st.empty()
        model_status = st.empty()

    with param_col:
        st.markdown("### ⚙ Parameters")
        min_det_conf = st.slider("Detection Confidence", 0.0, 1.0, 0.7)
        min_track_conf = st.slider("Tracking Confidence", 0.0, 1.0, 0.7)
        max_hands = st.slider("Max Hands", 1, 4, 2)

    with info_col:
        st.markdown("### 📊 Info")
        landmark_info = st.empty()
        connection_info = st.empty()
        resolution_info = st.empty()
        time_info = st.empty()

# ---------------- FAST CAMERA OPEN HELPER ----------------
def open_camera_fast(index=0):
    """
    Opens the camera as fast as possible.
    - On Windows: uses DirectShow (CAP_DSHOW) to skip slow MSMF negotiation
    - On Linux/Mac: uses V4L2 / default backend
    Explicitly sets a lower buffer size and fixed resolution to reduce init time.
    """
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

    if not cap.isOpened():
        # fallback to default backend
        cap = cv2.VideoCapture(index)

    # Set small buffer — reduces latency on startup
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Fix resolution to avoid auto-negotiation delay
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Fix FPS explicitly — avoids camera probing all modes
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Warm up: read and discard a few frames so first real frame is instant
    for _ in range(3):
        cap.read()

    return cap

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

if start and not stop:

    camera_status.info("Opening camera...")

    cap = open_camera_fast(0)

    if not cap.isOpened():
        camera_status.error("Camera Error: Cannot open")
    else:
        # Load MediaPipe with static_image_mode=False for speed
        hands = mp_hands.Hands(
            static_image_mode=False,          # streaming mode — faster
            max_num_hands=max_hands,
            model_complexity=0,               # 0 = lite model, fastest
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf
        )

        camera_status.success("Camera Active")
        model_status.success("Model Loaded")

        prev_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                camera_status.error("Camera Error")
                break

            frame = cv2.flip(frame, 1)

            # Convert once and reuse
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Mark as not writeable for faster MediaPipe processing
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                hands_status.success(f"Hands: {num_hands}")
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmark_info.info("21 Landmarks/Hand")
                connection_info.info("20 Connections/Hand")
            else:
                hands_status.warning("Hands: 0")
                landmark_info.info("0 Landmarks")
                connection_info.info("0 Connections")

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            fps_status.info(f"FPS: {int(fps)}")
            resolution_info.info(f"{frame.shape[1]}x{frame.shape[0]}")
            time_info.info(f"{round(1000/fps, 2) if fps > 0 else 0} ms")

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if stop:
                break

        cap.release()