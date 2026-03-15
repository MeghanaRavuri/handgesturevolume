import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math

st.set_page_config(layout="wide", page_title="Gesture Volume Control")

PX_TO_MM = 0.2646

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');

html, body {
    height: 100% !important;
    margin: 0 !important;
    overflow: hidden !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main {
    background: #f7f5ff !important;
    height: 100vh !important;
    overflow: hidden !important;
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu, footer { display: none !important; }

.block-container {
    padding: 10px 18px 0 18px !important;
    max-width: 100% !important;
    overflow: hidden !important;
}

[data-testid="stVerticalBlock"] { gap: 0 !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
.element-container { margin: 2px !important; padding: 0 !important; }
.stMarkdown { margin: 2px !important; padding: 0 !important; }
div[data-testid="column"] { padding: 0 4px !important; }

/* Camera — full width matching volume bar, capped height */
[data-testid="stImage"] { line-height: 0 !important; width: 100% !important; }
[data-testid="stImage"] img {
    border-radius: 12px !important;
    width: 100% !important;
    min-width: 100% !important;
    max-height: 430px !important;
    object-fit: fill !important;
    display: block !important;
}

.stButton > button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 22px !important;
    border-radius: 22px !important;
    border: none !important;
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(109,40,217,.35) !important;
    cursor: pointer !important;
    white-space: nowrap !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #5b21b6) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Right panel ───────────────────────────────────────────────────────────────
def right_panel(dist_mm: float, gesture: str) -> str:
    dv = int(round(dist_mm))
    dp = min(dist_mm / 60.0, 1.0) * 100
    t  = max(min(dp, 96), 4)

    slider = (
        '<div style="position:relative;height:7px;background:#e8e1ff;'
        'border-radius:99px;margin:10px 0 4px;">'
        f'<div style="position:absolute;left:0;top:0;height:100%;width:{dp:.1f}%;'
        'background:linear-gradient(90deg,#7c3aed,#a78bfa);border-radius:99px;"></div>'
        f'<div style="position:absolute;top:50%;left:{t:.1f}%;'
        'transform:translate(-50%,-50%);width:14px;height:14px;'
        'background:#fff;border:3px solid #7c3aed;border-radius:50%;'
        'box-shadow:0 2px 8px rgba(109,40,217,.5);"></div>'
        '</div>'
        '<div style="display:flex;justify-content:space-between;'
        'font-family:Outfit,sans-serif;font-size:11px;color:#b0a5d8;">'
        '<span>0mm</span><span>30mm</span><span>60mm</span></div>'
    )

    def card(html, mb="10px"):
        return (
            f'<div style="background:#fff;border-radius:14px;padding:14px 16px 12px;'
            f'margin-bottom:{mb};box-shadow:0 2px 10px rgba(109,40,217,.09);">{html}</div>'
        )

    def sec(icon, label):
        return (
            f'<div style="font-family:Outfit,sans-serif;font-size:11px;font-weight:600;'
            f'color:#9b8ec4;text-transform:uppercase;letter-spacing:.09em;'
            f'margin-bottom:8px;">{icon}&nbsp;&nbsp;{label}</div>'
        )

    def grow(name, sub, dbg, dfg):
        active = gesture == name
        bg  = "#f3eeff" if active else "transparent"
        bdr = "#c4b5fd" if active else "#ece8f8"
        return (
            f'<div style="display:flex;align-items:center;gap:10px;background:{bg};'
            f'border:1.5px solid {bdr};border-radius:10px;padding:8px 12px;margin-bottom:6px;">'
            f'<div style="width:22px;height:22px;border-radius:50%;flex-shrink:0;'
            f'background:{dbg};color:{dfg};display:flex;align-items:center;'
            f'justify-content:center;font-size:10px;">&#9679;</div>'
            f'<div>'
            f'<div style="font-family:Outfit,sans-serif;font-size:13px;'
            f'font-weight:600;color:#1a103c;">{name}</div>'
            f'<div style="font-family:Outfit,sans-serif;font-size:11px;'
            f'color:#b0a5d8;margin-top:1px;">{sub}</div>'
            f'</div></div>'
        )

    dist_card = card(
        sec("&#128207;", "Distance Measurement") +
        f'<div style="font-family:Outfit,sans-serif;font-size:44px;font-weight:800;'
        f'color:#7c3aed;line-height:1.1;margin-bottom:3px;">{dv}</div>'
        f'<div style="font-family:Outfit,sans-serif;font-size:12px;'
        f'color:#b0a5d8;margin-bottom:8px;">millimeters</div>' + slider
    )

    gesture_card = card(
        sec("&#9995;", "Gesture States") +
        grow("Open Hand", "Distance > 40mm",        "#d1fae5", "#059669") +
        grow("Pinch",     "10mm < Distance < 40mm", "#fef3c7", "#d97706") +
        grow("Closed",    "Distance < 10mm",         "#fee2e2", "#dc2626"),
        mb="0"
    )

    return (
        '<div style="padding-left:8px;">'
        + dist_card + gesture_card +
        '</div>'
    )


def vol_strip(pct: int) -> str:
    """Volume bar rendered ABOVE the camera feed — always visible."""
    return (
        f'<div style="background:#fff;border-radius:10px;'
        f'padding:9px 16px 8px;margin-bottom:6px;'
        f'box-shadow:0 2px 8px rgba(109,40,217,.08);">'
        f'<div style="font-family:Outfit,sans-serif;font-size:13px;'
        f'font-weight:700;color:#1a103c;margin-bottom:5px;">&#128266;&nbsp; Volume: {pct}%</div>'
        f'<div style="height:8px;background:#e8e1ff;border-radius:99px;overflow:hidden;">'
        f'<div style="height:100%;width:{pct}%;'
        f'background:linear-gradient(90deg,#7c3aed,#a78bfa);'
        f'border-radius:99px;transition:width .15s ease;"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-family:Outfit,sans-serif;font-size:10px;color:#b0a5d8;margin-top:3px;">'
        f'<span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>'
        f'</div></div>'
    )


# ── AUDIO (optional — Windows only) ──────────────────────────────────────────
AUDIO_OK = False
vol_ctrl = None
minVol, maxVol = -65.0, 0.0
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    _dev   = AudioUtilities.GetSpeakers()
    _iface = _dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    vol_ctrl = _iface.QueryInterface(IAudioEndpointVolume)
    vr = vol_ctrl.GetVolumeRange()
    minVol, maxVol = vr[0], vr[1]
    AUDIO_OK = True
except Exception:
    AUDIO_OK = False


# ── HEADER ────────────────────────────────────────────────────────────────────
hl, hr = st.columns([3, 2])
with hl:
    st.markdown(
        '<p style="font-family:Outfit,sans-serif;font-size:20px;font-weight:700;'
        'color:#1a103c;margin:0;letter-spacing:-.02em;padding:8px 0 6px 0;">'
        'Gesture Volume Control</p>',
        unsafe_allow_html=True,
    )
with hr:
    b1, b2, b3 = st.columns(3)
    start    = b1.button("▶ Start")
    pause    = b2.button("⏸ Pause")
    settings = b3.button("⚙ Settings")

st.markdown(
    '<hr style="border:none;border-top:1.5px solid #e0d8ff;margin:4px 0 8px 0;">',
    unsafe_allow_html=True,
)

# ── MAIN COLUMNS ──────────────────────────────────────────────────────────────
cam, panel = st.columns([3.5, 1.4])

with cam:
    # ✅ Volume bar FIRST — always visible at the top of the camera column
    vol_ph   = st.empty()
    # Camera feed below — height capped via CSS so bar never gets pushed off
    frame_ph = st.empty()

with panel:
    panel_ph = st.empty()

# ── Initial render ────────────────────────────────────────────────────────────
vol_ph.markdown(vol_strip(0), unsafe_allow_html=True)
panel_ph.markdown(right_panel(0.0, "Open Hand"), unsafe_allow_html=True)
frame_ph.markdown(
    '<div style="background:#1a103c;border-radius:12px;height:400px;'
    'display:flex;align-items:center;justify-content:center;'
    'color:#9b8ec4;font-family:Outfit,sans-serif;font-size:16px;">'
    '&#128247;&nbsp; Press &#9654; Start to begin</div>',
    unsafe_allow_html=True,
)

# ── MEDIAPIPE + CAMERA LOOP ───────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

if start:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Warm-up frames for fast open
    for _ in range(3): cap.read()

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            dist_px    = 0
            dist_mm    = 0.0
            gesture    = "Open Hand"
            volPercent = 0

            if results.multi_hand_landmarks:
                for lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, lms, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(160, 60, 255), thickness=2, circle_radius=4),
                        mp_draw.DrawingSpec(color=(200, 160, 255), thickness=2),
                    )

                    h, w, _ = frame.shape
                    th = lms.landmark[4]
                    ix = lms.landmark[8]
                    x1, y1 = int(th.x * w), int(th.y * h)
                    x2, y2 = int(ix.x * w), int(ix.y * h)

                    dist_px = int(math.hypot(x2 - x1, y2 - y1))
                    dist_mm = dist_px * PX_TO_MM

                    # Classify gesture
                    if dist_mm > 40:
                        gesture      = "Open Hand"
                        badge_color  = (30, 160, 60)
                    elif 10 < dist_mm <= 40:
                        gesture      = "Pinch"
                        badge_color  = (20, 120, 200)
                    else:
                        gesture      = "Closed"
                        badge_color  = (40, 40, 200)

                    # Thumb + index dots and line
                    cv2.circle(frame, (x1, y1), 10, (255, 0, 180), -1)
                    cv2.circle(frame, (x2, y2), 10, (255, 0, 180), -1)
                    cv2.line(frame, (x1, y1), (x2, y2), (200, 80, 255), 3)

                    # Distance label at midpoint
                    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                    lbl = f"{int(round(dist_mm))}mm"
                    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(frame,
                                  (mx - lw//2 - 8, my - lh - 12),
                                  (mx + lw//2 + 8, my + 4),
                                  (30, 10, 60), -1)
                    cv2.putText(frame, lbl, (mx - lw//2, my - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 160, 255), 2)

                    # Gesture badge top-left
                    bl = f"{gesture} Gesture"
                    (bw, bh), _ = cv2.getTextSize(bl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (12, 12), (12 + bw + 20, 12 + bh + 14),
                                  badge_color, -1)
                    cv2.putText(frame, bl, (22, 12 + bh + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Map distance to volume
                    volPercent = int(np.interp(dist_px, [20, 200], [0, 100]))
                    if AUDIO_OK and vol_ctrl is not None:
                        vol_db = np.interp(dist_px, [20, 200], [minVol, maxVol])
                        try:
                            vol_ctrl.SetMasterVolumeLevel(float(vol_db), None)
                        except Exception:
                            pass

            # ✅ Update volume bar FIRST, then frame — order matches layout
            vol_ph.markdown(vol_strip(volPercent), unsafe_allow_html=True)
            frame_ph.image(frame, channels="BGR", use_container_width=True)
            panel_ph.markdown(right_panel(dist_mm, gesture), unsafe_allow_html=True)

            if pause:
                break

    cap.release()