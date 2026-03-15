"""
GestureVol — Hand Gesture Volume Control System
Single page · One camera · All features · Clean layout
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math, time, platform

st.set_page_config(layout="wide", page_title="GestureVol",
                   initial_sidebar_state="collapsed")

PX_TO_MM = 0.2646

# ══════════════════════════════════════════════════════════
#  CSS — minimal, no overflow, proper spacing
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Mono:wght@400;500&display=swap');

header,footer,#MainMenu,[data-testid="stToolbar"],
[data-testid="stDecoration"],[data-testid="stStatusWidget"],
.stDeployButton,section[data-testid="stSidebar"],
button[data-testid="collapsedControl"] { display:none !important; }

html, body {
    margin:0 !important; padding:0 !important;
    background:#0d0b16 !important;
}
[data-testid="stAppViewContainer"], .main {
    background:#0d0b16 !important;
}
/* ---- Remove ALL default gaps ---- */
.block-container {
    padding: 0 12px 12px 12px !important;
    max-width:100% !important;
}
[data-testid="stVerticalBlock"] { gap:0 !important; }
div[data-testid="stVerticalBlock"] > div { gap:0 !important; }
.element-container { margin:0 !important; padding:0 !important; }
.stMarkdown { margin:0 !important; padding:0 !important; }
div[data-testid="column"] { padding:0 5px !important; }

/* ---- Buttons ---- */
.stButton > button {
    font-family:'Syne',sans-serif !important;
    font-weight:700 !important; font-size:12px !important;
    padding:8px 0 !important; border-radius:8px !important;
    border:1px solid rgba(139,92,246,0.4) !important;
    background:rgba(109,40,217,0.22) !important;
    color:#c4b5fd !important; width:100% !important;
    margin-top:0 !important;
}
.stButton > button:hover {
    background:rgba(109,40,217,0.5) !important;
    color:#fff !important;
}

/* ---- Sliders ---- */
.stSlider { padding:0 !important; margin:0 !important; }
.stSlider > div > div > div { background:#7c3aed !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background:#c084fc !important; border-color:#7c3aed !important;
    width:12px !important; height:12px !important;
}
.stSlider label {
    font-family:'Syne',sans-serif !important;
    font-size:10px !important; color:#6b5b9a !important;
    font-weight:700 !important; margin:0 !important;
}

/* ---- Alerts ---- */
[data-testid="stNotificationContentSuccess"] {
    background:rgba(34,197,94,.1) !important; color:#86efac !important;
    border-left:3px solid #22c55e !important;
    font-size:11px !important; padding:3px 8px !important;
    font-family:'Syne',sans-serif !important;
}
[data-testid="stNotificationContentInfo"] {
    background:rgba(139,92,246,.1) !important; color:#c4b5fd !important;
    border-left:3px solid #8b5cf6 !important;
    font-size:11px !important; padding:3px 8px !important;
    font-family:'Syne',sans-serif !important;
}
[data-testid="stNotificationContentWarning"] {
    background:rgba(245,158,11,.1) !important; color:#fcd34d !important;
    border-left:3px solid #f59e0b !important;
    font-size:11px !important; padding:3px 8px !important;
    font-family:'Syne',sans-serif !important;
}
.stAlert { margin:2px 0 !important; }

/* ---- Images ---- */
[data-testid="stImage"] { line-height:0 !important; }
[data-testid="stImage"] img {
    border-radius:10px !important;
    width:100% !important; display:block !important;
    max-height:920px !important; object-fit:fill !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  AUDIO
# ══════════════════════════════════════════════════════════
AUDIO_OK = False; vol_ctrl = None; minVol, maxVol = -65.0, 0.0
try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    _dev = AudioUtilities.GetSpeakers()
    _iface = _dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    vol_ctrl = _iface.QueryInterface(IAudioEndpointVolume)
    vr = vol_ctrl.GetVolumeRange(); minVol, maxVol = vr[0], vr[1]
    AUDIO_OK = True
except Exception: pass

# ══════════════════════════════════════════════════════════
#  MEDIAPIPE HELPERS
# ══════════════════════════════════════════════════════════
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

def open_cam(idx=0):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if platform.system()=="Windows" else cv2.CAP_V4L2)
    if not cap.isOpened(): cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_FPS,30)
    for _ in range(3): cap.read()
    return cap

def fdist(lm,w,h):
    t=lm.landmark[4]; i=lm.landmark[8]
    x1,y1=int(t.x*w),int(t.y*h); x2,y2=int(i.x*w),int(i.y*h)
    return math.hypot(x2-x1,y2-y1),(x1,y1),(x2,y2)

def classify(mm):
    return "Open Hand" if mm>40 else ("Pinch" if mm>10 else "Closed")

def sys_vol(px):
    pct=int(np.interp(px,[20,300],[0,100]))  # extended range so 100% is reachable
    if AUDIO_OK and vol_ctrl:
        try: vol_ctrl.SetMasterVolumeLevel(float(np.interp(px,[20,300],[minVol,maxVol])),None)
        except: pass
    return pct

def draw_frame(frame,gesture,quality,tp,ip,dist_mm,vol,fps,nh):
    h,w,_=frame.shape
    # top bar
    ov=frame.copy(); cv2.rectangle(ov,(0,0),(w,42),(7,4,16),-1)
    cv2.addWeighted(ov,.78,frame,.22,0,frame)
    cv2.putText(frame,"GestureVol",(10,27),cv2.FONT_HERSHEY_SIMPLEX,.75,(170,110,255),2,cv2.LINE_AA)
    txt=f"FPS:{int(fps)}  HANDS:{nh}  VOL:{vol}%"
    tw,_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,.42,1)[0]
    cv2.putText(frame,txt,(w-tw-10,27),cv2.FONT_HERSHEY_SIMPLEX,.42,(110,80,180),1,cv2.LINE_AA)
    # measurement line
    if tp and ip:
        cv2.line(frame,tp,ip,(185,75,255),2,cv2.LINE_AA)
        cv2.circle(frame,tp,7,(255,45,195),-1,cv2.LINE_AA)
        cv2.circle(frame,ip,7,(255,45,195),-1,cv2.LINE_AA)
        mx,my=(tp[0]+ip[0])//2,(tp[1]+ip[1])//2
        lbl=f"{int(round(dist_mm))}mm"
        lw,lh=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,.55,2)[0]
        cv2.rectangle(frame,(mx-lw//2-6,my-lh-8),(mx+lw//2+6,my+3),(16,5,36),-1)
        cv2.putText(frame,lbl,(mx-lw//2,my-1),cv2.FONT_HERSHEY_SIMPLEX,.55,(205,145,255),2,cv2.LINE_AA)
    # bottom bar
    ov2=frame.copy(); cv2.rectangle(ov2,(0,h-46),(w,h),(7,4,16),-1)
    cv2.addWeighted(ov2,.78,frame,.22,0,frame)
    GC={"Open Hand":(34,197,94),"Pinch":(167,139,250),"Closed":(245,158,11)}
    if gesture!="None":
        gc=GC.get(gesture,(120,80,200))
        gl=f"  {gesture.upper()}  "
        gw,gh=cv2.getTextSize(gl,cv2.FONT_HERSHEY_SIMPLEX,.5,1)[0]
        cv2.rectangle(frame,(10,h-38),(10+gw+4,h-38+gh+8),gc,-1,cv2.LINE_AA)
        cv2.putText(frame,gl,(12,h-27),cv2.FONT_HERSHEY_SIMPLEX,.5,(10,4,18),1,cv2.LINE_AA)
    QC={"DETECTED":(120,55,230),"GOOD DETECT":(34,197,94),"NO HAND":(55,55,75)}
    qc=QC.get(quality,(75,55,115))
    ql=f"  {quality}  "
    qw,_=cv2.getTextSize(ql,cv2.FONT_HERSHEY_SIMPLEX,.4,1)[0]
    cv2.rectangle(frame,(w-qw-14,h-35),(w-8,h-35+18),qc,-1,cv2.LINE_AA)
    cv2.putText(frame,ql,(w-qw-12,h-23),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
    # vol bar
    bw=int((vol/100)*(w-4))
    cv2.rectangle(frame,(2,h-6),(w-2,h-2),(22,10,44),-1)
    if bw>0: cv2.rectangle(frame,(2,h-6),(2+bw,h-2),(148,70,248),-1)
    return frame

# ══════════════════════════════════════════════════════════
#  HTML BUILDING BLOCKS
# ══════════════════════════════════════════════════════════

# Section heading — consistent across all panels
def H(icon, label):
    return (
        f'<div style="font-family:Syne,sans-serif;font-size:11px;font-weight:700;'
        f'color:#7c5cbf;text-transform:uppercase;letter-spacing:0.1em;'
        f'padding:10px 0 5px 0;border-bottom:1px solid rgba(139,92,246,0.15);'
        f'margin-bottom:8px;">{icon}&nbsp; {label}</div>'
    )

# Glass card wrapper
def C(inner):
    return (
        '<div style="background:rgba(255,255,255,0.03);'
        'border:1px solid rgba(139,92,246,0.16);border-radius:10px;'
        'padding:12px 14px;margin-bottom:8px;">'
        + inner + '</div>'
    )

def vol_card(pct):
    col="#22c55e" if pct>60 else "#a78bfa" if pct>30 else "#f59e0b"
    bar=max(pct,1)
    return C(
        H("🔊","Volume Control") +
        f'<div style="text-align:center;padding:6px 0 4px;">'
        f'<span style="font-family:Syne,sans-serif;font-size:54px;font-weight:800;'
        f'color:{col};letter-spacing:-2px;line-height:1;">{pct}</span>'
        f'<span style="font-family:Syne,sans-serif;font-size:20px;color:#4a3a6e;'
        f'font-weight:600;">%</span>'
        f'<div style="font-family:Syne,sans-serif;font-size:9px;color:#5a4d80;'
        f'text-transform:uppercase;letter-spacing:0.1em;margin-top:4px;">Current Volume</div>'
        f'</div>'
        f'<div style="background:rgba(109,40,217,.12);border-radius:99px;'
        f'height:6px;margin:8px 0 4px;overflow:hidden;">'
        f'<div style="height:100%;width:{bar}%;background:{col};border-radius:99px;"></div></div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-family:DM Mono,monospace;font-size:9px;color:#3a2d5a;">'
        f'<span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>'
    )

def dist_card(dist_px, dist_mm_val):
    dp=min(dist_mm_val/60.0,1.0)*100; t=max(min(dp,96),4)
    return C(
        H("📏","Distance Measurement") +
        f'<div style="display:flex;align-items:flex-end;gap:18px;padding:4px 0 8px;">'
        f'<div><div style="font-family:DM Mono,monospace;font-size:32px;'
        f'font-weight:500;color:#7c3aed;line-height:1;">{dist_px}</div>'
        f'<div style="font-family:Syne,sans-serif;font-size:9px;color:#5a4d80;margin-top:2px;">pixels</div></div>'
        f'<div style="width:1px;height:36px;background:rgba(139,92,246,.18);align-self:center;"></div>'
        f'<div><div style="font-family:DM Mono,monospace;font-size:32px;'
        f'font-weight:500;color:#c084fc;line-height:1;">{dist_mm_val}</div>'
        f'<div style="font-family:Syne,sans-serif;font-size:9px;color:#5a4d80;margin-top:2px;">mm</div></div>'
        f'</div>'
        f'<div style="position:relative;height:5px;background:rgba(109,40,217,.15);border-radius:99px;margin-bottom:4px;">'
        f'<div style="position:absolute;left:0;top:0;height:100%;width:{dp:.0f}%;'
        f'background:linear-gradient(90deg,#7c3aed,#c084fc);border-radius:99px;"></div>'
        f'<div style="position:absolute;top:50%;left:{t:.0f}%;transform:translate(-50%,-50%);'
        f'width:10px;height:10px;background:#c084fc;border:2px solid #0d0b16;border-radius:50%;"></div></div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-family:DM Mono,monospace;font-size:9px;color:#3a2d5a;">'
        f'<span>0mm</span><span>30mm</span><span>60mm</span></div>'
    )

def gest_card(gesture):
    rows=""
    for name,sub,dot in [
        ("Open Hand","Distance > 40mm","#22c55e"),
        ("Pinch","10mm < Distance < 40mm","#a78bfa"),
        ("Closed","Distance < 10mm","#f59e0b"),
    ]:
        active=gesture==name
        bg="rgba(139,92,246,.12)" if active else "transparent"
        bdr="rgba(139,92,246,.35)" if active else "rgba(255,255,255,.04)"
        tc="#e9d5ff" if active else "#5a4d80"
        badge=(
            '<span style="font-family:Syne,sans-serif;font-size:8px;font-weight:700;'
            'color:#a78bfa;background:rgba(139,92,246,.18);'
            'padding:1px 6px;border-radius:4px;margin-left:auto;">ACTIVE</span>'
        ) if active else ""
        rows+=(
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'background:{bg};border:1px solid {bdr};border-radius:8px;'
            f'padding:7px 10px;margin-bottom:5px;">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:{dot};'
            f'flex-shrink:0;box-shadow:0 0 6px {dot};"></div>'
            f'<div style="flex:1;">'
            f'<div style="font-family:Syne,sans-serif;font-size:12px;font-weight:600;color:{tc};">{name}</div>'
            f'<div style="font-family:DM Mono,monospace;font-size:9px;color:#3a2d5a;margin-top:1px;">{sub}</div>'
            f'</div>' + badge + '</div>'
        )
    return C(H("🤚","Gesture Recognition") + rows)

def chart_card(vol_pct):
    # SVG line chart — Distance (X) vs Volume % (Y)
    SVG_W, SVG_H = 320, 160
    pl, pr, pt, pb = 36, 10, 12, 36
    pw = SVG_W - pl - pr
    ph = SVG_H - pt - pb

    pts = " ".join([
        f"{pl + int((i/39)*pw)},{pt + ph - int((i/39)*ph)}"
        for i in range(40)
    ])
    fill_pts = f"{pl},{pt+ph} " + pts + f" {pl+pw},{pt+ph}"

    dot_x = pl + int((vol_pct/100)*pw)
    dot_y = pt + ph - int((vol_pct/100)*ph)

    y_labels = "".join([
        f'<text x="{pl-5}" y="{pt + ph - int((v/100)*ph) + 3}" '
        f'text-anchor="end" font-family="DM Mono,monospace" '
        f'font-size="9" fill="#3a2d5a">{v}</text>'
        for v in [0, 20, 40, 60, 80, 100]
    ])

    grids = "".join([
        f'<line x1="{pl}" y1="{pt + ph - int((v/100)*ph)}" '
        f'x2="{pl+pw}" y2="{pt + ph - int((v/100)*ph)}" '
        f'stroke="rgba(139,92,246,0.12)" stroke-width="1" stroke-dasharray="3,3"/>'
        for v in [20, 40, 60, 80, 100]
    ])

    x_labels = "".join([
        f'<text x="{pl + int((v/60)*pw)}" y="{pt+ph+16}" '
        f'text-anchor="middle" font-family="DM Mono,monospace" '
        f'font-size="9" fill="#3a2d5a">{v}mm</text>'
        for v in [0, 15, 30, 45, 60]
    ])

    red_dash = (
        f'<line x1="{pl}" y1="{pt}" x2="{pl+pw}" y2="{pt}" '
        f'stroke="#e53e3e" stroke-width="1" stroke-dasharray="4,4" opacity="0.5"/>'
    )

    svg = (
        f'<svg width="100%" viewBox="0 0 {SVG_W} {SVG_H}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;overflow:visible;">'
        f'<defs>'
        f'<linearGradient id="lg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="#7c3aed" stop-opacity="0.3"/>'
        f'<stop offset="100%" stop-color="#7c3aed" stop-opacity="0.02"/>'
        f'</linearGradient></defs>'
        + red_dash + grids
        + f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pt+ph}" stroke="#3a2d5a" stroke-width="1"/>'
        + f'<line x1="{pl}" y1="{pt+ph}" x2="{pl+pw}" y2="{pt+ph}" stroke="#3a2d5a" stroke-width="1"/>'
        + f'<polygon points="{fill_pts}" fill="url(#lg)"/>'
        + f'<polyline points="{pts}" fill="none" stroke="#7c3aed" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        + f'<circle cx="{dot_x}" cy="{dot_y}" r="5" fill="#e53e3e" stroke="#0d0b16" stroke-width="2"/>'
        + y_labels + x_labels
        + f'<text x="{pl+pw//2}" y="{SVG_H-2}" text-anchor="middle" font-family="Syne,sans-serif" font-size="9" fill="#5a4d80" font-weight="600">\u2190 Distance (mm) \u2192</text>'
        + f'<text x="10" y="{pt+ph//2}" text-anchor="middle" font-family="Syne,sans-serif" font-size="8" fill="#5a4d80" font-weight="600" transform="rotate(-90,10,{pt+ph//2})">Vol%</text>'
        + f'</svg>'
    )

    legend = (
        f'<div style="display:flex;gap:14px;margin-top:4px;padding-left:36px;">'
        f'<span style="font-family:DM Mono,monospace;font-size:10px;color:#7c3aed;">'
        f'\u2014 Volume Curve</span>'
        f'<span style="font-family:DM Mono,monospace;font-size:10px;color:#e53e3e;">'
        f'\u25cf Now: {vol_pct}%</span>'
        f'</div>'
    )

    return C(
        H("📊","Distance → Volume Map") +
        f'<div style="font-family:Syne,sans-serif;font-size:10px;font-weight:600;'
        f'color:#5a4d80;margin-bottom:6px;">Gesture Distance \u2192 System Volume</div>'
        + svg + legend
    )

def metrics_card(fps,hands,dist_mm_v,latency):
    def box(val,lbl,col):
        return (
            f'<div style="background:rgba(255,255,255,.03);'
            f'border:1px solid rgba(139,92,246,.12);border-radius:8px;'
            f'padding:8px 4px;text-align:center;flex:1;">'
            f'<div style="font-family:DM Mono,monospace;font-size:19px;'
            f'font-weight:500;color:{col};line-height:1;">{val}</div>'
            f'<div style="font-family:Syne,sans-serif;font-size:8px;color:#3a2d5a;'
            f'margin-top:3px;text-transform:uppercase;letter-spacing:.07em;">{lbl}</div>'
            f'</div>'
        )
    return C(
        H("📈","Live Metrics") +
        f'<div style="display:flex;gap:6px;margin-bottom:6px;">'
        + box(fps,"FPS","#a78bfa") + box(hands,"Hands","#22c55e")
        + f'</div>'
        f'<div style="display:flex;gap:6px;">'
        + box(dist_mm_v,"Dist mm","#c084fc") + box(latency,"Latency","#f59e0b")
        + f'</div>'
    )

def stats_card(cam,res,hand_t,angle,gesture_v):
    def row(lbl,val,col="#6b5b9a"):
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:5px 0;border-bottom:1px solid rgba(139,92,246,.07);">'
            f'<span style="font-family:Syne,sans-serif;font-size:10px;color:#4a3a6e;">{lbl}</span>'
            f'<span style="font-family:DM Mono,monospace;font-size:10px;color:{col};">{val}</span>'
            f'</div>'
        )
    return C(
        H("👁","Detection Stats") +
        row("Camera",cam,"#22c55e" if cam=="Active ✓" else "#4a3a6e") +
        row("Model","MediaPipe Lite","#6b5b9a") +
        row("Landmarks","21 / hand","#6b5b9a") +
        row("Connections","20 / hand","#6b5b9a") +
        row("Resolution",res,"#6b5b9a") +
        row("Hand Type",hand_t,"#a78bfa") +
        row("Hand Angle",angle,"#c084fc") +
        row("Gesture",gesture_v,"#e9d5ff")
    )

def history_card(hist):
    N=len(hist); CH=52
    bars=""
    for i,v in enumerate(hist):
        bh=max(int((v/100)*CH),1)
        bg="#c084fc" if i==N-1 else "rgba(139,92,246,.38)"
        bars+=(
            f'<div style="flex:1;display:flex;align-items:flex-end;height:{CH}px;">'
            f'<div style="width:100%;height:{bh}px;background:{bg};'
            f'border-radius:1px 1px 0 0;min-height:1px;"></div></div>'
        )
    return C(
        H("📉","Volume History") +
        f'<div style="display:flex;gap:1px;height:{CH}px;align-items:flex-end;'
        f'border-bottom:1px solid rgba(139,92,246,.18);margin-bottom:4px;">{bars}</div>'
        f'<div style="display:flex;justify-content:space-between;">'
        f'<span style="font-family:DM Mono,monospace;font-size:8px;color:#3a2d5a;">oldest</span>'
        f'<span style="font-family:DM Mono,monospace;font-size:8px;color:#5a4d80;">last {N} frames</span>'
        f'<span style="font-family:DM Mono,monospace;font-size:8px;color:#c084fc;">now</span>'
        f'</div>'
    )

# ══════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════
aud_ok  = AUDIO_OK
aud_lbl = "pycaw ✓" if aud_ok else "Simulation"
aud_col = "#22c55e" if aud_ok else "#f59e0b"

st.markdown(
    f'<div style="background:rgba(255,255,255,.02);'
    f'border-bottom:1px solid rgba(139,92,246,.14);'
    f'padding:10px 16px 10px 16px;'
    f'display:flex;align-items:center;justify-content:space-between;'
    f'margin-bottom:10px;">'

    f'<div style="display:flex;align-items:center;gap:10px;">'
    f'<div style="width:30px;height:30px;'
    f'background:linear-gradient(135deg,#7c3aed,#c084fc);'
    f'border-radius:8px;display:flex;align-items:center;'
    f'justify-content:center;font-size:15px;flex-shrink:0;">✋</div>'
    f'<span style="font-family:Syne,sans-serif;font-size:18px;'
    f'font-weight:800;color:#e9d5ff;letter-spacing:-.02em;">GestureVol</span>'
    f'</div>'

    f'<span style="font-family:DM Mono,monospace;font-size:10px;color:{aud_col};'
    f'background:rgba(139,92,246,.08);padding:5px 12px;border-radius:6px;'
    f'border:1px solid rgba(139,92,246,.2);">Audio: {aud_lbl}</span>'
    f'</div>',
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════
#  CONTROLS ROW — full width above columns
# ══════════════════════════════════════════════════════════
ct1,ct2,ct3,ct4,ct5 = st.columns([1,1,1.4,1.4,1])
with ct1: btn_start = st.button("▶  START", key="start")
with ct2: btn_stop  = st.button("⏹  STOP",  key="stop")
with ct3: det_conf  = st.slider("Detection Confidence",  0.0,1.0,0.65,key="det")
with ct4: trk_conf  = st.slider("Tracking Confidence",   0.0,1.0,0.60,key="trk")
with ct5: max_hands = st.slider("Max Hands", 1,2,1, key="mh")

st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  3 COLUMNS
# ══════════════════════════════════════════════════════════
col_l, col_m, col_r = st.columns([2.8, 1.8, 1.8])

# ── LEFT: camera feed + status ────────────────────────────
with col_l:
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:11px;font-weight:700;'
        'color:#7c5cbf;text-transform:uppercase;letter-spacing:.1em;'
        'padding:0 0 5px 0;border-bottom:1px solid rgba(139,92,246,.15);'
        'margin-bottom:6px;">📷&nbsp; Live Camera Feed</div>',
        unsafe_allow_html=True)

    cam_ph = st.empty()
    cam_ph.markdown(
        '<div style="background:#070510;border-radius:10px;height:920px;'
        'display:flex;align-items:center;justify-content:center;'
        'border:1px solid rgba(139,92,246,.1);">'
        '<span style="font-family:Syne,sans-serif;font-size:13px;'
        'color:#2a1f45;">✋&nbsp; Press START to begin</span></div>',
        unsafe_allow_html=True)

    st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

    sb1,sb2,sb3,sb4 = st.columns(4)
    cam_stat  = sb1.empty(); hand_stat = sb2.empty()
    fps_stat  = sb3.empty(); mod_stat  = sb4.empty()
    cam_stat.warning("Camera Off")
    hand_stat.info("Hands: 0")
    fps_stat.info("FPS: —")
    mod_stat.info("Model Ready")

# ── MIDDLE ────────────────────────────────────────────────
with col_m:
    vol_ph   = st.empty()
    dist_ph  = st.empty()
    gest_ph  = st.empty()
    chart_ph = st.empty()

    vol_ph.markdown(vol_card(0),           unsafe_allow_html=True)
    dist_ph.markdown(dist_card(0,0),       unsafe_allow_html=True)
    gest_ph.markdown(gest_card("None"),    unsafe_allow_html=True)
    chart_ph.markdown(chart_card(0),       unsafe_allow_html=True)

# ── RIGHT ─────────────────────────────────────────────────
with col_r:
    met_ph   = st.empty()
    stats_ph = st.empty()
    hist_ph  = st.empty()

    met_ph.markdown(metrics_card("—","—","—","—"),    unsafe_allow_html=True)
    stats_ph.markdown(stats_card("Inactive","—","—","—","—"), unsafe_allow_html=True)
    hist_ph.markdown(history_card([0]*40),            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  CAMERA LOOP
# ══════════════════════════════════════════════════════════
if btn_start and not btn_stop:
    cap = open_cam()
    hm  = mp_hands.Hands(
        static_image_mode=False, max_num_hands=max_hands,
        model_complexity=0, min_detection_confidence=det_conf,
        min_tracking_confidence=trk_conf)

    prev_t=0; smooth=0; vol_hist=[0]*40
    cam_stat.success("Camera Active"); mod_stat.success("Model Loaded")

    while True:
        ok,frame=cap.read()
        if not ok: break
        frame=cv2.flip(frame,1); h,w,_=frame.shape

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        rgb.flags.writeable=False; res=hm.process(rgb); rgb.flags.writeable=True

        px=0; mm=0.0; gest="None"; tp=ip=None
        nh=0; htype="—"; hconf=0.0; angle=0

        if res.multi_hand_landmarks:
            nh=len(res.multi_hand_landmarks)
            for lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame,lm,mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(150,55,250),thickness=2,circle_radius=3),
                    mp_draw.DrawingSpec(color=(190,90,250),thickness=2))
                px,tp,ip=fdist(lm,w,h)
                mm=px*PX_TO_MM; gest=classify(mm)
                wr=lm.landmark[0]; mc=lm.landmark[9]
                angle=int(math.degrees(math.atan2(mc.y-wr.y,mc.x-wr.x)))
            if res.multi_handedness:
                htype=res.multi_handedness[0].classification[0].label
                hconf=res.multi_handedness[0].classification[0].score

        raw=sys_vol(px) if px>0 else 0
        # Only smooth when hand is detected; snap to 0 when no hand
        if px > 0:
            smooth = int(0.6*smooth + 0.4*raw)
            smooth = min(smooth, 100)  # clamp to 100
        else:
            smooth = int(smooth * 0.85)  # decay when no hand
        vol_hist=vol_hist[1:]+[smooth]

        curr_t=time.time(); fps=1/(curr_t-prev_t) if prev_t else 0; prev_t=curr_t
        lat=round(1000/fps,1) if fps else 0

        QMAP={"Open Hand":"DETECTED","Pinch":"GOOD DETECT","Closed":"DETECTED","None":"NO HAND"}
        quality=QMAP.get(gest,"NO HAND")

        frame=draw_frame(frame,gest,quality,tp,ip,mm,smooth,fps,nh)
        # Resize to fill full camera column height matching middle column
        frame=cv2.resize(frame,(640,920),interpolation=cv2.INTER_LINEAR)
        cam_ph.image(frame,channels="BGR",width="stretch")

        # status row
        cam_stat.success("Camera Active")
        if nh > 0:
            hand_stat.success(f"Hands: {nh}")
        else:
            hand_stat.warning("Hands: 0")
        fps_stat.info(f"FPS: {int(fps)}")
        mod_stat.success("Model Loaded")

        # middle
        vol_ph.markdown(vol_card(smooth),                    unsafe_allow_html=True)
        dist_ph.markdown(dist_card(int(px),int(round(mm))),  unsafe_allow_html=True)
        gest_ph.markdown(gest_card(gest),                     unsafe_allow_html=True)
        chart_ph.markdown(chart_card(smooth),                 unsafe_allow_html=True)

        # right
        met_ph.markdown(
            metrics_card(str(int(fps)),str(nh),f"{int(mm)}mm",f"{lat}ms"),
            unsafe_allow_html=True)
        hconf_str=f"{htype} ({hconf:.2f})" if htype!="—" else "—"
        stats_ph.markdown(
            stats_card(
                "Active ✓", f"{w}×{h}", hconf_str,
                f"{angle}°", gest if gest!="None" else "—"),
            unsafe_allow_html=True)
        hist_ph.markdown(history_card(vol_hist), unsafe_allow_html=True)

        if btn_stop: break

    cap.release(); hm.close()
    cam_stat.warning("Camera Off")