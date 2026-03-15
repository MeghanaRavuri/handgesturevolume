import streamlit as st
import cv2, mediapipe as mp, numpy as np, math, matplotlib.pyplot as plt, io, base64

st.set_page_config(layout="wide", page_title="Milestone 3 – Volume Mapping")

# ── Audio ─────────────────────────────────────────────────────────────────────
AUDIO_OK = False; vol_ctrl = None; minVol, maxVol = -65.0, 0.0
try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    dev = AudioUtilities.GetSpeakers()
    iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    vol_ctrl = iface.QueryInterface(IAudioEndpointVolume)
    vr = vol_ctrl.GetVolumeRange(); minVol, maxVol = vr[0], vr[1]; AUDIO_OK = True
except: pass

def map_vol(d):
    return int(np.clip((np.clip(d,20,200)-20)/180*100, 0, 100))

def set_vol(dist_px):
    v = map_vol(dist_px)
    if AUDIO_OK and vol_ctrl:
        try: vol_ctrl.SetMasterVolumeLevel(float(np.interp(dist_px,[20,200],[minVol,maxVol])),None)
        except: pass
    return v

# ── Graph — wide, full right column width ────────────────────────────────────
def make_graph(live_d=None, live_v=None):
    xs = np.linspace(20, 200, 120)
    ys = [map_vol(x) for x in xs]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#ffffff'); ax.set_facecolor('#f9f7ff')
    ax.fill_between(xs, ys, alpha=0.12, color='#7c3aed')
    ax.plot(xs, ys, color='#7c3aed', lw=2.5, label='Volume Curve')
    if live_d and live_v is not None:
        ax.scatter([live_d],[live_v],color='#ef4444',s=70,zorder=7,label=f'Now: {live_v}%')
        ax.axvline(live_d,color='#ef4444',lw=1,ls='--',alpha=0.4)
        ax.axhline(live_v,color='#ef4444',lw=1,ls='--',alpha=0.4)
    ax.set_xlabel("Pinch Distance (px)", fontsize=10, color='#6b7280')
    ax.set_ylabel("Volume (%)", fontsize=10, color='#6b7280')
    ax.set_title("Gesture Distance → System Volume", fontsize=11, fontweight='bold', color='#1a103c', pad=8)
    ax.set_xlim(20,200); ax.set_ylim(0,108)
    ax.tick_params(labelsize=9, colors='#9ca3af')
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#e5e7eb')
    ax.grid(True, alpha=0.25, color='#e0d8ff')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.85)
    plt.tight_layout(pad=0.8)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap');
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],.main{
    background:#f0eeff!important;overflow:hidden!important;height:100%!important;margin:0!important;}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer{display:none!important;}
.block-container{padding:8px 18px 0!important;max-width:100%!important;overflow:hidden!important;}
[data-testid="stVerticalBlock"]{gap:0!important;}
div[data-testid="stVerticalBlock"]>div{gap:0!important;}
.element-container,.stMarkdown{margin:0!important;padding:0!important;}
div[data-testid="column"]{padding:0 6px!important;}
[data-testid="stImage"]{line-height:0!important;display:block!important;}
[data-testid="stImage"] img{
    width:100%!important;height:560px!important;
    object-fit:cover!important;border-radius:14px!important;display:block!important;}
.stButton>button{
    font-family:Outfit,sans-serif!important;font-weight:600!important;font-size:13px!important;
    padding:8px 22px!important;border-radius:22px!important;border:none!important;
    background:linear-gradient(135deg,#7c3aed,#6d28d9)!important;color:#fff!important;
    box-shadow:0 3px 14px rgba(109,40,217,.35)!important;}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
hl, hr = st.columns([3,2])
with hl:
    st.markdown('<p style="font-family:Outfit,sans-serif;font-size:17px;font-weight:700;'
                'color:#1a103c;margin:0;padding:5px 0 3px;">🎙 Milestone 3 – Volume Mapping &amp; Control</p>',
                unsafe_allow_html=True)
with hr:
    c1,c2,c3 = st.columns(3)
    start = c1.button("▶ Start"); pause = c2.button("⏸ Pause"); c3.button("⚙ Settings")
st.markdown('<hr style="border:none;border-top:1.5px solid #d8d0f8;margin:3px 0 8px;">', unsafe_allow_html=True)

# ── TWO COLUMNS: Left=camera only | Right=vol+dist+graph stacked ──────────────
left, right = st.columns([1, 1])

with left:
    cam_ph = st.empty()

with right:
    vol_ph   = st.empty()
    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    dist_ph  = st.empty()
    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    graph_ph = st.empty()

# ── Card styles ───────────────────────────────────────────────────────────────
CS = "background:#fff;border-radius:14px;padding:16px 20px;box-shadow:0 2px 12px rgba(109,40,217,.09);"
HS = "font-family:Outfit,sans-serif;font-size:10px;font-weight:700;color:#9b8ec4;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px;"

def vol_card(v):
    bw=max(min(v,100),0); kn=max(min(bw,96),4)
    vc='#14b87a' if v>60 else '#7c3aed' if v>30 else '#ef4444'
    return f"""<div style="{CS}">
      <div style="{HS}">🔊 Volume Level</div>
      <div style="display:flex;align-items:baseline;gap:5px;margin-bottom:12px;">
        <span style="font-family:Outfit,sans-serif;font-size:48px;font-weight:800;color:{vc};line-height:1;">{v}</span>
        <span style="font-family:Outfit,sans-serif;font-size:15px;color:#9b8ec4;">%</span>
      </div>
      <div style="position:relative;height:9px;background:#ede9fe;border-radius:99px;">
        <div style="position:absolute;left:0;top:0;height:100%;width:{bw}%;
                    background:linear-gradient(90deg,#7c3aed,#a78bfa);border-radius:99px;"></div>
        <div style="position:absolute;top:50%;left:{kn}%;transform:translate(-50%,-50%);
                    width:17px;height:17px;background:#fff;border:3px solid #7c3aed;
                    border-radius:50%;box-shadow:0 2px 6px rgba(109,40,217,.4);"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-family:Outfit,sans-serif;
                  font-size:9px;color:#c4b5fd;margin-top:5px;">
        <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
      </div>
    </div>"""

def dist_card(px, mm):
    return f"""<div style="{CS}">
      <div style="{HS}">📐 Distance</div>
      <div style="display:flex;gap:28px;align-items:flex-end;">
        <div>
          <div style="font-family:Outfit,sans-serif;font-size:48px;font-weight:800;color:#7c3aed;line-height:1;">{px}</div>
          <div style="font-family:Outfit,sans-serif;font-size:11px;color:#9b8ec4;margin-top:4px;">pixels</div>
        </div>
        <div style="width:1.5px;height:54px;background:#ede9fe;margin-bottom:6px;"></div>
        <div>
          <div style="font-family:Outfit,sans-serif;font-size:48px;font-weight:800;color:#a78bfa;line-height:1;">{int(round(mm))}</div>
          <div style="font-family:Outfit,sans-serif;font-size:11px;color:#9b8ec4;margin-top:4px;">mm</div>
        </div>
      </div>
    </div>"""

def graph_card(b64):
    return f"""<div style="{CS}padding:16px 20px 12px;">
      <div style="{HS}">📈 Distance vs Volume</div>
      <img src="data:image/png;base64,{b64}" style="width:100%;border-radius:10px;display:block;"/>
    </div>"""

# ── Initial render ────────────────────────────────────────────────────────────
cam_ph.markdown(
    '<div style="background:#1a103c;border-radius:14px;height:560px;'
    'display:flex;align-items:center;justify-content:center;'
    'color:#7c5cbf;font-family:Outfit,sans-serif;font-size:15px;">'
    '📷 Press ▶ Start</div>', unsafe_allow_html=True)

vol_ph.markdown(vol_card(0),       unsafe_allow_html=True)
dist_ph.markdown(dist_card(0,0.0), unsafe_allow_html=True)
init_g = make_graph()
graph_ph.markdown(graph_card(init_g), unsafe_allow_html=True)

# ── Camera loop ───────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
PX2MM    = 0.2646

if start:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

    with mp_hands.Hands(max_num_hands=1, model_complexity=0,
                        min_detection_confidence=0.65,
                        min_tracking_confidence=0.6) as hands:
        fidx=0; cached_g=init_g
        prev_vol=-1; prev_px=-1

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            res   = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            dist_px=0; dist_mm=0.0; vol_pct=0; gesture="Open Hand"

            if res.multi_hand_landmarks:
                for lms in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(160,60,255),thickness=2,circle_radius=4),
                        mp_draw.DrawingSpec(color=(200,160,255),thickness=2))
                    h,w,_=frame.shape
                    th=lms.landmark[4]; ix=lms.landmark[8]
                    x1,y1=int(th.x*w),int(th.y*h)
                    x2,y2=int(ix.x*w),int(ix.y*h)
                    dist_px=int(math.hypot(x2-x1,y2-y1))
                    dist_mm=dist_px*PX2MM
                    vol_pct=set_vol(dist_px)

                    if dist_mm>40:   gesture="Open Hand"; bc=(30,160,60)
                    elif dist_mm>10: gesture="Pinch";     bc=(20,120,200)
                    else:            gesture="Closed";    bc=(40,40,200)

                    cv2.circle(frame,(x1,y1),10,(255,0,180),-1)
                    cv2.circle(frame,(x2,y2),10,(255,0,180),-1)
                    cv2.line(frame,(x1,y1),(x2,y2),(200,80,255),3)
                    mx,my=(x1+x2)//2,(y1+y2)//2
                    lbl=f"{int(round(dist_mm))}mm"
                    (lw,lh),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
                    cv2.rectangle(frame,(mx-lw//2-8,my-lh-12),(mx+lw//2+8,my+4),(30,10,60),-1)
                    cv2.putText(frame,lbl,(mx-lw//2,my-3),cv2.FONT_HERSHEY_SIMPLEX,0.65,(220,160,255),2)
                    vtxt=f"Vol: {vol_pct}%"
                    (vw,vh),_=cv2.getTextSize(vtxt,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
                    cv2.rectangle(frame,(12,48),(12+vw+14,48+vh+10),(20,5,50),-1)
                    cv2.putText(frame,vtxt,(18,48+vh+2),cv2.FONT_HERSHEY_SIMPLEX,0.7,(180,255,160),2)
                    (bw2,bh2),_=cv2.getTextSize(gesture,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
                    cv2.rectangle(frame,(12,12),(12+bw2+16,12+bh2+10),bc,-1)
                    cv2.putText(frame,gesture,(18,12+bh2+2),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            fidx += 1

            # Camera: every frame
            cam_ph.image(frame, channels="BGR", width="stretch")

            # Cards: only on change
            if vol_pct != prev_vol:
                vol_ph.markdown(vol_card(vol_pct), unsafe_allow_html=True)
                prev_vol = vol_pct
            if dist_px != prev_px:
                dist_ph.markdown(dist_card(dist_px, dist_mm), unsafe_allow_html=True)
                prev_px = dist_px

            # Graph: every 15 frames
            if fidx % 15 == 0:
                cached_g = make_graph(dist_px if dist_px>0 else None,
                                      vol_pct if dist_px>0 else None)
                graph_ph.markdown(graph_card(cached_g), unsafe_allow_html=True)

            if pause: break
    cap.release()