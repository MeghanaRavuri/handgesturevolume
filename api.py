"""
api.py — FastAPI Backend for GestureVol
Run:  uvicorn api:app --reload
Test: Postman or browser at http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import math

app = FastAPI(
    title="GestureVol API",
    description="Hand Gesture Volume Control System — All 4 Milestones",
    version="1.0.0"
)

PX_TO_MM = 0.2646

# ─────────────────────────────────────────────
#  REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class DistanceInput(BaseModel):
    pixels: float

class GestureInput(BaseModel):
    distance_mm: float

class VolumeInput(BaseModel):
    distance_mm: float

class FeedbackInput(BaseModel):
    gesture: str


# ─────────────────────────────────────────────
#  CORE LOGIC
# ─────────────────────────────────────────────

def px_to_mm(px: float) -> float:
    return round(px * PX_TO_MM, 2)

def classify_gesture(mm: float) -> str:
    if mm > 40:   return "Open Hand"
    elif mm > 10: return "Pinch"
    else:         return "Closed"

def calculate_volume(mm: float) -> int:
    pct = round(np.interp(mm, [5, 60], [0, 100]))
    return max(0, min(int(pct), 100))

def quality_feedback(gesture: str) -> str:
    q = {
        "Open Hand": "DETECTED",
        "Pinch":     "GOOD DETECT",
        "Closed":    "DETECTED",
        "None":      "NO HAND"
    }
    return q.get(gesture, "NO HAND")


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────

# M1 — Health check / system status
@app.get("/")
def root():
    return {
        "system": "GestureVol",
        "status": "running",
        "milestones": ["M1-Detection", "M2-Classification", "M3-Volume", "M4-UI"]
    }

@app.get("/status")
def status():
    """M1 — Check if the system is active"""
    return {
        "camera": "ready",
        "model": "MediaPipe Lite",
        "landmarks_per_hand": 21,
        "connections_per_hand": 20,
        "px_to_mm_ratio": PX_TO_MM
    }

# M2 — Pixel to mm conversion
@app.post("/convert")
def convert_pixels(data: DistanceInput):
    """M2 — Convert pixel distance to millimeters"""
    mm = px_to_mm(data.pixels)
    return {
        "pixels": data.pixels,
        "millimeters": mm
    }

# M2 — Gesture classification
@app.post("/classify")
def classify(data: GestureInput):
    """M2 — Classify gesture from finger distance in mm"""
    gesture = classify_gesture(data.distance_mm)
    return {
        "distance_mm": data.distance_mm,
        "gesture": gesture,
        "rule": "Open Hand > 40mm | Pinch 10-40mm | Closed < 10mm"
    }

# M3 — Volume mapping
@app.post("/volume")
def volume(data: VolumeInput):
    """M3 — Map finger distance to volume percentage"""
    vol = calculate_volume(data.distance_mm)
    gesture = classify_gesture(data.distance_mm)
    return {
        "distance_mm": data.distance_mm,
        "gesture": gesture,
        "volume_percent": vol,
        "mapping": "5mm=0%  →  60mm=100%"
    }

# M4 — Quality feedback
@app.post("/feedback")
def feedback(data: FeedbackInput):
    """M4 — Get quality feedback label for a gesture"""
    quality = quality_feedback(data.gesture)
    return {
        "gesture": data.gesture,
        "quality_label": quality
    }

# Full pipeline — all milestones at once
@app.post("/pipeline")
def pipeline(data: DistanceInput):
    """All Milestones — Full pipeline from pixel distance to volume"""
    mm      = px_to_mm(data.pixels)
    gesture = classify_gesture(mm)
    volume  = calculate_volume(mm)
    quality = quality_feedback(gesture)
    return {
        "input_pixels":    data.pixels,
        "distance_mm":     mm,
        "gesture":         gesture,
        "volume_percent":  volume,
        "quality_label":   quality
    }