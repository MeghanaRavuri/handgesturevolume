import pytest
import math
import numpy as np

def classify(mm):
    return "Open Hand" if mm > 40 else ("Pinch" if mm > 10 else "Closed")

def sys_vol(px):
    return int(np.interp(px, [20, 300], [0, 100]))

def px_to_mm(px):
    return px * 0.2646

def smooth_vol(current, raw):
    return min(int(0.6 * current + 0.4 * raw), 100)

def get_quality(gesture):
    q = {"Open Hand":"DETECTED","Pinch":"GOOD DETECT","Closed":"DETECTED","None":"NO HAND"}
    return q.get(gesture, "NO HAND")

class TestM1:
    def test_zero_pixels(self):
        assert px_to_mm(0) == 0.0
    def test_100_pixels(self):
        assert round(px_to_mm(100), 2) == 26.46
    def test_positive(self):
        assert px_to_mm(50) > 0

class TestM2:
    def test_open_hand(self):
        assert classify(41) == "Open Hand"
    def test_pinch(self):
        assert classify(25) == "Pinch"
    def test_closed(self):
        assert classify(5) == "Closed"
    def test_distance(self):
        assert math.hypot(3, 4) == 5.0

class TestM3:
    def test_min_vol(self):
        assert sys_vol(20) == 0
    def test_max_vol(self):
        assert sys_vol(300) == 100
    def test_smooth_reaches_100(self):
        s = 0
        for _ in range(200):
            s = smooth_vol(s, 100)
        assert s == 100

class TestM4:
    def test_pinch_feedback(self):
        assert get_quality("Pinch") == "GOOD DETECT"
    def test_open_feedback(self):
        assert get_quality("Open Hand") == "DETECTED"
    def test_no_hand(self):
        assert get_quality("None") == "NO HAND"

class TestIntegration:
    def test_pipeline(self):
        assert classify(px_to_mm(250)) == "Open Hand"
        assert sys_vol(250) >= 75
    def test_all_volumes_valid(self):
        for px in range(0, 400, 10):
            assert 0 <= sys_vol(px) <= 100
