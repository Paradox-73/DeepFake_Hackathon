"""Optional, pluggable face detection. detector: none | haar | dnn | 'file.py:factory'."""
import os

import cv2
import numpy as np

from .utils import load_callable


class HaarDetector:
    def __init__(self, cfg):
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.c = cv2.CascadeClassifier(path)
        if self.c.empty():
            raise RuntimeError("Could not load Haar cascade " + path)

    def detect(self, bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.c.detectMultiScale(g, 1.1, 5, minSize=(48, 48))
        if len(boxes) == 0:
            return None
        return tuple(int(v) for v in max(boxes, key=lambda b: b[2] * b[3]))


class DnnDetector:
    """OpenCV res10 SSD. Needs cfg.proto (deploy.prototxt) and cfg.model (.caffemodel) paths."""

    def __init__(self, cfg):
        self.net = cv2.dnn.readNetFromCaffe(cfg["proto"], cfg["model"])
        self.conf = cfg.get("confidence", 0.5)

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        det = self.net.forward()[0, 0]
        best = None
        for d in det:
            if d[2] >= self.conf and (best is None or d[2] > best[2]):
                best = d
        if best is None:
            return None
        x1, y1, x2, y2 = best[3] * w, best[4] * h, best[5] * w, best[6] * h
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def get_detector(cfg):
    """cfg: dict with key 'detector'. Returns object with .detect(bgr)->(x,y,w,h)|None, or None."""
    name = (cfg or {}).get("detector", "none")
    if name in (None, "none", False):
        return None
    if name == "haar":
        return HaarDetector(cfg)
    if name == "dnn":
        return DnnDetector(cfg)
    return load_callable(name)(cfg)  # custom plug-in factory


def crop_face(bgr, box, margin=0.3):
    """Expand box by margin, make it square, clip to image."""
    h, w = bgr.shape[:2]
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    s = max(bw, bh) * (1 + margin)
    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    x1, y1 = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return bgr
    return bgr[y0:y1, x0:x1]
