"""Face and facial-feature detection with OpenCV Haar cascades.

Faces are found on the whole frame; eyes are searched only in the upper half of each face and the
smile only in the lower half, which keeps the cascades fast and cuts false detections.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, width, height in frame pixels


@dataclass
class Face:
    box: Box
    eyes: List[Box] = field(default_factory=list)
    smile: Optional[Box] = None


def _as_boxes(found) -> List[Box]:
    """detectMultiScale returns an empty tuple or an N x 4 array; always return a list of int tuples."""
    if found is None or len(found) == 0:
        return []
    return [tuple(int(v) for v in b) for b in found]


class FaceDetector:
    """Finds frontal faces and, optionally, the eyes and smile inside each face."""

    def __init__(self, min_face: int = 60):
        base = cv2.data.haarcascades
        self._face = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self._eye = cv2.CascadeClassifier(base + "haarcascade_eye_tree_eyeglasses.xml")
        self._smile = cv2.CascadeClassifier(base + "haarcascade_smile.xml")
        if self._face.empty() or self._eye.empty() or self._smile.empty():
            raise RuntimeError("OpenCV Haar cascade files are missing from this OpenCV install")
        self.min_face = min_face

    def detect(self, frame: np.ndarray, features: bool = True) -> List[Face]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes = _as_boxes(self._face.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=7, minSize=(self.min_face, self.min_face)))
        faces = [Face(box=b) for b in boxes]
        if features:
            for face in faces:
                self._add_features(gray, face)
        return faces

    def _add_features(self, gray: np.ndarray, face: Face) -> None:
        x, y, w, h = face.box
        upper = gray[y:y + h // 2, x:x + w]
        eye_min = max(8, w // 9)
        eyes = _as_boxes(self._eye.detectMultiScale(
            upper, scaleFactor=1.1, minNeighbors=5, minSize=(eye_min, eye_min)))
        # keep the two largest candidates, left to right
        eyes = sorted(sorted(eyes, key=lambda b: b[2] * b[3], reverse=True)[:2], key=lambda b: b[0])
        face.eyes = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes]

        top = y + h // 2
        lower = gray[top:y + h, x:x + w]
        smiles = _as_boxes(self._smile.detectMultiScale(
            lower, scaleFactor=1.5, minNeighbors=15, minSize=(max(10, w // 4), max(6, h // 10))))
        if smiles:
            sx, sy, sw, sh = max(smiles, key=lambda b: b[2] * b[3])
            face.smile = (x + sx, top + sy, sw, sh)


def crop(frame: np.ndarray, box: Box, margin: float = 0.15) -> np.ndarray:
    """Face crop with a small margin, clipped to the frame."""
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    H, W = frame.shape[:2]
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    return frame[y0:y1, x0:x1]
