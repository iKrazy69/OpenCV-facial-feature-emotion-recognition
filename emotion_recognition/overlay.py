"""Everything that is drawn on the video: face brackets, emotion labels, probability bars and the HUD."""
from typing import Optional, Sequence

import cv2
import numpy as np

from . import EMOTION_COLORS, EMOTIONS
from .emotions import Probs, dominant
from .faces import Face

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE = (245, 245, 245)
DARK = (28, 24, 22)
MUTED = (150, 150, 150)
ACCENT = (80, 200, 255)


def _blend_rect(img: np.ndarray, p0, p1, color, alpha: float) -> None:
    """Semi-transparent filled rectangle, clipped to the image."""
    H, W = img.shape[:2]
    x0, y0 = max(0, int(p0[0])), max(0, int(p0[1]))
    x1, y1 = min(W, int(p1[0])), min(H, int(p1[1]))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _corner_box(img, box, color, thickness=2) -> None:
    """Corner brackets instead of a full rectangle: reads as 'tracking' and hides less of the face."""
    x, y, w, h = box
    k = max(10, min(w, h) // 4)
    for (cx, cy, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * k, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * k), color, thickness, cv2.LINE_AA)


def _scale(frame_w: int) -> float:
    return max(0.45, min(1.0, frame_w / 1280))


def draw_face(img: np.ndarray, face: Face, track_id: int, probs: Optional[Probs],
              show_features: bool = True, show_bars: bool = True) -> None:
    s = _scale(img.shape[1])
    x, y, w, h = face.box
    if probs is None:
        color, title = MUTED, f"#{track_id}  reading..."
    else:
        label, conf = dominant(probs)
        color, title = EMOTION_COLORS[label], f"#{track_id}  {label.upper()}  {conf * 100:.0f}%"
    _corner_box(img, face.box, color, max(2, int(3 * s)))

    if show_features:
        for (ex, ey, ew, eh) in face.eyes:
            cv2.circle(img, (ex + ew // 2, ey + eh // 2), max(3, min(ew, eh) // 3), (255, 230, 120), 1, cv2.LINE_AA)
        if face.smile is not None:
            sx, sy, sw, sh = face.smile
            cv2.ellipse(img, (sx + sw // 2, sy + sh // 2), (sw // 2, max(3, sh // 3)), 0, 0, 180,
                        (120, 230, 255), 2, cv2.LINE_AA)

    # label chip above the face
    fs = 0.6 * s
    (tw, th), base = cv2.getTextSize(title, FONT, fs, 1)
    pad = int(8 * s)
    hud_bottom = int(50 * s)  # keep the chip clear of the title bar
    cy = y - pad if y - th - 2 * pad > hud_bottom else y + h + th + 2 * pad
    _blend_rect(img, (x, cy - th - pad), (x + tw + 2 * pad, cy + pad // 2), color, 0.85)
    cv2.putText(img, title, (x + pad, cy - pad // 4), FONT, fs, DARK, 1, cv2.LINE_AA)

    if show_bars and probs is not None:
        _draw_bars(img, face.box, probs, s)


def _draw_bars(img, box, probs: Probs, s: float) -> None:
    """Small panel with all seven probabilities next to the face (right side, or left if it does not fit)."""
    x, y, w, h = box
    H, W = img.shape[:2]
    row, bar_w, name_w, pad = int(20 * s), int(110 * s), int(78 * s), int(8 * s)
    panel_w, panel_h = name_w + bar_w + int(52 * s) + 2 * pad, row * len(EMOTIONS) + 2 * pad
    px = x + w + int(12 * s)
    if px + panel_w > W:
        px = max(0, x - panel_w - int(12 * s))
    py = min(max(int(50 * s), y), max(0, H - panel_h))  # below the title bar, above the bottom edge
    _blend_rect(img, (px, py), (px + panel_w, py + panel_h), DARK, 0.7)
    fs = 0.42 * s
    for i, emo in enumerate(EMOTIONS):
        p = probs[emo]
        ry = py + pad + i * row
        cv2.putText(img, emo, (px + pad, ry + int(row * 0.7)), FONT, fs, WHITE, 1, cv2.LINE_AA)
        bx = px + pad + name_w
        cv2.rectangle(img, (bx, ry + int(row * 0.25)), (bx + bar_w, ry + int(row * 0.75)), (70, 66, 64), -1)
        cv2.rectangle(img, (bx, ry + int(row * 0.25)), (bx + int(bar_w * p), ry + int(row * 0.75)),
                      EMOTION_COLORS[emo], -1)
        cv2.putText(img, f"{p * 100:3.0f}%", (bx + bar_w + int(6 * s), ry + int(row * 0.7)), FONT, fs, WHITE, 1,
                    cv2.LINE_AA)


def draw_hud(img: np.ndarray, fps: Optional[float], n_faces: int, status: str = "",
             recording: bool = False, hints: Sequence[str] = ()) -> None:
    """Title bar with FPS, face count and status, plus a key-hint bar at the bottom."""
    s = _scale(img.shape[1])
    H, W = img.shape[:2]
    bar = int(46 * s)
    _blend_rect(img, (0, 0), (W, bar), DARK, 0.72)
    cv2.putText(img, "Facial Emotion Recognition", (int(16 * s), int(30 * s)), FONT, 0.75 * s, WHITE, 1, cv2.LINE_AA)
    parts = [f"faces {n_faces}"]
    if fps is not None:
        parts.insert(0, f"{fps:4.1f} FPS")
    if status:
        parts.append(status)
    right = "   ".join(parts)
    (tw, _), _ = cv2.getTextSize(right, FONT, 0.55 * s, 1)
    cv2.putText(img, right, (W - tw - int(16 * s), int(30 * s)), FONT, 0.55 * s, ACCENT, 1, cv2.LINE_AA)
    if recording:
        cv2.circle(img, (W - tw - int(34 * s), int(24 * s)), int(7 * s), (60, 60, 230), -1, cv2.LINE_AA)
    if hints:
        text = "   ".join(hints)
        hb = int(32 * s)
        _blend_rect(img, (0, H - hb), (W, H), DARK, 0.6)
        cv2.putText(img, text, (int(16 * s), H - int(11 * s)), FONT, 0.45 * s, MUTED, 1, cv2.LINE_AA)
