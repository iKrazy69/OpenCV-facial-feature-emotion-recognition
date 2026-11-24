"""Keeps the same id on the same face from frame to frame (greedy IoU matching).

The id lets the app smooth each person's emotion over time instead of mixing faces together. A new track is only
"confirmed" after it has been seen in ``min_hits`` frames in a row, which filters out short false detections.
"""
from typing import Dict, List, Sequence

from .faces import Box


def iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15, min_hits: int = 3):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.tracks: Dict[int, Dict] = {}
        self._next_id = 1

    def update(self, boxes: Sequence[Box]) -> List[int]:
        """Return one track id per box, in the same order as ``boxes``."""
        pairs = sorted(
            ((iou(t["box"], b), tid, i) for tid, t in self.tracks.items() for i, b in enumerate(boxes)),
            reverse=True)
        ids: List[int] = [0] * len(boxes)
        used_tracks, used_boxes = set(), set()
        for score, tid, i in pairs:
            if score < self.iou_threshold:
                break
            if tid in used_tracks or i in used_boxes:
                continue
            ids[i] = tid
            used_tracks.add(tid)
            used_boxes.add(i)

        for i, box in enumerate(boxes):
            if i in used_boxes:
                track = self.tracks[ids[i]]
                streak = track["streak"] + 1
                track.update(box=tuple(box), missed=0, streak=streak,
                             confirmed=track["confirmed"] or streak >= self.min_hits)
            else:
                ids[i] = self._next_id
                self._next_id += 1
                self.tracks[ids[i]] = {"box": tuple(box), "missed": 0, "streak": 1, "confirmed": self.min_hits <= 1}

        for tid in list(self.tracks):
            if tid not in ids:
                self.tracks[tid]["missed"] += 1
                self.tracks[tid]["streak"] = 0
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        return ids

    def is_confirmed(self, track_id: int) -> bool:
        track = self.tracks.get(track_id)
        return track is not None and track["confirmed"]

    @property
    def active_ids(self) -> List[int]:
        return list(self.tracks)
