"""Records every emotion reading of a session and summarises it at the end."""
import csv
import time
from collections import Counter
from typing import Dict, List, Optional

from . import EMOTIONS
from .emotions import Probs, dominant


class SessionLog:
    def __init__(self):
        self.start = time.time()
        self.rows: List[Dict] = []

    def add(self, track_id: int, probs: Probs, now: Optional[float] = None) -> None:
        label, conf = dominant(probs)
        t = (now if now is not None else time.time()) - self.start
        self.rows.append({"t_seconds": round(t, 3), "face_id": track_id, "emotion": label,
                          "confidence": round(conf, 4), **{e: round(probs[e], 4) for e in EMOTIONS}})

    def shares(self) -> Dict[str, float]:
        """Share of readings per dominant emotion (all faces together)."""
        counts = Counter(r["emotion"] for r in self.rows)
        n = sum(counts.values())
        return {e: counts.get(e, 0) / n for e in EMOTIONS} if n else {}

    def save_csv(self, path: str) -> None:
        fields = ["t_seconds", "face_id", "emotion", "confidence", *EMOTIONS]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def summary(self) -> str:
        shares = self.shares()
        if not shares:
            return "No emotion readings in this session."
        faces = len({r["face_id"] for r in self.rows})
        lines = [f"Session: {len(self.rows)} readings, {faces} face(s), "
                 f"{self.rows[-1]['t_seconds']:.0f} s"]
        for emo, share in sorted(shares.items(), key=lambda kv: kv[1], reverse=True):
            if share > 0:
                lines.append(f"  {emo:<9} {'#' * round(share * 30):<30} {share * 100:5.1f}%")
        return "\n".join(lines)
