"""Emotion analysis (DeepFace), temporal smoothing, and a background worker that keeps the video smooth."""
import queue
import threading
import time
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from . import EMOTIONS

Probs = Dict[str, float]  # emotion -> probability, sums to 1


def normalise(scores: Dict[str, float]) -> Probs:
    """DeepFace gives percentages; return probabilities over the seven emotions that sum to 1."""
    vals = {e: max(0.0, float(scores.get(e, 0.0))) for e in EMOTIONS}
    total = sum(vals.values())
    if total <= 0:
        return {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
    return {e: v / total for e, v in vals.items()}


def dominant(probs: Probs) -> Tuple[str, float]:
    label = max(probs, key=probs.get)
    return label, probs[label]


class EmotionSmoother:
    """Exponential moving average of each face's probabilities, so labels do not flicker frame to frame."""

    def __init__(self, alpha: float = 0.35):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: Dict[int, Probs] = {}

    def update(self, track_id: int, probs: Probs) -> Probs:
        old = self._state.get(track_id)
        new = probs if old is None else {e: self.alpha * probs[e] + (1 - self.alpha) * old[e] for e in EMOTIONS}
        self._state[track_id] = new
        return new

    def get(self, track_id: int) -> Optional[Probs]:
        return self._state.get(track_id)

    def keep_only(self, track_ids: Iterable[int]) -> None:
        keep = set(track_ids)
        for tid in list(self._state):
            if tid not in keep:
                del self._state[tid]


class DeepFaceModel:
    """Thin wrapper so the heavy TensorFlow import happens once, and only when it is needed."""

    def __init__(self):
        from deepface import DeepFace  # slow import (TensorFlow); done once
        self._deepface = DeepFace
        # warm-up so the first real face is not slow
        self.predict(np.zeros((64, 64, 3), dtype=np.uint8))

    def predict(self, face_bgr: np.ndarray) -> Probs:
        result = self._deepface.analyze(face_bgr, actions=["emotion"], enforce_detection=False,
                                        detector_backend="skip", silent=True)
        if isinstance(result, list):
            result = result[0]
        return normalise(result["emotion"])


class AsyncAnalyzer:
    """Runs the emotion model on a background thread.

    The main loop hands over face crops with ``submit``; the worker analyses the most recent crop of each face
    at most once every ``interval`` seconds, and ``results`` returns whatever finished since the last call.
    The video keeps its frame rate even though one analysis takes much longer than one frame.
    """

    def __init__(self, interval: float = 0.25, model_factory: Callable[[], object] = DeepFaceModel):
        self.interval = interval
        self._model_factory = model_factory
        self._pending: Dict[int, np.ndarray] = {}
        self._last_run: Dict[int, float] = {}
        self._results: "queue.Queue[Tuple[int, Probs]]" = queue.Queue()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.ready = threading.Event()
        self.error: Optional[str] = None
        self._thread = threading.Thread(target=self._run, name="emotion-worker", daemon=True)
        self._thread.start()

    def submit(self, track_id: int, face_bgr: np.ndarray) -> None:
        if face_bgr.size == 0:
            return
        now = time.monotonic()
        with self._lock:
            last = self._last_run.get(track_id)  # None: never analysed (monotonic() may start near 0)
            if last is not None and now - last < self.interval:
                return
            self._pending[track_id] = face_bgr.copy()
        self._wake.set()

    def results(self):
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            model = self._model_factory()
        except Exception as exc:  # model missing, no internet for the first download, ...
            self.error = f"emotion model failed to load: {exc}"
            self.ready.set()
            return
        self.ready.set()
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            while not self._stop.is_set():
                with self._lock:
                    if not self._pending:
                        break
                    track_id, face = self._pending.popitem()
                    self._last_run[track_id] = time.monotonic()
                try:
                    self._results.put((track_id, model.predict(face)))
                except Exception:
                    pass  # a bad crop should never stop the worker
