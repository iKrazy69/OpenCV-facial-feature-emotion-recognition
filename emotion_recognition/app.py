"""The two ways to run it: a live stream (webcam or video file) and a single image."""
import os
import time
from typing import Callable, Optional, Union

import cv2
import numpy as np

from .emotions import AsyncAnalyzer, DeepFaceModel, EmotionSmoother
from .faces import FaceDetector, crop
from .overlay import draw_face, draw_hud
from .session import SessionLog
from .tracker import FaceTracker

WINDOW = "Facial Emotion Recognition"
HINTS = ("Q quit", "S screenshot", "R record", "F features", "B bars", "M mirror", "SPACE pause")


class FpsMeter:
    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.value: Optional[float] = None
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = max(now - self._last, 1e-6)
        self._last = now
        inst = 1.0 / dt
        self.value = inst if self.value is None else self.smoothing * self.value + (1 - self.smoothing) * inst
        return self.value


class EmotionPipeline:
    """One video frame in, one annotated frame out. Holds everything that lives across frames."""

    def __init__(self, interval: float = 0.25, min_face: int = 60, features: bool = True, bars: bool = True,
                 model_factory: Callable[[], object] = DeepFaceModel):
        self.detector = FaceDetector(min_face=min_face)
        self.tracker = FaceTracker()
        self.smoother = EmotionSmoother()
        self.analyzer = AsyncAnalyzer(interval=interval, model_factory=model_factory)
        self.log = SessionLog()
        self.fps = FpsMeter()
        self.features, self.bars = features, bars
        self.recording = False

    @property
    def status(self) -> str:
        if self.analyzer.error:
            return "model error"
        return "" if self.analyzer.ready.is_set() else "loading model..."

    def process(self, frame: np.ndarray) -> np.ndarray:
        faces = self.detector.detect(frame, features=self.features)
        ids = self.tracker.update([f.box for f in faces])
        # only faces seen in a few frames in a row count; one-frame false detections never reach the model
        tracked = [(f, tid) for f, tid in zip(faces, ids) if self.tracker.is_confirmed(tid)]
        for face, tid in tracked:
            self.analyzer.submit(tid, crop(frame, face.box))
        for tid, probs in self.analyzer.results():
            self.log.add(tid, probs)
            self.smoother.update(tid, probs)
        self.smoother.keep_only(self.tracker.active_ids)

        view = frame.copy()
        for face, tid in tracked:
            draw_face(view, face, tid, self.smoother.get(tid), show_features=self.features, show_bars=self.bars)
        draw_hud(view, self.fps.tick(), len(tracked), self.status, self.recording, HINTS)
        return view

    def close(self) -> None:
        self.analyzer.close()


def _open_source(source: Union[int, str]):
    src = int(source) if isinstance(source, str) and source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        hint = (" (on macOS, allow camera access for your terminal in System Settings > Privacy & Security > Camera)"
                if isinstance(src, int) else "")
        raise SystemExit(f"Could not open video source {source!r}{hint}")
    return cap, isinstance(src, int)


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def run_stream(source: Union[int, str] = 0, interval: float = 0.25, min_face: int = 60,
               features: bool = True, bars: bool = True, mirror: bool = True,
               record_path: Optional[str] = None, log_path: Optional[str] = None,
               output_dir: str = ".") -> SessionLog:
    cap, is_camera = _open_source(source)
    pipe = EmotionPipeline(interval=interval, min_face=min_face, features=features, bars=bars)
    writer, paused, view, error_shown = None, False, None, False
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    cam_fps = cam_fps if 5 <= cam_fps <= 120 else 20.0

    def start_recording(frame, path=None):
        nonlocal writer
        path = path or os.path.join(output_dir, f"recording_{_stamp()}.mp4")
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), cam_fps, (w, h))
        pipe.recording = writer.isOpened()
        print(f"Recording to {path}" if pipe.recording else f"Could not record to {path}")

    def stop_recording():
        nonlocal writer
        if writer is not None:
            writer.release()
            print("Recording stopped")
        writer, pipe.recording = None, False

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("Opening the video. The emotion model loads in the background (about 20 s the first time).")
    try:
        while True:
            if not paused or view is None:
                ok, frame = cap.read()
                if not ok:
                    if is_camera:
                        print("The camera stopped sending frames.")
                    break
                if is_camera and mirror:
                    frame = cv2.flip(frame, 1)
                view = pipe.process(frame)
                if pipe.analyzer.error and not error_shown:
                    print(pipe.analyzer.error)
                    error_shown = True
                if record_path:
                    start_recording(view, record_path)
                    record_path = None
                if writer is not None:
                    writer.write(view)

            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = os.path.join(output_dir, f"screenshot_{_stamp()}.png")
                cv2.imwrite(path, view)
                print(f"Screenshot saved: {path}")
            elif key == ord("r"):
                stop_recording() if writer is not None else start_recording(view)
            elif key == ord("f"):
                pipe.features = not pipe.features
            elif key == ord("b"):
                pipe.bars = not pipe.bars
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord(" "):
                paused = not paused
    finally:
        cap.release()
        stop_recording()
        pipe.close()
        cv2.destroyAllWindows()
        if log_path:
            pipe.log.save_csv(log_path)
            print(f"Session log saved: {log_path}")
        print(pipe.log.summary())
    return pipe.log


def annotate_image(image: np.ndarray, features: bool = True, bars: bool = True, min_face: int = 40, model=None):
    """Detect faces, read each emotion and return (annotated copy, [(face, probs), ...])."""
    detector = FaceDetector(min_face=min_face)
    model = model or DeepFaceModel()
    faces = detector.detect(image, features=features)
    out, results = image.copy(), []
    for i, face in enumerate(faces, start=1):
        probs = model.predict(crop(image, face.box))
        results.append((face, probs))
        draw_face(out, face, i, probs, show_features=features, show_bars=bars)
    draw_hud(out, None, len(faces), "image")
    return out, results


def run_image(path: str, out_path: Optional[str] = None, features: bool = True, bars: bool = True,
              min_face: int = 40, show: bool = True):
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"Could not read image {path!r}")
    out, results = annotate_image(image, features=features, bars=bars, min_face=min_face)
    if not results:
        print("No face found in the image.")
    for i, (face, probs) in enumerate(results, start=1):
        top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print(f"face #{i} at {face.box}: " + ", ".join(f"{e} {p * 100:.1f}%" for e, p in top))
    if out_path:
        cv2.imwrite(out_path, out)
        print(f"Saved: {out_path}")
    if show:
        cv2.imshow(WINDOW, out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out, results
