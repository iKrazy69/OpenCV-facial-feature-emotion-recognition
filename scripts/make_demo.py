"""Re-creates the images in docs/ from the public-domain test photo (NASA, via scikit-image).

    python scripts/make_demo.py

docs/demo.png  one annotated photo (image mode)
docs/demo.gif  the live pipeline on a slow zoom over the same photo: tracking, smoothing, bars and HUD
"""
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from emotion_recognition.app import EmotionPipeline, annotate_image  # noqa: E402
from emotion_recognition.emotions import DeepFaceModel  # noqa: E402

PHOTO = os.path.join(ROOT, "tests", "data", "astronaut.png")
DOCS = os.path.join(ROOT, "docs")


def zoom_frames(img: np.ndarray, n: int = 40, size=(960, 540)):
    """A slow push-in on the photo over a dark background, like a 16:9 video."""
    W, H = size
    bg = np.full((H, W, 3), (34, 28, 24), dtype=np.uint8)
    for i in range(n):
        t = i / (n - 1)
        side = int(H * (1.0 + 0.45 * t))
        fg = cv2.resize(img, (side, side), interpolation=cv2.INTER_CUBIC)
        frame = bg.copy()
        x0 = (W - side) // 2 + int(60 * t)
        y0 = int(-0.12 * side * t)
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        w = min(side - sx0, W - dx0)
        h = min(side - sy0, H - dy0)
        frame[dy0:dy0 + h, dx0:dx0 + w] = fg[sy0:sy0 + h, sx0:sx0 + w]
        yield frame


def save_gif(frames, path: str, width: int = 640, ms: int = 90) -> None:
    from PIL import Image
    pics = []
    for f in frames:
        h = int(f.shape[0] * width / f.shape[1])
        rgb = cv2.cvtColor(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        pics.append(Image.fromarray(rgb).quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                                  dither=Image.Dither.FLOYDSTEINBERG))
    pics[0].save(path, save_all=True, append_images=pics[1:], duration=ms, loop=0, optimize=True)


def main() -> None:
    os.makedirs(DOCS, exist_ok=True)
    photo = cv2.imread(PHOTO)
    big = cv2.resize(photo, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    model = DeepFaceModel()
    out, results = annotate_image(big, model=model, min_face=80)
    cv2.imwrite(os.path.join(DOCS, "demo.png"), out)
    print("docs/demo.png:", [(f.box, max(p, key=p.get)) for f, p in results])

    pipe = EmotionPipeline(interval=0.1, model_factory=lambda: model)
    pipe.analyzer.ready.wait(60)
    views = []
    for frame in zoom_frames(photo):
        views.append(pipe.process(frame))
        time.sleep(0.04)  # give the background worker time, as a real camera would
    pipe.close()
    save_gif(views, os.path.join(DOCS, "demo.gif"))
    print(f"docs/demo.gif: {len(views)} frames, {len(pipe.log.rows)} emotion readings")


if __name__ == "__main__":
    main()
