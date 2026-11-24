"""Fast tests that need no webcam and no TensorFlow:  python -m unittest discover tests"""
import os
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from emotion_recognition import EMOTIONS  # noqa: E402
from emotion_recognition.cli import build_parser  # noqa: E402
from emotion_recognition.emotions import AsyncAnalyzer, EmotionSmoother, dominant, normalise  # noqa: E402
from emotion_recognition.faces import Face, crop  # noqa: E402
from emotion_recognition.overlay import draw_face, draw_hud  # noqa: E402
from emotion_recognition.session import SessionLog  # noqa: E402
from emotion_recognition.tracker import FaceTracker, iou  # noqa: E402


def probs(**kw):
    return normalise({e: kw.get(e, 0.0) for e in EMOTIONS})


class FakeModel:
    """Stands in for DeepFace: always 'happy'."""
    def predict(self, face):
        return probs(happy=90, neutral=10)


class TestEmotionMath(unittest.TestCase):
    def test_normalise_sums_to_one_and_keeps_all_emotions(self):
        p = normalise({"happy": 50, "sad": 25, "angry": 25, "unknown": 99})
        self.assertEqual(set(p), set(EMOTIONS))
        self.assertAlmostEqual(sum(p.values()), 1.0)
        self.assertAlmostEqual(p["happy"], 0.5)

    def test_normalise_all_zero_is_uniform(self):
        p = normalise({})
        self.assertAlmostEqual(p["fear"], 1 / len(EMOTIONS))

    def test_dominant(self):
        self.assertEqual(dominant(probs(surprise=70, happy=30))[0], "surprise")

    def test_smoother_moves_gradually_and_forgets(self):
        s = EmotionSmoother(alpha=0.5)
        s.update(1, probs(happy=100))
        out = s.update(1, probs(sad=100))
        self.assertAlmostEqual(out["happy"], 0.5)
        self.assertAlmostEqual(out["sad"], 0.5)
        s.keep_only([])
        self.assertIsNone(s.get(1))

    def test_smoother_rejects_bad_alpha(self):
        with self.assertRaises(ValueError):
            EmotionSmoother(alpha=0)


class TestTracker(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 5, 5)), 0.0)

    def test_ids_follow_moving_faces(self):
        t = FaceTracker()
        a, b = t.update([(10, 10, 100, 100), (300, 10, 100, 100)])
        self.assertNotEqual(a, b)
        b2, a2 = t.update([(305, 12, 100, 100), (14, 12, 100, 100)])  # order swapped, small moves
        self.assertEqual((a2, b2), (a, b))

    def test_face_is_confirmed_only_after_min_hits(self):
        t = FaceTracker(min_hits=3)
        (tid,) = t.update([(0, 0, 50, 50)])
        self.assertFalse(t.is_confirmed(tid))
        t.update([(2, 0, 50, 50)])
        self.assertFalse(t.is_confirmed(tid))
        t.update([(4, 0, 50, 50)])
        self.assertTrue(t.is_confirmed(tid))

    def test_sightings_with_gaps_are_not_confirmed(self):
        t = FaceTracker(min_hits=3)
        (tid,) = t.update([(0, 0, 50, 50)])
        for _ in range(4):
            t.update([])
            t.update([(0, 0, 50, 50)])  # seen every other frame: a flickering false detection
        self.assertFalse(t.is_confirmed(tid))

    def test_lost_face_is_dropped_after_max_missed(self):
        t = FaceTracker(max_missed=2)
        (tid,) = t.update([(0, 0, 50, 50)])
        for _ in range(3):
            t.update([])
        self.assertNotIn(tid, t.active_ids)
        (new,) = t.update([(0, 0, 50, 50)])
        self.assertNotEqual(new, tid)


class TestDrawingAndSession(unittest.TestCase):
    def test_crop_is_clipped_to_frame(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        self.assertEqual(crop(frame, (90, 90, 30, 30), margin=0).shape[:2], (10, 10))
        self.assertEqual(crop(frame, (90, 90, 30, 30)).shape[:2], (14, 14))  # 15% margin, then clipped

    def test_overlay_draws_without_errors_at_any_size(self):
        for w, h in ((320, 240), (1280, 720), (1920, 1080)):
            img = np.zeros((h, w, 3), np.uint8)
            face = Face(box=(w - 80, 5, 70, 70), eyes=[(w - 70, 20, 15, 15)], smile=(w - 65, 50, 30, 10))
            draw_face(img, face, 3, probs(happy=80, neutral=20))
            draw_face(img, face, 4, None)
            draw_hud(img, 29.7, 2, "loading model...", recording=True, hints=("Q quit",))
            self.assertGreater(int(img.sum()), 0)

    def test_session_summary_and_csv(self):
        log = SessionLog()
        log.add(1, probs(happy=100), now=log.start + 1)
        log.add(1, probs(sad=100), now=log.start + 2)
        log.add(2, probs(happy=100), now=log.start + 3)
        self.assertAlmostEqual(log.shares()["happy"], 2 / 3)
        self.assertIn("2 face(s)", log.summary())
        path = os.path.join(ROOT, "tests", "_session_test.csv")
        try:
            log.save_csv(path)
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(len(fh.read().strip().splitlines()), 4)
        finally:
            os.remove(path)


class TestAsyncAnalyzer(unittest.TestCase):
    def test_results_arrive_and_interval_limits_rate(self):
        an = AsyncAnalyzer(interval=10.0, model_factory=FakeModel)
        self.assertTrue(an.ready.wait(5))
        face = np.zeros((40, 40, 3), np.uint8)
        an.submit(7, face)
        deadline, got = time.time() + 5, []
        while not got and time.time() < deadline:
            got = an.results()
            time.sleep(0.01)
        an.submit(7, face)  # within the 10 s interval: ignored
        time.sleep(0.2)
        later = an.results()
        an.close()
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][0], 7)
        self.assertEqual(dominant(got[0][1])[0], "happy")
        self.assertEqual(later, [])

    def test_model_load_failure_is_reported_not_raised(self):
        def broken():
            raise RuntimeError("no weights")
        an = AsyncAnalyzer(model_factory=broken)
        self.assertTrue(an.ready.wait(5))
        self.assertIn("no weights", an.error)
        an.close()


class TestCli(unittest.TestCase):
    def test_defaults_and_flags(self):
        a = build_parser().parse_args([])
        self.assertEqual(a.source, "0")
        self.assertFalse(a.no_features)
        a = build_parser().parse_args(["--image", "x.png", "--out", "y.png", "--no-bars"])
        self.assertEqual((a.image, a.out, a.no_bars), ("x.png", "y.png", True))

    def test_source_and_image_are_exclusive(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--source", "1", "--image", "x.png"])


if __name__ == "__main__":
    unittest.main()
