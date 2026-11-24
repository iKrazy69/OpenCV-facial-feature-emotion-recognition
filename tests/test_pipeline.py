"""End-to-end tests on a real photo (public-domain NASA portrait from scikit-image).

The detection tests need only OpenCV. The emotion test loads DeepFace (about 20 s) and is skipped when DeepFace
or its downloaded weights are missing.
"""
import os
import sys
import time
import unittest

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from emotion_recognition.app import EmotionPipeline, annotate_image  # noqa: E402
from emotion_recognition.faces import FaceDetector  # noqa: E402
from tests.test_core import FakeModel  # noqa: E402

PHOTO = os.path.join(ROOT, "tests", "data", "astronaut.png")
WEIGHTS = os.path.expanduser("~/.deepface/weights/facial_expression_model_weights.h5")


def _deepface_available() -> bool:
    try:
        import deepface  # noqa: F401
    except ImportError:
        return False
    return os.path.exists(WEIGHTS)


class TestOnRealPhoto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.photo = cv2.imread(PHOTO)
        assert cls.photo is not None, "missing tests/data/astronaut.png"

    def test_one_face_with_eyes_and_smile(self):
        faces = FaceDetector(min_face=40).detect(self.photo)
        self.assertEqual(len(faces), 1)
        self.assertEqual(len(faces[0].eyes), 2)
        self.assertIsNotNone(faces[0].smile)

    def test_pipeline_keeps_one_id_on_a_moving_face(self):
        pipe = EmotionPipeline(interval=0.0, min_face=40, model_factory=FakeModel)
        try:
            pipe.analyzer.ready.wait(5)
            for dx in range(0, 30, 3):
                shifted = cv2.warpAffine(self.photo, cv2.getRotationMatrix2D((0, 0), 0, 1) + [[0, 0, dx], [0, 0, 0]],
                                         self.photo.shape[1::-1])
                view = pipe.process(shifted)
                time.sleep(0.02)
            self.assertEqual(view.shape, self.photo.shape)
            self.assertEqual(len(pipe.tracker.active_ids), 1)
            self.assertGreater(len(pipe.log.rows), 0)
            self.assertEqual(pipe.log.rows[-1]["emotion"], "happy")
        finally:
            pipe.close()

    @unittest.skipUnless(_deepface_available(), "DeepFace or its emotion weights are not installed")
    def test_real_model_reads_the_smile_as_happy(self):
        out, results = annotate_image(self.photo, min_face=40)
        self.assertEqual(len(results), 1)
        probs = results[0][1]
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)
        self.assertEqual(max(probs, key=probs.get), "happy")
        self.assertEqual(out.shape, self.photo.shape)


if __name__ == "__main__":
    unittest.main()
