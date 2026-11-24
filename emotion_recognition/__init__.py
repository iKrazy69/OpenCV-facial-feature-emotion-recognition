"""Real-time facial feature and emotion recognition with OpenCV and DeepFace."""

__version__ = "1.0.0"

# The seven expressions the DeepFace emotion model predicts, in a fixed display order.
EMOTIONS = ("happy", "neutral", "surprise", "sad", "angry", "fear", "disgust")

# BGR colours, one per emotion, used for boxes, labels and bars.
EMOTION_COLORS = {
    "happy": (80, 200, 255),     # warm yellow
    "neutral": (200, 200, 200),  # grey
    "surprise": (255, 160, 60),  # sky blue
    "sad": (200, 110, 40),       # deep blue
    "angry": (60, 60, 230),      # red
    "fear": (200, 90, 170),      # purple
    "disgust": (80, 170, 60),    # green
}
