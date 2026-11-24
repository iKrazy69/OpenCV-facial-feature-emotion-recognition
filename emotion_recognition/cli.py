"""Command line: python -m emotion_recognition [options]"""
import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m emotion_recognition",
        description="Real-time facial feature and emotion recognition (OpenCV + DeepFace).")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--source", default="0",
                     help="webcam index (0, 1, ...) or a video file path (default: 0)")
    src.add_argument("--image", help="analyse one image instead of a video stream")
    p.add_argument("--out", help="with --image: save the annotated image here")
    p.add_argument("--no-show", action="store_true", help="with --image: do not open a window")
    p.add_argument("--interval", type=float, default=0.25,
                   help="seconds between emotion readings of the same face (default: 0.25)")
    p.add_argument("--min-face", type=int, default=60, help="smallest face to detect, in pixels (default: 60)")
    p.add_argument("--no-features", action="store_true", help="do not mark eyes and smile")
    p.add_argument("--no-bars", action="store_true", help="hide the per-face probability bars")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the webcam picture")
    p.add_argument("--record", metavar="FILE.mp4", help="start recording the annotated video right away")
    p.add_argument("--log", metavar="FILE.csv", help="save every emotion reading of the session as CSV")
    p.add_argument("--output-dir", default="captures", help="folder for screenshots and recordings (default: captures)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval <= 0:
        print("--interval must be greater than 0", file=sys.stderr)
        return 2
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # keep TensorFlow start-up noise out of the console

    from .app import run_image, run_stream  # imported late so --help is instant
    if args.image:
        run_image(args.image, out_path=args.out, features=not args.no_features, bars=not args.no_bars,
                  min_face=min(args.min_face, 40), show=not args.no_show)
        return 0
    os.makedirs(args.output_dir, exist_ok=True)
    run_stream(source=args.source, interval=args.interval, min_face=args.min_face,
               features=not args.no_features, bars=not args.no_bars, mirror=not args.no_mirror,
               record_path=args.record, log_path=args.log, output_dir=args.output_dir)
    return 0
