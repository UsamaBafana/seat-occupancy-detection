"""
main.py
Office Occupancy Detection System — Entry Point

Modes:
  run        : Run live pipeline on webcam/video/RTSP
  train      : Fine-tune YOLOv8 on custom dataset
  enrol      : Enrol faces into the recognition gallery
  evaluate   : Compute metrics from prediction + ground truth CSVs
  visualise  : Generate charts from a logged occupancy CSV

Examples:
  # Live detection (webcam)
  python main.py run

  # Live detection (video file)
  python main.py run --source path/to/office.mp4

  # Fine-tune on custom data
  python main.py train --data data/data.yaml --epochs 50

  # Enrol a person via webcam
  python main.py enrol --name Alice --webcam --count 8

  # Evaluate predictions
  python main.py evaluate --gt output/ground_truth.csv --pred output/predictions.csv

  # Generate visualisations
  python main.py visualise --log output/occupancy_1234567890.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

# ── Logging setup ─────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
    colorize=True,
)
logger.add(
    "logs/run_{time}.log",
    rotation="10 MB",
    level="DEBUG",
    enqueue=True,
)


# ── CLI ───────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Office Occupancy Detection System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # run
    p_run = sub.add_parser("run", help="Run live occupancy detection")
    p_run.add_argument("--source",  default=None,           help="Video source (0=webcam, path, rtsp://)")
    p_run.add_argument("--config",  default="config/settings.yaml", help="Path to settings.yaml")
    p_run.add_argument("--weights", default=None,           help="Override YOLOv8 weights path")
    p_run.add_argument("--seats",   default=None,           help="Path to seats.json (skip interactive drawing)")
    p_run.add_argument("--no-gui",  action="store_true",    help="Disable display window")
    p_run.add_argument("--no-face", action="store_true",    help="Disable face recognition")

    # train
    p_train = sub.add_parser("train", help="Fine-tune YOLOv8 on custom dataset")
    p_train.add_argument("--data",    default="data/data.yaml", help="data.yaml path")
    p_train.add_argument("--model",   default="yolov8n.pt",     help="Base weights")
    p_train.add_argument("--epochs",  type=int, default=50)
    p_train.add_argument("--imgsz",   type=int, default=640)
    p_train.add_argument("--batch",   type=int, default=16)
    p_train.add_argument("--device",  default="",               help="cpu / 0 / mps")
    p_train.add_argument("--name",    default="occupancy_v1")
    p_train.add_argument("--resume",  action="store_true")

    # enrol
    p_enrol = sub.add_parser("enrol", help="Enrol faces into recognition gallery")
    p_enrol.add_argument("--name",    required=True, help="Person name")
    p_enrol.add_argument("--source",  default=None,  help="Source image directory")
    p_enrol.add_argument("--webcam",  action="store_true")
    p_enrol.add_argument("--count",   type=int, default=5, help="Number of webcam captures")
    p_enrol.add_argument("--gallery", default="known_faces/")
    p_enrol.add_argument("--list",    action="store_true", help="List enrolled identities")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate predictions against ground truth")
    p_eval.add_argument("--gt",   required=True, help="Ground truth CSV path")
    p_eval.add_argument("--pred", required=True, help="Predictions CSV path")
    p_eval.add_argument("--out",  default="output/", help="Output directory for reports")
    p_eval.add_argument("--plot", action="store_true", help="Show confusion matrix plot")
    p_eval.add_argument("--direction", default="nearest", choices=["nearest", "backward"],
                        help="How to match labels to predictions: nearest (default) or "
                             "backward = last prediction at/before each label (use for clip-time logs)")
    p_eval.add_argument("--tolerance", type=float, default=2.0,
                        help="Max seconds between a label and its matched prediction (default 2; "
                             "use > io.log_interval_seconds with --direction backward)")

    # visualise
    p_vis = sub.add_parser("visualise", help="Generate charts from occupancy log CSV")
    p_vis.add_argument("--log",  required=True, help="Occupancy log CSV path")
    p_vis.add_argument("--out",  default="output/", help="Output directory for plots")
    p_vis.add_argument("--show", action="store_true", help="Show plots interactively")

    return parser


# ── Mode handlers ─────────────────────────────────────────────

def run_mode(args: argparse.Namespace) -> None:
    from config.config_loader import load_config, override_config
    from src.pipeline import OccupancyPipeline

    cfg = load_config(args.config)

    overrides: dict = {}
    if args.source is not None:
        overrides["io.source"] = args.source
    if args.weights:
        overrides["model.custom_weights"] = args.weights
    if args.no_gui:
        overrides["display.show_window"] = False
    if args.no_face:
        overrides["face_recognition.enabled"] = False
    if overrides:
        cfg = override_config(cfg, overrides)

    pipeline = OccupancyPipeline(cfg)

    # Pre-load seats if provided
    if args.seats:
        pipeline.setup_seats(load_json=args.seats)

    pipeline.run()


def train_mode(args: argparse.Namespace) -> None:
    from src.train import train
    train(args)


def enrol_mode(args: argparse.Namespace) -> None:
    from src.enrol_faces import enrol_from_directory, enrol_from_webcam, list_gallery
    if args.list:
        list_gallery(args.gallery)
    elif args.source:
        enrol_from_directory(args.name, args.source, args.gallery)
    elif args.webcam:
        enrol_from_webcam(args.name, args.gallery, count=args.count)
    else:
        print("Provide --source <dir> or --webcam")
        sys.exit(1)


def evaluate_mode(args: argparse.Namespace) -> None:
    from evaluate.metrics import OccupancyEvaluator
    evaluator = OccupancyEvaluator(
        args.gt, args.pred,
        tolerance_sec=args.tolerance,
        direction=args.direction,
    )
    evaluator.print_report()
    evaluator.save_report(args.out)
    if args.plot:
        evaluator.plot_confusion_matrix(
            save_path=Path(args.out) / "confusion_matrix.png",
            show=True,
        )


def visualise_mode(args: argparse.Namespace) -> None:
    from src.visualiser import generate_all
    generate_all(args.log, output_dir=args.out, show=args.show)


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "run":       run_mode,
        "train":     train_mode,
        "enrol":     enrol_mode,
        "evaluate":  evaluate_mode,
        "visualise": visualise_mode,
    }

    handler = dispatch.get(args.mode)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
