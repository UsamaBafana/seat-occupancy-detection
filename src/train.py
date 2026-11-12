"""
src/train.py
Fine-tunes YOLOv8 on a custom office occupancy dataset.

Dataset must be in Ultralytics YOLO format:
  data/
    images/
      train/  *.jpg
      val/    *.jpg
    labels/
      train/  *.txt
      val/    *.txt
  data.yaml   ← class definitions

Classes (suggested):
  0: person
  1: seated_person   (optional, if you label separately)

Usage:
  python -m src.train --data data/data.yaml --epochs 50 --model yolov8n.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for occupancy detection")
    parser.add_argument("--data",    type=str, default="data/data.yaml",  help="Path to data.yaml")
    parser.add_argument("--model",   type=str, default="yolov8n.pt",      help="Base weights")
    parser.add_argument("--epochs",  type=int, default=50,                help="Training epochs")
    parser.add_argument("--imgsz",   type=int, default=640,               help="Image size")
    parser.add_argument("--batch",   type=int, default=16,                help="Batch size (-1 = auto)")
    parser.add_argument("--device",  type=str, default="",                help="cpu / 0 / 0,1 / mps")
    parser.add_argument("--project", type=str, default="runs/train",      help="Output project dir")
    parser.add_argument("--name",    type=str, default="occupancy_v1",    help="Run name")
    parser.add_argument("--resume",  action="store_true",                  help="Resume from last checkpoint")
    return parser.parse_args()


def create_data_yaml_template(output_path: str = "data/data.yaml") -> None:
    """Generate a template data.yaml if one doesn't exist."""
    path = Path(output_path)
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    template = """\
# data.yaml — YOLOv8 dataset configuration
# Edit paths to point to your actual images/labels directories.

path: data           # root directory
train: images/train
val:   images/val
test:  images/test   # optional

# Classes
nc: 1                # number of classes
names:
  0: person

# ── Optional augmentation hints ─────────────────────────────
# These are passed directly to Ultralytics trainer via CLI or code.
# hsv_h: 0.015
# hsv_s: 0.7
# hsv_v: 0.4
# degrees: 10
# translate: 0.1
# scale: 0.5
# fliplr: 0.5
"""
    path.write_text(template)
    logger.info(f"[Train] Created template data.yaml → {path}")
    print(f"\n[!] Edit {path} to point to your dataset before training.\n")


def train(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as err:
        raise ImportError("pip install ultralytics") from err

    data_path = Path(args.data)
    if not data_path.exists():
        logger.warning(f"data.yaml not found at {data_path} — creating template.")
        create_data_yaml_template(str(data_path))
        return

    logger.info(f"[Train] Starting fine-tune  model={args.model}  epochs={args.epochs}")

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or "cpu",
        project="runs/train",
        name="occupancy_v1",
        resume=args.resume,
        # ── Augmentation ──────────────────────────────
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        # ── Early stopping ────────────────────────────
        patience=15,
        # ── Logging ───────────────────────────────────
        verbose=True,
        plots=True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    logger.info(f"[Train] Done. Best weights → {best_weights}")

    # ── Validation on val split ───────────────────────────────
    logger.info("[Train] Running validation on best weights…")
    best_model = YOLO(str(best_weights))
    val_results = best_model.val(data=str(data_path), imgsz=args.imgsz)

    print("\n── Validation Metrics ──")
    print(f"  mAP50     : {val_results.box.map50:.4f}")
    print(f"  mAP50-95  : {val_results.box.map:.4f}")
    print(f"  Precision : {val_results.box.mp:.4f}")
    print(f"  Recall    : {val_results.box.mr:.4f}")

    return best_weights


if __name__ == "__main__":
    args = parse_args()
    train(args)
