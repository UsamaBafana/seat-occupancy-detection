"""
run_dual.py
Launcher for the multi-camera occupancy pipeline.

Examples (run from PowerShell):
    python run_dual.py --sources 0 1
    python run_dual.py --sources 0 1 --no-face
    python run_dual.py --sources 0 1 --no-face --every 3 --imgsz 320
    python run_dual.py --sources 0 1 2 3 4 --no-face --every 3 --imgsz 320
    python run_dual.py --sources 0 1 --no-face --save-video     # also saves output/annotated_camN_<time>.mp4
    python run_dual.py --sources 0 1 --no-face --log-resources  # logs CPU/RAM/stage ms for validate_cost_model.py

Keys in a camera window:
    1-9  mark the start of test step N (shown on screen/video, saved to output/markers_<time>.csv)
    0    between steps
    q    quit
    
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config.config_loader import load_config, override_config
from src.multi_pipeline import MultiCameraPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual / multi-camera occupancy detection")
    parser.add_argument("--sources", nargs="+", default=["0", "1"],
                        help="Camera indices, e.g. --sources 0 1")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--weights", default=None, help="Override YOLOv8 weights path")
    parser.add_argument("--no-face", action="store_true", help="Disable face recognition")
    parser.add_argument("--every", type=int, default=2,
                        help="Run detection every N-th frame (default 2)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Override detection image size, e.g. 320 (smaller = faster)")
    parser.add_argument("--log-resources", action="store_true",
                        help="Log CPU / RAM / per-stage ms every second to output/resources_<time>.csv "
                             "(for validate_cost_model.py)")
    parser.add_argument("--save-video", action="store_true",
                        help="Save each camera's annotated view to output/annotated_camN_<time>.mp4")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    cfg = load_config(args.config)

    overrides: dict = {}
    if args.weights:
        overrides["model.custom_weights"] = args.weights
    if args.no_face:
        overrides["face_recognition.enabled"] = False
    if args.imgsz:
        overrides["model.imgsz"] = args.imgsz
    if overrides:
        cfg = override_config(cfg, overrides)

    pipeline = MultiCameraPipeline(cfg, args.sources, process_every=args.every,
                                   save_video=args.save_video,
                                   log_resources=args.log_resources)
    pipeline.run()


if __name__ == "__main__":
    main()
