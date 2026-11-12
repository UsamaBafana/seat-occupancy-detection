"""
save_seat_preview.py
Save a screenshot of one camera with its saved seat ROIs drawn on top, to check
that a seats_camN.json file still lines up with the camera's current view.

Usage (from the project root):
    python scripts/save_seat_preview.py --source 0 --seats output/seats_cam0.json --out output/seats_cam0.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Make the project root importable when this file is run as `python scripts/<name>.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.seat_manager import SeatManager


def grab_frame(source: str, warmup: int = 30):
    """Read one settled frame from a webcam index (matches multi_pipeline's
    DirectShow backend so 'camera 0' means the same physical device here
    as it does in run_dual.py)."""
    src_val = int(source) if source.isdigit() else source
    backend = cv2.CAP_DSHOW if source.isdigit() else 0

    cap = cv2.VideoCapture(src_val, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    frame = None
    for _ in range(max(1, warmup)):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()

    if frame is None:
        raise RuntimeError(f"Could not read a frame from: {source}")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="Save an annotated seat-ROI screenshot")
    ap.add_argument("--source", required=True, help="webcam index (0, 1, ...)")
    ap.add_argument("--seats",  required=True, help="path to saved seats_camN.json")
    ap.add_argument("--out",    required=True, help="path to save the annotated PNG")
    args = ap.parse_args()

    mgr = SeatManager()
    mgr.load_from_json(args.seats)

    if not mgr.seats:
        print(f"[!] No seats loaded from {args.seats} — check the path.")
        return

    frame = grab_frame(args.source)
    annotated = mgr.draw_on_frame(frame.copy())

    cv2.imwrite(args.out, annotated)
    print(f"[SeatPreview] {len(mgr.seats)} seat(s) drawn from {args.seats}")
    print(f"[SeatPreview] Saved -> {args.out}")


if __name__ == "__main__":
    main()
