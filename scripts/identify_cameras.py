"""
identify_cameras.py
Quick visual check: which physical camera is CURRENTLY index 0 vs index 1?

Run this right before every run_dual.py session — Windows doesn't
guarantee camera index order stays the same across restarts, sleep/wake,
or USB replug, especially for external webcams. This takes one snapshot
from each index and saves them side by side so you can glance and know
for certain, instead of assuming.

Usage:
  python scripts/identify_cameras.py
  python scripts/identify_cameras.py --out output/camera_check.png
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def grab_labeled_frame(index: int, warmup: int = 30):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[!] Could not open camera index {index}")
        return None

    frame = None
    for _ in range(max(1, warmup)):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()

    if frame is None:
        print(f"[!] Could not read a frame from camera index {index}")
        return None

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(frame, f"Index {index}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 120), 2)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="Identify which physical camera is which index")
    ap.add_argument("--indices", nargs="+", type=int, default=[0, 1],
                     help="camera indices to check (default: 0 1)")
    ap.add_argument("--out", default="output/camera_check.png",
                     help="where to save the side-by-side comparison image")
    args = ap.parse_args()

    frames = []
    for idx in args.indices:
        f = grab_labeled_frame(idx)
        if f is not None:
            frames.append(f)

    if not frames:
        print("[!] No cameras responded. Check connections.")
        return

    # Resize all to the same height so they stack cleanly side by side.
    target_h = min(f.shape[0] for f in frames)
    resized = []
    for f in frames:
        scale = target_h / f.shape[0]
        w = int(f.shape[1] * scale)
        resized.append(cv2.resize(f, (w, target_h)))

    combined = np.hstack(resized)
    cv2.imwrite(args.out, combined)
    print(f"[IdentifyCameras] Saved -> {args.out}")
    print("Open that image and check which physical camera is labeled Index 0 vs Index 1.")
    print("If it's not what you expect, swap the order in your --sources flag, e.g.:")
    print("  python run_dual.py --sources 1 0 --no-face")


if __name__ == "__main__":
    main()
