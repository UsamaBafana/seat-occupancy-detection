"""
setup_and_run.py
One command: detect cameras -> auto-detect seats per camera (skip if
already saved) -> click-to-pair any shared desks -> launch the live
dual/multi-camera pipeline.

This chains your existing, already-tested scripts (test_autodetect.py,
map_shared_seat.py, run_dual.py) in sequence rather than reimplementing
them, so you keep the two checkpoints that already caught real bugs:
  - a quick camera preview so you pick the right physical camera for
    each slot (this is what caught the swapped-index bug earlier)
  - click-to-pair for shared desks (this is what confirms it's really
    the same physical desk, not just a matching seat_id string)
Everything else — remembering flags, typing paths, running each script
in the right order — is now automatic.

Usage:
  python scripts/setup_and_run.py
  python scripts/setup_and_run.py --conf 0.35 --min-ratio 0.35 --match-iou 0.3
  python scripts/setup_and_run.py --redetect        # force re-running auto-detect
                                             # even if seats_camN.json exists
  python scripts/setup_and_run.py --skip-setup      # skip straight to run_dual.py,
                                             # reusing last session's cameras

NOTE: the click-to-pair step (map_shared_seat.py) currently only pairs
the first two cameras. If you're running more than two, pair those two
first and add further pairs by running map_shared_seat.py directly
afterward with different --cam0/--cam1 arguments.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DUAL = str(PROJECT_ROOT / "run_dual.py")
TEST_AUTODETECT = str(PROJECT_ROOT / "scripts" / "test_autodetect.py")
MAP_SHARED_SEAT = str(PROJECT_ROOT / "scripts" / "map_shared_seat.py")


def find_cameras(max_index: int = 6) -> list[int]:
    """Try opening camera indices 0..max_index-1, return the ones that work."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found


def preview_grid(indices: list[int], out_path: str = "output/camera_check.png") -> None:
    """Save a quick labeled side-by-side snapshot so you can tell indices apart."""
    frames = []
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        frame = None
        for _ in range(20):
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        if frame is None:
            continue
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(frame, f"Index {idx}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 120), 2)
        frames.append(frame)

    if not frames:
        return

    h = min(f.shape[0] for f in frames)
    resized = [cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)) for f in frames]
    combined = np.hstack(resized)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, combined)
    print(f"[Setup] Saved a preview of every detected camera -> {out_path}")


def run(cmd: list[str]) -> None:
    print(f"\n[Setup] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[Setup] Command exited with an error (code {result.returncode}). Stopping.")
        sys.exit(result.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="One-command camera setup + run")
    ap.add_argument("--conf",       type=float, default=0.35)
    ap.add_argument("--min-ratio",  type=float, default=0.35)
    ap.add_argument("--match-iou",  type=float, default=0.3)
    ap.add_argument("--frames",     type=int,   default=100)
    ap.add_argument("--weights",    default="models/yolov8n.pt")
    ap.add_argument("--skip-setup", action="store_true",
                     help="skip straight to run_dual.py using the last session's cameras")
    ap.add_argument("--redetect",   action="store_true",
                     help="re-run auto-detect even if a seats_camN.json already exists")
    args = ap.parse_args()

    last_sources_file = Path("output/.last_sources.txt")

    if args.skip_setup:
        if not last_sources_file.exists():
            print("[!] No previous session found — run once without --skip-setup first.")
            return
        sources = last_sources_file.read_text().strip().split(",")
        run([sys.executable, RUN_DUAL, "--sources", *sources, "--no-face"])
        return

    # 1. Find connected cameras automatically
    print("[Setup] Looking for connected cameras...")
    detected = find_cameras()
    if not detected:
        print("[!] No cameras found. Check connections.")
        return
    print(f"[Setup] Found camera index(es): {detected}")

    if len(detected) > 1:
        preview_grid(detected)
        print("[Setup] Open output/camera_check.png to see which physical camera is which.")
        chosen = input(
            f"Enter the indices to use, in order, comma-separated, no quotes "
            f"(e.g. {detected[0]},{detected[1]}): "
        ).strip()
        sources = [s.strip().strip("'\"") for s in chosen.split(",") if s.strip().strip("'\"")]
    else:
        sources = [str(detected[0])]

    last_sources_file.parent.mkdir(parents=True, exist_ok=True)
    last_sources_file.write_text(",".join(sources))

    # 2. Auto-detect seats per camera (skip if already saved, unless --redetect)
    for i, src in enumerate(sources):
        seats_path = Path(f"output/seats_cam{i}.json")
        if seats_path.exists() and not args.redetect:
            print(f"[Setup] cam{i}: using existing {seats_path} (pass --redetect to redo)")
            continue

        run([
            sys.executable, TEST_AUTODETECT,
            "--source", src,
            "--live",
            "--frames", str(args.frames),
            "--conf", str(args.conf),
            "--min-ratio", str(args.min_ratio),
            "--match-iou", str(args.match_iou),
            "--weights", args.weights,
            "--save", str(seats_path),
        ])

    # 3. Click-to-pair shared seats, only if more than one camera
    if len(sources) > 1:
        print("\n[Setup] If any desk is visible to more than one camera, pair it now.")
        print("        If none overlap, just press 'q' immediately without clicking.\n")
        run([sys.executable, MAP_SHARED_SEAT,
             "--cam0-source", sources[0], "--cam1-source", sources[1],
             "--cam0-seats", "output/seats_cam0.json",
             "--cam1-seats", "output/seats_cam1.json"])

    # 4. Launch the real pipeline
    run([sys.executable, RUN_DUAL, "--sources", *sources, "--no-face"])


if __name__ == "__main__":
    main()