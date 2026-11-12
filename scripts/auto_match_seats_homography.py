"""
auto_match_seats_homography.py
Fully automatic shared-seat matching using a calibrated homography — no
per-seat clicking needed (only the one-time calibrate_homography.py step).

For each cam1 seat, projects its ground-contact point through the
homography into cam0's coordinate space, then matches it to whichever
cam0 seat's ground-contact point is closest, if within --max-distance
pixels.

Writes config/shared_seats.yaml in the exact same format
map_shared_seat.py does, so multi_pipeline.py picks it up automatically
— no other changes needed. This is the automatic counterpart to
map_shared_seat.py's manual click-to-pair; use whichever you trust more
for a given session.

Usage:
  python scripts/calibrate_homography.py --cam0-source 0 --cam1-source 1   (once)
  python scripts/auto_match_seats_homography.py                            (every time seats change)
  python scripts/auto_match_seats_homography.py --max-distance 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Make the project root importable when this file is run as `python scripts/<name>.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.seat_manager import SeatManager, SeatROI


def ground_point(seat: SeatROI) -> tuple[float, float]:
    """Bottom-center of the seat's bbox — the closest approximation to
    where it touches the floor, which is what the ground-plane
    homography actually maps between the two camera views."""
    x1, y1, x2, y2 = seat.bbox
    return ((x1 + x2) / 2.0, y2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-match shared seats via calibrated homography")
    ap.add_argument("--cam0-seats", default="output/seats_cam0.json")
    ap.add_argument("--cam1-seats", default="output/seats_cam1.json")
    ap.add_argument("--homography", default="config/homography_cam1_to_cam0.json")
    ap.add_argument("--out", default="config/shared_seats.yaml")
    ap.add_argument("--max-distance", type=float, default=80.0,
                     help="max pixel distance (in cam0's frame) between a projected cam1 "
                          "seat and a cam0 seat to consider them the same physical desk")
    args = ap.parse_args()

    homog_path = Path(args.homography)
    if not homog_path.exists():
        print(f"[!] No homography found at {homog_path} — run calibrate_homography.py first.")
        return
    with open(homog_path) as f:
        data = json.load(f)
    H = np.array(data["homography_cam1_to_cam0"], dtype=np.float64)

    mgr0 = SeatManager()
    mgr0.load_from_json(args.cam0_seats)
    mgr1 = SeatManager()
    mgr1.load_from_json(args.cam1_seats)

    if not mgr0.seats or not mgr1.seats:
        print("[!] One or both seat files are empty — run test_autodetect.py (or "
              "draw_seats_manual.py) on both cameras first.")
        return

    pairs = []
    print("\n[AutoMatch] Projecting cam1 seats into cam0's coordinate space via homography...\n")

    for seat1 in mgr1.seats:
        gx, gy = ground_point(seat1)
        pt = np.array([[[gx, gy]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(pt, H)[0][0]
        px, py = projected

        best_seat = None
        best_dist = float("inf")
        for seat0 in mgr0.seats:
            cx, cy = ground_point(seat0)
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_seat = seat0

        if best_seat is not None and best_dist <= args.max_distance:
            global_id = f"desk{len(pairs) + 1}"
            pairs.append({
                "global_id": global_id,
                "cameras": {"cam0": best_seat.seat_id, "cam1": seat1.seat_id},
            })
            print(f"  cam1 '{seat1.seat_id}' -> projected near cam0 '{best_seat.seat_id}' "
                  f"(distance={best_dist:.1f}px) -> matched as '{global_id}'")
        else:
            nearest = f"{best_dist:.1f}px" if best_seat else "n/a"
            print(f"  cam1 '{seat1.seat_id}' -> no cam0 seat within {args.max_distance}px "
                  f"(nearest: {nearest}) -> not matched, treated as a separate seat")

    if not pairs:
        print("\n[AutoMatch] No matches found — nothing saved. Try increasing --max-distance, "
              "or re-check the calibration points were spread out and accurate.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump({"shared_seats": pairs}, f, default_flow_style=False)

    print(f"\n[AutoMatch] Saved {len(pairs)} shared seat pair(s) -> {out_path}")
    print("             settings.yaml was not touched.")


if __name__ == "__main__":
    main()
