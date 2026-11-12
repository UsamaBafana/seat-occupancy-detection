"""
record_ground_truth.py
Hand-log ground-truth seat occupancy in real time while a session is running.

A small window is shown; key presses mark the true state of a seat, and each
press is written as one row in the same CSV schema as the pipeline's own
occupancy log (timestamp, frame_id, seat_id, state, confidence, occupant_id),
so the two files can be compared directly by evaluate/metrics.py.

Keys:  o = occupied   v = vacant   1-9 = select seat (with --seats)   q = quit

Usage (from the project root):
    python scripts/record_ground_truth.py --seat global_desk1
    python scripts/record_ground_truth.py --seats global_desk1 global_desk2 cam0_seat_01

For frame-accurate labelling of a recorded clip, use label_ground_truth.py instead.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

COLUMNS = ["timestamp", "frame_id", "seat_id", "state", "confidence", "occupant_id"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Hand-log ground truth occupancy in real time")
    ap.add_argument("--seat",  default=None, help="single seat id (default mode)")
    ap.add_argument("--seats", nargs="+", default=None,
                    help="multiple seat ids, selected by number key")
    ap.add_argument("--out",   default=None,
                    help="output CSV path (default: output/ground_truth_<time>.csv)")
    args = ap.parse_args()

    if args.seats:
        seat_ids = args.seats
    elif args.seat:
        seat_ids = [args.seat]
    else:
        seat_ids = ["global_desk1"]

    out_path = Path(args.out) if args.out else Path("output") / f"ground_truth_{int(time.time())}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    active_idx = 0
    rows_written = 0

    print("\n[GroundTruth] Logging to:", out_path)
    print("[GroundTruth] Seats:", seat_ids)
    print("[GroundTruth] Press 'o'=occupied  'v'=vacant  digits=select seat  'q'=stop\n")

    canvas = np.zeros((160, 520, 3), dtype=np.uint8)
    win = "Ground Truth Logger - keep this window focused"
    cv2.namedWindow(win)

    def log_row(seat_id: str, state: str) -> None:
        nonlocal rows_written
        row = {
            "timestamp": time.time(),
            "frame_id": 0,          # unused now — matching is by timestamp
            "seat_id": seat_id,
            "state": state,
            "confidence": 1.0,
            "occupant_id": "",
        }
        with open(out_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)
        rows_written += 1
        print(f"  [{time.strftime('%H:%M:%S')}] {seat_id} -> {state}")

    while True:
        display = canvas.copy()
        active_seat = seat_ids[active_idx]
        cv2.putText(display, f"Active seat: {active_seat}", (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 120), 2)
        cv2.putText(display, "'o'=occupied  'v'=vacant  'q'=stop", (14, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        if len(seat_ids) > 1:
            cv2.putText(display, "digits 1-9 = select seat", (14, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(display, f"Rows logged: {rows_written}", (14, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
        cv2.imshow(win, display)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("o"):
            log_row(active_seat, "occupied")
        elif key == ord("v"):
            log_row(active_seat, "vacant")
        elif ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(seat_ids):
                active_idx = idx
                print(f"  Active seat -> {seat_ids[active_idx]}")

    cv2.destroyAllWindows()
    print(f"\n[GroundTruth] Saved {rows_written} row(s) -> {out_path}")


if __name__ == "__main__":
    main()
