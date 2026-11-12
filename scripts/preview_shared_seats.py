"""
preview_shared_seats.py
Read-only visual check of config/shared_seats.yaml: shows both cameras
side by side with the paired global seat drawn and labeled on BOTH
sides in the same color, so you can eyeball "yes, that's the same desk"
— the same visual proof map_shared_seat.py gives you while you're
clicking pairs, but for a pairing that already exists (e.g. one written
automatically by auto_match_seats_homography.py).

Usage:
  python scripts/preview_shared_seats.py --cam0-source 0 --cam1-source 1
  python scripts/preview_shared_seats.py --cam0-source 0 --cam1-source 1 --out output/shared_seats_preview.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Make the project root importable when this file is run as `python scripts/<name>.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.seat_manager import SeatManager


def grab_frame(source: str, warmup: int = 30):
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
    ap = argparse.ArgumentParser(description="Visually confirm a global (shared) seat pairing")
    ap.add_argument("--cam0-source", default="0")
    ap.add_argument("--cam1-source", default="1")
    ap.add_argument("--cam0-seats", default="output/seats_cam0.json")
    ap.add_argument("--cam1-seats", default="output/seats_cam1.json")
    ap.add_argument("--shared", default="config/shared_seats.yaml")
    ap.add_argument("--out", default="output/shared_seats_preview.png",
                     help="path to save the annotated side-by-side image")
    ap.add_argument("--no-window", action="store_true",
                     help="skip the live preview window, just save the PNG")
    args = ap.parse_args()

    shared_path = Path(args.shared)
    if not shared_path.exists():
        print(f"[!] No shared seats file at {shared_path} — run map_shared_seat.py or "
              f"auto_match_seats_homography.py first.")
        return
    with open(shared_path) as f:
        shared = yaml.safe_load(f) or {}
    pairs = shared.get("shared_seats", [])
    if not pairs:
        print(f"[!] {shared_path} has no shared_seats entries — nothing to show.")
        return

    mgr0 = SeatManager()
    mgr0.load_from_json(args.cam0_seats)
    mgr1 = SeatManager()
    mgr1.load_from_json(args.cam1_seats)

    def find_seat(mgr, seat_id):
        return next((s for s in mgr.seats if s.seat_id == seat_id), None)

    frame0 = grab_frame(args.cam0_source)
    frame1 = grab_frame(args.cam1_source)

    h = min(frame0.shape[0], frame1.shape[0])

    def resize(v):
        scale = h / v.shape[0]
        return cv2.resize(v, (int(v.shape[1] * scale), h)), scale

    view0, scale0 = resize(frame0)
    view1, scale1 = resize(frame1)
    offset_x = view0.shape[1]
    combined = np.hstack([view0, view1])

    colors = [(0, 220, 120), (0, 160, 255), (255, 120, 0), (200, 0, 200),
              (0, 220, 220), (255, 255, 0), (150, 80, 255), (80, 200, 80)]

    print(f"\n[Preview] {len(pairs)} global seat pairing(s) found in {shared_path}:\n")

    shown = 0
    for i, pair in enumerate(pairs):
        global_id = pair.get("global_id", f"desk{i + 1}")
        cams = pair.get("cameras", {})
        seat0_id, seat1_id = cams.get("cam0"), cams.get("cam1")
        seat0 = find_seat(mgr0, seat0_id) if seat0_id else None
        seat1 = find_seat(mgr1, seat1_id) if seat1_id else None

        if seat0 is None or seat1 is None:
            print(f"  [!] '{global_id}': couldn't find cam0='{seat0_id}' / cam1='{seat1_id}' "
                  f"in the loaded seat files — skipping.")
            continue

        color = colors[shown % len(colors)]
        shown += 1
        x1, y1, x2, y2 = seat0.bbox
        dx1, dy1, dx2, dy2 = int(x1 * scale0), int(y1 * scale0), int(x2 * scale0), int(y2 * scale0)
        cv2.rectangle(combined, (dx1, dy1), (dx2, dy2), color, 3)
        cv2.putText(combined, global_id, (dx1 + 4, dy1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        x1, y1, x2, y2 = seat1.bbox
        dx1, dy1, dx2, dy2 = int(x1 * scale1) + offset_x, int(y1 * scale1), int(x2 * scale1) + offset_x, int(y2 * scale1)
        cv2.rectangle(combined, (dx1, dy1), (dx2, dy2), color, 3)
        cv2.putText(combined, global_id, (dx1 + 4, dy1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        print(f"  '{global_id}': cam0 '{seat0_id}' <-> cam1 '{seat1_id}'  (same color box, both sides)")

    cv2.line(combined, (offset_x, 0), (offset_x, combined.shape[0]), (255, 255, 255), 2)
    cv2.putText(combined, "cam0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "cam1", (offset_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)
    print(f"\n[Preview] Saved -> {out_path}")
    print("[Preview] Check: is the same-colored box sitting on the same physical desk in both halves?")

    if not args.no_window:
        cv2.imshow("Global seat pairing check (cam0 left | cam1 right) - any key to close", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
