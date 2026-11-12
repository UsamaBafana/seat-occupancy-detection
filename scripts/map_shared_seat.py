"""
map_shared_seat.py
Click-to-pair shared desks across two cameras — no settings.yaml editing.

VISUAL FEEDBACK
---------------
  Green box + seat_id  : the original seat, as detected/drawn
  Yellow box           : your click, waiting on its match on the other side
  Green-filled box + desk name : a confirmed pair (stays visible so you
                                  can see your progress across the session)

Run this any time after re-running test_autodetect.py (or drawing seats
manually) on both cameras.

Usage:
  python scripts/map_shared_seat.py --cam0-source 0 --cam1-source 1
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

from src.seat_manager import SeatManager, SeatROI


def grab_frame(source: str, warmup: int = 30):
    if not source.isdigit():
        img = cv2.imread(source)
        if img is not None:
            return img
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source,
                            cv2.CAP_DSHOW if source.isdigit() else 0)
    if source.isdigit():
        # Must match run_dual.py's resolution, or the saved seat boxes
        # (pixel coordinates) won't line up with this preview.
        try:
            from config.config_loader import load_config
            io = load_config("config/settings.yaml").io
            cw, ch = io.get("camera_width", None), io.get("camera_height", None)
            if cw and ch:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cw))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(ch))
        except Exception as e:
            print(f"[!] Could not read camera resolution from settings.yaml: {e}")
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


def find_clicked_seat(mgr: SeatManager, x: int, y: int):
    for seat in mgr.seats:
        x1, y1, x2, y2 = seat.bbox
        if x1 <= x <= x2 and y1 <= y <= y2:
            return seat
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Click-to-pair shared seats across two cameras")
    ap.add_argument("--cam0-source", default="0")
    ap.add_argument("--cam1-source", default="1")
    ap.add_argument("--cam0-seats",  default="output/seats_cam0.json")
    ap.add_argument("--cam1-seats",  default="output/seats_cam1.json")
    ap.add_argument("--out",         default="config/shared_seats.yaml")
    args = ap.parse_args()

    mgr0 = SeatManager()
    mgr0.load_from_json(args.cam0_seats)
    mgr1 = SeatManager()
    mgr1.load_from_json(args.cam1_seats)

    if not mgr0.seats or not mgr1.seats:
        print("[!] One or both seat files are empty — run test_autodetect.py "
              "(or draw_seats_manual.py) on both cameras first.")
        return

    frame0 = grab_frame(args.cam0_source)
    frame1 = grab_frame(args.cam1_source)

    view0 = mgr0.draw_on_frame(frame0.copy())
    view1 = mgr1.draw_on_frame(frame1.copy())

    h = min(view0.shape[0], view1.shape[0])

    def resize(v):
        scale = h / v.shape[0]
        return cv2.resize(v, (int(v.shape[1] * scale), h))

    view0r, view1r = resize(view0), resize(view1)
    offset_x = view0r.shape[1]
    base_canvas = np.hstack([view0r, view1r])   # never modified after this point

    # Two 1280-wide views side by side don't fit on a laptop screen, so the
    # window is shown scaled down; clicks are mapped back to full size.
    MAX_DISPLAY_W = 1800
    show_scale = min(1.0, MAX_DISPLAY_W / base_canvas.shape[1])

    # original -> displayed coordinate scale factors (for reading clicks back)
    scale0 = view0.shape[1] / view0r.shape[1]
    scale1 = view1.shape[1] / view1r.shape[1]
    vscale0 = view0.shape[0] / view0r.shape[0]
    vscale1 = view1.shape[0] / view1r.shape[0]

    # displayed -> original inverse (for drawing highlights back onto the canvas)
    disp_scale0 = view0r.shape[1] / view0.shape[1]
    disp_scale1 = view1r.shape[1] / view1.shape[1]
    disp_vscale0 = view0r.shape[0] / view0.shape[0]
    disp_vscale1 = view1r.shape[0] / view1.shape[0]

    def draw_box(canvas, side: str, seat: SeatROI, color, thickness=3, label=None):
        x1, y1, x2, y2 = seat.bbox
        if side == "cam0":
            dx1, dy1 = int(x1 * disp_scale0), int(y1 * disp_vscale0)
            dx2, dy2 = int(x2 * disp_scale0), int(y2 * disp_vscale0)
        else:
            dx1 = int(x1 * disp_scale1) + offset_x
            dx2 = int(x2 * disp_scale1) + offset_x
            dy1, dy2 = int(y1 * disp_vscale1), int(y2 * disp_vscale1)
        cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), color, thickness)
        if label:
            cv2.putText(canvas, label, (dx1 + 4, dy1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    print("\n[MapSharedSeat] Click a seat on the LEFT (cam0), then the matching "
          "seat on the RIGHT (cam1).")
    print("                Repeat for more shared desks. Press 'q' when done.\n")

    pairs: list[dict] = []
    confirmed_canvas = base_canvas.copy()   # accumulates permanent pair marks
    clicked_seat: dict[str, SeatROI | None] = {"cam0": None, "cam1": None}
    win_name = "Click matching seats  (cam0 left | cam1 right)  -  'q' to finish"

    def redraw():
        display = confirmed_canvas.copy()
        if clicked_seat["cam0"]:
            draw_box(display, "cam0", clicked_seat["cam0"], (0, 255, 255))   # yellow
        if clicked_seat["cam1"]:
            draw_box(display, "cam1", clicked_seat["cam1"], (0, 255, 255))   # yellow
        if show_scale < 1.0:
            display = cv2.resize(display, None, fx=show_scale, fy=show_scale,
                                 interpolation=cv2.INTER_AREA)
        cv2.imshow(win_name, display)

    def on_mouse(event, x, y, flags, param):
        nonlocal confirmed_canvas
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x, y = int(x / show_scale), int(y / show_scale)   # window -> full-size canvas

        if x < offset_x:
            orig_x, orig_y = int(x * scale0), int(y * vscale0)
            seat = find_clicked_seat(mgr0, orig_x, orig_y)
            if seat:
                clicked_seat["cam0"] = seat
                print(f"  cam0 seat selected: {seat.seat_id}")
        else:
            local_x = x - offset_x
            orig_x, orig_y = int(local_x * scale1), int(y * vscale1)
            seat = find_clicked_seat(mgr1, orig_x, orig_y)
            if seat:
                clicked_seat["cam1"] = seat
                print(f"  cam1 seat selected: {seat.seat_id}")

        if clicked_seat["cam0"] and clicked_seat["cam1"]:
            global_id = f"desk{len(pairs) + 1}"
            pairs.append({
                "global_id": global_id,
                "cameras": {
                    "cam0": clicked_seat["cam0"].seat_id,
                    "cam1": clicked_seat["cam1"].seat_id,
                },
            })
            # Burn a permanent green mark + label into confirmed_canvas so it
            # survives every future redraw for the rest of this session.
            draw_box(confirmed_canvas, "cam0", clicked_seat["cam0"], (0, 220, 120), label=global_id)
            draw_box(confirmed_canvas, "cam1", clicked_seat["cam1"], (0, 220, 120), label=global_id)
            print(f"  -> Paired as '{global_id}'. Click another pair, or 'q' to finish.\n")
            clicked_seat["cam0"] = None
            clicked_seat["cam1"] = None

        redraw()

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)
    redraw()

    while True:
        if (cv2.waitKey(20) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()

    if not pairs:
        print("[MapSharedSeat] No pairs made — nothing saved.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump({"shared_seats": pairs}, f, default_flow_style=False)

    print(f"[MapSharedSeat] Saved {len(pairs)} shared seat pair(s) -> {out_path}")
    print("                settings.yaml was not touched.")


if __name__ == "__main__":
    main()