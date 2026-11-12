import sys
from pathlib import Path

"""
test_autodetect.py
Quick standalone check for SeatManager auto-detection.


Live (shows the camera in real time, keeps chairs seen consistently — recommended):
  python scripts/test_autodetect.py --source 0 --live
  python scripts/test_autodetect.py --source 0 --live --frames 40 --min-ratio 0.6
  press 'q' in the window to stop sampling early

Save the result so a normal run can load it with --seats:
  python scripts/test_autodetect.py --source 0 --live --save output/seats.json

CAMERA CONFIRMATION
-------------------
Windows can silently reassign which physical camera is index 0 / 1
between separate process launches — a check done in one script run
(e.g. list_cameras.py) is NOT guaranteed to match what a DIFFERENT
script run opens a moment later, even with the same backend. So before
detection starts, this script opens the requested camera index and
shows a quick live preview: press 'c' to confirm it's the camera you
expect, or 'q' to abort. Because this happens in the SAME process
invocation as the detection that follows, there's no gap for Windows
to reorder anything in between — what you see in the confirm window
is guaranteed to be what gets used for detection right after.
"""

import argparse

import cv2

# Make the project root importable when this file is run as `python scripts/<name>.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.seat_manager import SeatManager


def grab_frame(source: str, warmup: int = 30):
    """Read one frame from a webcam index, video path, or image path.

    For live cameras, discard the first `warmup` frames so auto-exposure /
    white-balance can settle before we keep one.
    """
    if not source.isdigit():
        img = cv2.imread(source)
        if img is not None:          # it was an image file
            return img

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source, cv2.CAP_DSHOW if source.isdigit() else 0)
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


def confirm_camera(source: str) -> bool:
    """
    Show a quick live preview of `source` and require the user to press
    'c' to continue before any detection runs. Only applies to webcam
    indices — video/image file sources are passed through unconfirmed
    since there's no ambiguity there.

    Returns True if confirmed, False if aborted.
    """
    if not source.isdigit():
        return True

    cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    print(f"\n[Confirm] Previewing camera index {source} — "
          f"press 'c' if this is the camera you expect, 'q' to abort.\n")

    confirmed = False
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        display = frame.copy()
        cv2.rectangle(display, (0, 0), (display.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(display, f"Source {source}  —  'c'=confirm   'q'=abort",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)
        cv2.imshow("Confirm camera before detecting", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            confirmed = True
            break
        if key == ord("q"):
            confirmed = False
            break

    cap.release()
    cv2.destroyWindow("Confirm camera before detecting")
    return confirmed


def main():
    ap = argparse.ArgumentParser(description="Test chair-based seat auto-detection")
    ap.add_argument("--source",   default="0",
                    help="webcam index (0), video path, or image path")
    ap.add_argument("--weights",  default="yolov8n.pt",
                    help="COCO-pretrained weights — NOT best_v4.pt")
    ap.add_argument("--conf",     type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--save",     default=None,
                    help="optional path to save draft seats (e.g. output/seats.json)")
    # live options
    ap.add_argument("--live",      action="store_true",
                    help="watch live video instead of one snapshot")
    ap.add_argument("--frames",    type=int, default=30, help="(live) frames to sample")
    ap.add_argument("--min-ratio", type=float, default=0.60,
                    help="(live) keep chairs seen in >= this fraction of frames")
    ap.add_argument("--match-iou", type=float, default=0.50,
                    help="(live) IoU to treat boxes across frames as the same chair")
    ap.add_argument("--min-area",  type=int, default=1500,
                    help="ignore chair boxes smaller than this many pixels")
    ap.add_argument("--skip-confirm", action="store_true",
                    help="skip the live camera-confirmation preview")
    args = ap.parse_args()

    if not args.skip_confirm:
        if not confirm_camera(args.source):
            print("[Aborted] Camera not confirmed — nothing was detected or saved.")
            return

    mgr = SeatManager()

    if args.live:
        # The live method opens its own preview window and shows a final
        # freeze-frame, so there is nothing more to display here.
        seats = mgr.auto_detect_seats_live(
            source=args.source,
            weights=args.weights,
            confidence=args.conf,
            frames=args.frames,
            min_ratio=args.min_ratio,
            match_iou=args.match_iou,
            min_area=args.min_area,
        )
    else:
        frame = grab_frame(args.source)
        seats = mgr.auto_detect_seats(frame, weights=args.weights,
                                      confidence=args.conf, min_area=args.min_area)
        annotated = mgr.draw_on_frame(frame.copy())
        cv2.imshow("Auto-detected seats — press any key to close", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if seats and args.save:
        mgr.save_to_json(args.save)
        print(f"Saved {len(seats)} draft seat(s) -> {args.save}")


if __name__ == "__main__":
    main()