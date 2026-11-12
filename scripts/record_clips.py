"""
record_clips.py
Record several webcams at the same time to MP4, for repeatable offline runs.

Each camera's real frame rate is measured first and the file is written at
that rate, so the saved clip plays back in real time and can be fed to the
pipeline, benchmark.py or label_ground_truth.py later.

Usage (from the project root):
    python scripts/record_clips.py --cams 0 1 --seconds 300 --name 3person
    python scripts/record_clips.py --cams 0 1 --width 1280 --height 720 --preview

Output: recordings/<name>_cam1.mp4, <name>_cam2.mp4, ... numbered in --cams order
(folder set with --out).
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2

BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY   # same backend as the pipeline
SETTLE_S = 2.0      # let exposure/white balance settle before measuring
MEASURE_S = 3.0     # seconds used to measure each camera's real FPS


class CameraRecorder(threading.Thread):
    def __init__(self, label, source, out_path, args, start_evt, stop_evt):
        super().__init__(daemon=True)
        self.label = label
        self.source = source
        self.out_path = out_path
        self.args = args
        self.start_evt = start_evt
        self.stop_evt = stop_evt
        self.ready = threading.Event()
        self.error = None
        self.fps = None
        self.size = None
        self.frames_written = 0
        self.read_failures = 0
        self.t_first = None
        self.t_last = None
        self.latest = None            # latest frame for the preview window
        self.lock = threading.Lock()

    def run(self):
        cap = None
        writer = None
        try:
            cap = cv2.VideoCapture(self.source, BACKEND)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open camera {self.source}")
            if self.args.width and self.args.height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)

            # Settle, then measure the real frame rate (the reported one is often wrong)
            t_end = time.time() + SETTLE_S
            while time.time() < t_end:
                cap.read()
            n, t0 = 0, time.time()
            frame = None
            while time.time() - t0 < MEASURE_S:
                ok, f = cap.read()
                if ok:
                    frame = f
                    n += 1
            if frame is None or n < 5:
                raise RuntimeError(f"camera {self.source} is not delivering frames")
            self.fps = n / (time.time() - t0)
            h, w = frame.shape[:2]
            self.size = (w, h)

            writer = cv2.VideoWriter(str(self.out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     round(self.fps, 2), self.size)
            if not writer.isOpened():
                raise RuntimeError(f"cannot create {self.out_path}")
        except Exception as exc:
            self.error = str(exc)
            self.ready.set()
            if cap is not None:
                cap.release()
            return

        self.ready.set()

        # Keep reading while waiting so the camera buffer stays fresh
        while not self.start_evt.is_set() and not self.stop_evt.is_set():
            cap.read()

        consecutive_fail = 0
        while not self.stop_evt.is_set():
            ok, frame = cap.read()
            if not ok:
                self.read_failures += 1
                consecutive_fail += 1
                if consecutive_fail > 50:
                    self.error = "camera stopped delivering frames"
                    break
                continue
            consecutive_fail = 0
            now = time.time()
            if self.t_first is None:
                self.t_first = now
            self.t_last = now
            writer.write(frame)
            self.frames_written += 1
            with self.lock:
                self.latest = frame

        writer.release()
        cap.release()

    def actual_fps(self):
        if self.t_first is None or self.t_last is None or self.frames_written < 2:
            return None
        return (self.frames_written - 1) / (self.t_last - self.t_first)


def parse_args():
    p = argparse.ArgumentParser(description="Record several webcams at the same time")
    p.add_argument("--cams", nargs="+", default=["0", "1"],
                   help="Camera indices (or video paths/URLs), e.g. 0 1")
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--name", default="", help="File name prefix, e.g. empty or 1person")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--preview", action="store_true", help="Show a live preview (q to stop)")
    p.add_argument("--delay", type=float, default=5, help="Countdown before recording starts")
    p.add_argument("--out", default="recordings")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.name}_" if args.name else ""

    start_evt = threading.Event()
    stop_evt = threading.Event()
    recorders = []
    for i, cam in enumerate(args.cams, start=1):
        path = out_dir / f"{prefix}cam{i}.mp4"
        if path.exists() and not args.overwrite:
            sys.exit(f"{path} already exists. Use a different --name or add --overwrite.")
        source = int(cam) if cam.isdigit() else cam
        recorders.append(CameraRecorder(f"cam{i}", source, path, args, start_evt, stop_evt))

    print(f"Opening {len(recorders)} camera(s) and measuring their frame rates...")
    for r in recorders:
        r.start()
    for r in recorders:
        r.ready.wait(timeout=60)

    failed = [r for r in recorders if r.error or not r.ready.is_set()]
    for r in recorders:
        if r.error:
            print(f"  {r.label} (camera {r.source}): ERROR - {r.error}")
        elif r.ready.is_set():
            print(f"  {r.label} (camera {r.source}): {r.size[0]}x{r.size[1]} @ {r.fps:.1f} fps "
                  f"-> {r.out_path}")
    if failed:
        stop_evt.set()
        sys.exit("Fix the camera problem above and try again "
                 "(check the index with list_cameras.py / identify_cameras.py).")

    for s in range(int(args.delay), 0, -1):
        print(f"  Starting in {s}...", end="\r", flush=True)
        time.sleep(1)
    start_evt.set()
    t_start = time.time()
    print(" " * 30, end="\r")
    print(f"RECORDING for {args.seconds:.0f} s" + (" - press q in the preview to stop early"
                                                   if args.preview else " - Ctrl+C to stop early"))

    try:
        while time.time() - t_start < args.seconds:
            if any(r.error for r in recorders):
                break
            if args.preview:
                tiles = []
                for r in recorders:
                    with r.lock:
                        f = None if r.latest is None else r.latest.copy()
                    if f is not None:
                        f = cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1])))
                        cv2.putText(f, r.label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
                        tiles.append(f)
                if tiles and all(t.shape[0] == tiles[0].shape[0] for t in tiles):
                    cv2.imshow("record_clips (q = stop)", cv2.hconcat(tiles))
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            else:
                left = args.seconds - (time.time() - t_start)
                print(f"  {left:5.0f} s left", end="\r", flush=True)
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    stop_evt.set()
    for r in recorders:
        r.join(timeout=10)
    if args.preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\nDone - recorded {elapsed:.0f} s.")
    for r in recorders:
        real = r.actual_fps()
        line = f"  {r.out_path}: {r.frames_written} frames"
        if real:
            line += f", real {real:.1f} fps (saved as {r.fps:.1f})"
            if abs(real - r.fps) / r.fps > 0.05:
                line += (f"\n    WARNING: frame rate drifted. When benchmarking this clip add "
                         f"--src-fps {real:.1f}")
        if r.read_failures:
            line += f", {r.read_failures} dropped reads"
        if r.error:
            line += f"  [ERROR: {r.error}]"
        print(line)


if __name__ == "__main__":
    main()
