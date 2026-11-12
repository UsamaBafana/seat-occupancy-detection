"""
label_ground_truth.py
Label seat occupancy on a RECORDED clip, frame-accurately, once.

Timestamps are CLIP TIME (seconds from the start of the video), the same
timeline the pipeline logs when its source is a video file. So you label a
clip once and can evaluate every pipeline run / setting against it.

Usage
-----
    # seat ids typed in
    python scripts/label_ground_truth.py --videos data/raw/cam0.mp4 --seats global_desk1 global_desk2 cam0_seat3

    # two synced clips side by side, seat ids read from a pipeline log of the same clips
    python scripts/label_ground_truth.py --videos data/raw/cam0.mp4 data/raw/cam1.mp4 --pred output/occupancy_123.csv

Controls (keep the video window focused)
----------------------------------------
    space        play / pause
    a / d        back / forward 1 frame
    j / l        back / forward 5 s
    [ / ]        slower / faster playback
    1-9          select seat
    o / v        selected seat is OCCUPIED / VACANT from this frame onward
    x            clear the selected seat's label at this frame (unlabelled from here)
    u            undo last change
    s            save
    q            save and quit

Tip: label the CHANGES only — set every seat's state on the first frame,
then jump through and press o / v at the frame a person actually sits down
or leaves. The dense ground-truth CSV is generated for you.

Outputs (next to --out, default output/)
----------------------------------------
    <clip>_gt_events.csv     your change points (reloaded automatically to continue)
    <clip>_ground_truth.csv  one row per seat every --step seconds, for
                             `python main.py evaluate --direction backward ...`
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

GT_COLUMNS = ["timestamp", "frame_id", "seat_id", "state", "confidence", "occupant_id"]
COLORS = {"occupied": (60, 60, 230), "vacant": (80, 200, 80), None: (90, 90, 90)}
PANEL_H_BASE = 70
ROW_H = 22


# ─────────────────────────────────────────────────────────────
#  Label store
# ─────────────────────────────────────────────────────────────

class Labels:
    """Per-seat change points: {seat_id: [(t, state|None), ...]} sorted by t."""

    def __init__(self, seat_ids: list[str]):
        self.seat_ids = list(seat_ids)
        self.events: dict[str, list[tuple[float, str | None]]] = {s: [] for s in seat_ids}
        self.history: list[tuple[str, float, tuple | None]] = []  # for undo

    def set(self, seat: str, t: float, state: str | None) -> None:
        ev = self.events[seat]
        prev = next((e for e in ev if abs(e[0] - t) < 1e-6), None)
        self.history.append((seat, t, prev))
        ev[:] = [e for e in ev if abs(e[0] - t) >= 1e-6]
        ev.append((t, state))
        ev.sort(key=lambda e: e[0])

    def undo(self) -> str | None:
        if not self.history:
            return None
        seat, t, prev = self.history.pop()
        ev = self.events[seat]
        ev[:] = [e for e in ev if abs(e[0] - t) >= 1e-6]
        if prev is not None:
            ev.append(prev)
            ev.sort(key=lambda e: e[0])
        return seat

    def state_at(self, seat: str, t: float) -> str | None:
        state = None
        for et, es in self.events[seat]:
            if et <= t + 1e-6:
                state = es
            else:
                break
        return state

    # ── persistence ──
    def save_events(self, path: Path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seat_id", "time_s", "state"])
            for seat in self.seat_ids:
                for t, s in self.events[seat]:
                    w.writerow([seat, f"{t:.3f}", s or ""])

    def load_events(self, path: Path) -> int:
        n = 0
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                seat = row["seat_id"]
                if seat not in self.events:
                    self.seat_ids.append(seat)
                    self.events[seat] = []
                self.events[seat].append((float(row["time_s"]), row["state"] or None))
                n += 1
        for ev in self.events.values():
            ev.sort(key=lambda e: e[0])
        return n

    def save_dense(self, path: Path, duration: float, fps: float, step: float) -> int:
        rows = 0
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=GT_COLUMNS)
            w.writeheader()
            t = 0.0
            while t <= duration + 1e-6:
                for seat in self.seat_ids:
                    s = self.state_at(seat, t)
                    if s is None:          # unlabelled stretch — skip
                        continue
                    w.writerow({
                        "timestamp": round(t, 3),
                        "frame_id": int(round(t * fps)),
                        "seat_id": seat,
                        "state": s,
                        "confidence": 1.0,
                        "occupant_id": "",
                    })
                    rows += 1
                t += step
        return rows


# ─────────────────────────────────────────────────────────────
#  Video helpers
# ─────────────────────────────────────────────────────────────

def seat_ids_from_pred(path: str) -> list[str]:
    seen = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sid = row.get("seat_id")
            if sid and sid not in seen:
                seen.append(sid)
    return sorted(seen)


def read_at(caps, idx: int):
    frames = []
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        frames.append(fr if ok else None)
    return frames


def compose(frames, height: int) -> np.ndarray:
    tiles = []
    for fr in frames:
        if fr is None:
            fr = np.zeros((height, int(height * 16 / 9), 3), np.uint8)
        h, w = fr.shape[:2]
        tiles.append(cv2.resize(fr, (int(w * height / h), height)))
    return np.hstack(tiles)


def draw_panel(width, labels: Labels, sel: int, t: float, idx: int, n_frames: int,
               fps: float, duration: float, playing: bool, speed: float, msg: str) -> np.ndarray:
    n = len(labels.seat_ids)
    h = PANEL_H_BASE + n * ROW_H + 16
    p = np.full((h, width, 3), 25, np.uint8)
    x0, x1 = 170, width - 20
    tl_w = max(1, x1 - x0)

    cv2.putText(p, f"t={t:7.2f}s  frame {idx}/{n_frames - 1}  {fps:.1f}fps  "
                   f"{'PLAY' if playing else 'PAUSE'} x{speed:g}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.putText(p, "space play | a/d frame | j/l 5s | 1-9 seat | o occ | v vac | x clear | u undo | s save | q quit",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
    if msg:
        cv2.putText(p, msg, (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

    for i, seat in enumerate(labels.seat_ids):
        y = PANEL_H_BASE - 8 + i * ROW_H
        cur = labels.state_at(seat, t)
        tag = f"{i + 1 if i < 9 else ' '} {seat[:16]}"
        col = (0, 220, 255) if i == sel else (210, 210, 210)
        cv2.putText(p, tag, (10, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        # timeline segments
        ev = labels.events[seat]
        bounds = [(0.0, None)] + ev
        for k, (st, ss) in enumerate(bounds):
            et = bounds[k + 1][0] if k + 1 < len(bounds) else duration
            if et <= st:
                continue
            a = x0 + int(st / duration * tl_w)
            b = x0 + int(et / duration * tl_w)
            cv2.rectangle(p, (a, y + 3), (max(a + 1, b), y + 17), COLORS[ss], -1)
        cv2.putText(p, cur or "-", (x0 - 62, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS[cur], 1)

    cx = x0 + int(t / duration * tl_w)
    cv2.line(p, (cx, PANEL_H_BASE - 10), (cx, PANEL_H_BASE - 8 + n * ROW_H), (255, 255, 255), 1)
    return p


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Frame-accurate seat occupancy labelling on recorded clips")
    ap.add_argument("--videos", nargs="+", required=True,
                    help="One clip, or several clips recorded together (shown side by side)")
    ap.add_argument("--seats", nargs="+", default=None, help="Seat ids exactly as the pipeline logs them")
    ap.add_argument("--pred", default=None, help="A pipeline occupancy CSV to read seat ids from")
    ap.add_argument("--step", type=float, default=1.0, help="Seconds between ground-truth rows (default 1)")
    ap.add_argument("--height", type=int, default=480, help="Display height per clip")
    ap.add_argument("--out", default="output", help="Output folder")
    args = ap.parse_args()

    seat_ids = list(args.seats or [])
    if args.pred:
        for s in seat_ids_from_pred(args.pred):
            if s not in seat_ids:
                seat_ids.append(s)
    if not seat_ids:
        ap.error("give seat ids with --seats, or a pipeline log with --pred")

    caps = [cv2.VideoCapture(v) for v in args.videos]
    for v, c in zip(args.videos, caps):
        if not c.isOpened():
            raise SystemExit(f"Cannot open {v}")
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps))
    if n_frames <= 0:
        raise SystemExit("Could not read frame count from the clip(s).")
    duration = max(1e-3, (n_frames - 1) / fps)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.videos[0]).stem
    events_path = out_dir / f"{stem}_gt_events.csv"
    dense_path = out_dir / f"{stem}_ground_truth.csv"

    labels = Labels(seat_ids)
    msg = ""
    if events_path.exists():
        n = labels.load_events(events_path)
        msg = f"Loaded {n} existing change point(s) from {events_path.name}"

    print(f"[Label] {len(args.videos)} clip(s), {n_frames} frames @ {fps:.2f} fps ({duration:.1f}s)")
    print(f"[Label] seats: {labels.seat_ids}")

    win = "Ground truth labeller"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    idx, sel, playing, speed = 0, 0, False, 1.0
    frames = read_at(caps, idx)
    dirty = False

    def save():
        labels.save_events(events_path)
        rows = labels.save_dense(dense_path, duration, fps, args.step)
        return f"Saved {rows} rows -> {dense_path.name}"

    while True:
        t = idx / fps
        view = compose(frames, args.height)
        panel = draw_panel(view.shape[1], labels, sel, t, idx, n_frames, fps,
                           duration, playing, speed, msg)
        cv2.imshow(win, np.vstack([view, panel]))

        delay = max(1, int(1000 / (fps * speed))) if playing else 30
        key = cv2.waitKey(delay) & 0xFF
        new_idx = idx

        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("d"):
            new_idx, playing = idx + 1, False
        elif key == ord("a"):
            new_idx, playing = idx - 1, False
        elif key == ord("l"):
            new_idx = idx + int(5 * fps)
        elif key == ord("j"):
            new_idx = idx - int(5 * fps)
        elif key == ord("]"):
            speed = min(16.0, speed * 2)
        elif key == ord("["):
            speed = max(0.125, speed / 2)
        elif ord("1") <= key <= ord("9"):
            if key - ord("1") < len(labels.seat_ids):
                sel = key - ord("1")
        elif key in (ord("o"), ord("v"), ord("x")):
            state = {"o": "occupied", "v": "vacant", "x": None}[chr(key)]
            seat = labels.seat_ids[sel]
            labels.set(seat, t, state)
            dirty = True
            msg = f"{seat} -> {state or 'unlabelled'} from {t:.2f}s"
        elif key == ord("u"):
            seat = labels.undo()
            dirty = dirty or seat is not None
            msg = f"Undid last change on {seat}" if seat else "Nothing to undo"
        elif key == ord("s"):
            msg = save()
            dirty = False
        elif playing:
            new_idx = idx + 1

        new_idx = max(0, min(n_frames - 1, new_idx))
        if new_idx != idx:
            if playing and new_idx == idx + 1:
                frames = [c.read()[1] for c in caps]        # sequential read is fast
            else:
                frames = read_at(caps, new_idx)
            idx = new_idx
            if idx == n_frames - 1:
                playing = False

    print("[Label]", save() if (dirty or not dense_path.exists()) else f"No changes; {dense_path} unchanged")
    print(f"[Label] change points: {events_path}")
    print("\nEvaluate a pipeline run on this clip with:")
    print(f"  python main.py evaluate --gt {dense_path} --pred output/occupancy_<run>.csv "
          f"--direction backward --tolerance 10")
    for c in caps:
        c.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
