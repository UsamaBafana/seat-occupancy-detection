"""
chair_bbox_margin.py
Measures the margin of error on the chair bounding box due to chair
movement, so seat-matching tolerances (overlap_threshold, match_iou in
SeatManager / config/settings.yaml) can be justified with a real number
instead of a guess.

WHY THIS EXISTS
----------------
Seat ROIs are captured once (drawn, or auto-detected from chairs) and then
reused as a FIXED box on every later frame. In reality chairs get nudged,
pushed in/out, or rotated slightly between sits — and even a chair nobody
touches doesn't land on the exact same pixel box every time YOLO detects
it (small jitter from lighting, NMS, etc). This script quantifies both
effects together as one empirical "margin of error": how far a chair's
detected box actually drifts from its own mean position across a sampling
window.

WHAT IT DOES
------------
1. Opens a webcam / video source and samples N frames (same
   confirm-camera + warmup pattern as test_autodetect.py).
2. Runs the COCO chair detector (class 56) on each sampled frame.
3. Groups per-frame chair boxes into tracks using the SAME IoU-matching
   SeatManager.auto_detect_seats_live() already uses (--match-iou), so
   the numbers below are directly comparable to that parameter.
4. For every chair track, computes across its sampled boxes:
     - center drift (px and % of box diagonal): mean / max / 95th pct
     - width & height variation (px and % of mean size)
     - IoU of every frame's box against the track's own mean box
       (this is the same quantity overlap_threshold / match_iou are
       tuned against — a chair that never moved would still show
       IoU < 1.0 here purely from detector jitter)
   The worst (lowest) per-track IoU and the largest center drift are the
   two headline "margin of error" numbers.
5. Prints a report and (optionally) saves it as JSON + a matplotlib PNG
   of each track's box drift over the sampling window.

USAGE (solo-friendly two-phase workflow)
-----------------------------------------
Since this only needs one person, don't try to time a mid-run nudge —
run it in two separate steps instead, with the physical chair-nudging
done in between while nothing is sampling:

  1. python scripts/chair_bbox_margin.py --source 0 --phase before
     (samples the chair at rest — this is your noise-floor baseline)

  2. Walk over, nudge/push the chair a realistic amount (however far it
     really drifts when someone gets up or pushes it in), then walk back.

  3. python scripts/chair_bbox_margin.py --source 0 --phase after
     (samples again; automatically loads the "before" report and prints
     how far the chair actually moved between phases, compared against
     each phase's own internal detector jitter as the noise floor)

Or, for a quick one-shot baseline-jitter-only reading with no comparison:
  python scripts/chair_bbox_margin.py --source 0 --phase single

Press 'q' in the preview window to stop sampling early.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

# Make the project root importable when this file is run as `python scripts/<name>.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.seat_manager import SeatManager

# ─────────────────────────────────────────────────────────────
#  Camera helpers (same pattern as test_autodetect.py)
# ─────────────────────────────────────────────────────────────

def confirm_camera(source: str) -> bool:
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
        cv2.imshow("Confirm camera before measuring", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            confirmed = True
            break
        if key == ord("q"):
            confirmed = False
            break

    cap.release()
    cv2.destroyWindow("Confirm camera before measuring")
    return confirmed


# ─────────────────────────────────────────────────────────────
#  Core measurement
# ─────────────────────────────────────────────────────────────

def measure(
    source: str,
    weights: str,
    confidence: float,
    frames: int,
    warmup: int,
    sample_every: int,
    match_iou: float,
    min_area: int,
    min_ratio: float,
    preview: bool,
):
    mgr = SeatManager()
    detector = mgr._ensure_chair_detector(None, weights, confidence, "cpu", 640, "chair")
    if detector is None:
        return None

    cap = cv2.VideoCapture(
        int(source) if source.isdigit() else source,
        cv2.CAP_DSHOW if source.isdigit() else 0,
    )
    if not cap.isOpened():
        print(f"[Margin] Cannot open source: {source}")
        return None

    win = "Measuring chair bbox margin of error"
    if preview:
        cv2.namedWindow(win)

    for _ in range(max(0, warmup)):
        ok, frame = cap.read()
        if preview and ok:
            view = frame.copy()
            mgr._banner(view, "Warming up camera...")
            cv2.imshow(win, view)
            cv2.waitKey(1)

    tracks: list[dict] = []
    sampled = 0
    read_idx = 0
    stopped_early = False

    print(f"[Margin] Sampling {frames} frame(s) (1 in every {sample_every}). "
          f"Press 'q' to stop early.")

    while sampled < frames:
        ok, frame = cap.read()
        if not ok:
            break
        read_idx += 1

        if read_idx % max(1, sample_every) == 0:
            sampled += 1

            for box in mgr._chair_boxes(detector.detect(frame), min_area):
                best, best_iou = None, 0.0
                for t in tracks:
                    iou = mgr._box_iou(box, t["mean"])
                    if iou > best_iou:
                        best, best_iou = t, iou
                if best is not None and best_iou >= match_iou:
                    best["boxes"].append(box)
                    best["frame_idx"].append(sampled)
                    best["frames_seen"].add(sampled)
                    best["mean"] = mgr._mean_box(best["boxes"])
                else:
                    tracks.append({
                        "boxes": [box], "frame_idx": [sampled],
                        "frames_seen": {sampled}, "mean": box,
                    })

            if sampled % 10 == 0:
                print(f"  ... {sampled}/{frames} frames sampled, {len(tracks)} candidate(s)")

        if preview:
            view = frame.copy()
            for t in tracks:
                x1, y1, x2, y2 = t["mean"]
                strong = sampled > 0 and len(t["frames_seen"]) / sampled >= min_ratio
                color = (0, 200, 80) if strong else (0, 165, 255)
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
            mgr._banner(view, f"Sampling {sampled}/{frames}  |  q=stop")
            cv2.imshow(win, view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                stopped_early = True
                break

    cap.release()
    if preview:
        cv2.destroyWindow(win)

    if sampled == 0:
        print("[Margin] No frames captured.")
        return None

    kept = [t for t in tracks if len(t["frames_seen"]) / sampled >= min_ratio]
    if not kept:
        print("[Margin] No chair was seen consistently enough — lower "
              "--min-ratio/--conf or check the scene.")
        return None

    report = {
        "sampled_frames": sampled,
        "stopped_early": stopped_early,
        "match_iou": match_iou,
        "min_ratio": min_ratio,
        "chairs": [],
    }

    for i, t in enumerate(kept):
        boxes = t["boxes"]
        mean_box = t["mean"]
        mx1, my1, mx2, my2 = mean_box
        mean_cx, mean_cy = (mx1 + mx2) / 2, (my1 + my2) / 2
        mean_w, mean_h = max(1, mx2 - mx1), max(1, my2 - my1)
        diag = (mean_w ** 2 + mean_h ** 2) ** 0.5

        ious, drifts, dws, dhs = [], [], [], []
        for b in boxes:
            x1, y1, x2, y2 = b
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            drifts.append(((cx - mean_cx) ** 2 + (cy - mean_cy) ** 2) ** 0.5)
            dws.append(abs((x2 - x1) - mean_w))
            dhs.append(abs((y2 - y1) - mean_h))
            ious.append(mgr._box_iou(b, mean_box))

        drifts_sorted = sorted(drifts)
        p95_idx = min(len(drifts_sorted) - 1, int(round(0.95 * (len(drifts_sorted) - 1))))

        chair_report = {
            "chair_id": f"chair_{i:02d}",
            "mean_box": [mx1, my1, mx2, my2],
            "seen_frames": len(t["frames_seen"]),
            "seen_ratio": round(len(t["frames_seen"]) / sampled, 3),
            "center_drift_px": {
                "mean": round(statistics.fmean(drifts), 1),
                "max": round(max(drifts), 1),
                "p95": round(drifts_sorted[p95_idx], 1),
                "pct_of_box_diagonal_max": round(100 * max(drifts) / diag, 1),
            },
            "width_variation_px": {"mean": round(statistics.fmean(dws), 1), "max": round(max(dws), 1)},
            "height_variation_px": {"mean": round(statistics.fmean(dhs), 1), "max": round(max(dhs), 1)},
            "iou_vs_own_mean": {
                "min": round(min(ious), 3),
                "mean": round(statistics.fmean(ious), 3),
            },
            "_raw_boxes": boxes,
            "_raw_frame_idx": t["frame_idx"],
        }
        report["chairs"].append(chair_report)

    return report


# ─────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────

def print_report(report: dict, overlap_threshold: float) -> None:
    print("\n" + "=" * 60)
    print("  CHAIR BOUNDING BOX — MARGIN OF ERROR REPORT")
    print("=" * 60)
    print(f"  Frames sampled : {report['sampled_frames']}"
          f"{'  (stopped early)' if report['stopped_early'] else ''}")
    print(f"  match_iou used : {report['match_iou']}")
    print("-" * 60)

    worst_iou = 1.0
    worst_drift_pct = 0.0

    for c in report["chairs"]:
        print(f"\n  {c['chair_id']}  (seen {c['seen_ratio']*100:.0f}% of frames, "
              f"mean box {c['mean_box']})")
        print(f"    center drift : mean {c['center_drift_px']['mean']}px, "
              f"p95 {c['center_drift_px']['p95']}px, "
              f"max {c['center_drift_px']['max']}px "
              f"({c['center_drift_px']['pct_of_box_diagonal_max']}% of box diagonal)")
        print(f"    size wobble  : width ±{c['width_variation_px']['max']}px, "
              f"height ±{c['height_variation_px']['max']}px (max)")
        print(f"    IoU vs its own mean box : min {c['iou_vs_own_mean']['min']}, "
              f"mean {c['iou_vs_own_mean']['mean']}")

        worst_iou = min(worst_iou, c["iou_vs_own_mean"]["min"])
        worst_drift_pct = max(worst_drift_pct, c["center_drift_px"]["pct_of_box_diagonal_max"])

    print("\n" + "-" * 60)
    print(f"  Worst-case IoU against a chair's own mean box : {worst_iou:.3f}")
    print(f"  Current config overlap_threshold              : {overlap_threshold}")
    if worst_iou < overlap_threshold:
        print(f"  --> A chair's OWN natural jitter/movement can drop its IoU below "
              f"overlap_threshold. Recommend padding seat ROIs by roughly "
              f"{worst_drift_pct:.0f}% (~{max(c['center_drift_px']['max'] for c in report['chairs']):.0f}px) "
              f"or lowering overlap_threshold.")
    else:
        print(f"  --> overlap_threshold has margin: worst observed jitter "
              f"(IoU {worst_iou:.3f}) stays above it.")
    print("=" * 60 + "\n")


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def compare_phases(before: dict, after: dict) -> list[dict]:
    """
    Match each 'after' chair to its nearest 'before' chair by center
    distance, and report how far it actually moved between phases —
    the real chair-movement number — next to each phase's own internal
    jitter (the noise floor a stationary chair still shows).

    Returns one result dict per matched chair (used for --trial logging).
    """
    def center(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    print("\n" + "=" * 60)
    print("  BEFORE -> AFTER COMPARISON  (actual chair movement)")
    print("=" * 60)

    results: list[dict] = []
    used = set()
    for a in after["chairs"]:
        acx, acy = center(a["mean_box"])
        best, best_dist = None, float("inf")
        for j, b in enumerate(before["chairs"]):
            if j in used:
                continue
            bcx, bcy = center(b["mean_box"])
            dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
            if dist < best_dist:
                best, best_dist, best_j = b, dist, j

        if best is None:
            print(f"\n  {a['chair_id']} (after): no 'before' chair to compare against.")
            continue
        used.add(best_j)

        bx1, by1, bx2, by2 = best["mean_box"]
        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
        diag = (bw ** 2 + bh ** 2) ** 0.5
        move_pct = 100 * best_dist / diag
        before_jitter = best["center_drift_px"]["max"]
        after_jitter = a["center_drift_px"]["max"]
        noise_floor = max(before_jitter, after_jitter)
        iou_before_after = _iou(best["mean_box"], a["mean_box"])
        real_movement = best_dist > noise_floor

        print(f"\n  before mean box {best['mean_box']}  ->  after mean box {a['mean_box']}")
        print(f"    moved {best_dist:.1f}px between phases ({move_pct:.0f}% of box diagonal)")
        print(f"    IoU (before-mean vs after-mean)                 : {iou_before_after:.3f}")
        print(f"    noise floor (max internal jitter, either phase) : {noise_floor:.1f}px")
        if not real_movement:
            print("    --> movement is within normal detector jitter — "
                  "treat this as \"didn't meaningfully move.\"")
        else:
            print(f"    --> movement of {best_dist:.1f}px exceeds the {noise_floor:.1f}px noise floor: "
                  f"this is real chair movement, not detector noise. "
                  f"Recommend seat-ROI padding of at least {best_dist:.0f}px "
                  f"(~{move_pct:.0f}% of box size).")

        results.append({
            "chair_id": a["chair_id"],
            "moved_px": round(best_dist, 2),
            "move_pct": round(move_pct, 2),
            "noise_floor_px": round(noise_floor, 2),
            "iou_before_after": round(iou_before_after, 4),
            "real_movement": real_movement,
        })

    print("=" * 60 + "\n")
    return results


# ─────────────────────────────────────────────────────────────
#  Trial log — accumulate multiple before/after runs into one
#  defensible margin-of-error number instead of a single anecdote.
# ─────────────────────────────────────────────────────────────

def log_trial(results: list[dict], tag: str, path: str = "output/chair_bbox_trials.json") -> list[dict]:
    import datetime

    out = Path(path)
    history: list[dict] = []
    if out.exists():
        try:
            history = json.loads(out.read_text())
        except Exception:
            history = []

    trial_num = len(history) + 1
    for r in results:
        history.append({
            "trial": trial_num,
            "tag": tag or f"trial_{trial_num}",
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            **r,
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(history, indent=2))
    return history


def print_margin_of_error(history: list[dict], overlap_threshold: float, min_trials: int = 3) -> None:
    n = len(history)
    moved = [h["moved_px"] for h in history]
    ious = [h["iou_before_after"] for h in history]
    noise_floors = [h["noise_floor_px"] for h in history]

    print("\n" + "=" * 60)
    print(f"  MARGIN OF ERROR — cumulative across {n} trial(s)")
    print("=" * 60)
    for h in history:
        flag = "REAL MOVE" if h["real_movement"] else "noise    "
        print(f"  trial {h['trial']:>2} [{h['tag']:<10}] {flag}  "
              f"moved {h['moved_px']:>5.1f}px  IoU {h['iou_before_after']:.3f}  "
              f"noise floor {h['noise_floor_px']:>5.1f}px")

    margin_px = max(moved + noise_floors)
    worst_iou = min(ious)

    print("-" * 60)
    print(f"  CONCRETE MARGIN OF ERROR (max observed, px)     : {margin_px:.1f}px")
    print(f"  Worst-case IoU across all trials                : {worst_iou:.3f}")
    print(f"  Config overlap_threshold                        : {overlap_threshold}")
    print(f"  Headroom (worst IoU - threshold)                : {worst_iou - overlap_threshold:.3f}")

    if n < min_trials:
        print(f"\n  --> Only {n}/{min_trials} trial(s) so far. Run at least {min_trials} "
              f"(vary the nudge: small / typical / large) before quoting this number "
              f"as a real margin of error — one run could be a fluke.")
    else:
        print(f"\n  --> {n} trials recorded (>= {min_trials}). Margin of error of "
              f"{margin_px:.1f}px is reportable: it is the largest chair-box displacement "
              f"seen across all trials, real movement or detector noise, whichever was larger.")
    print("=" * 60 + "\n")


def save_json(report: dict, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Margin] Saved full report -> {out}")


def save_plot(report: dict, path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Margin] matplotlib not available, skipping plot.")
        return

    n = len(report["chairs"])
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), squeeze=False)

    for i, c in enumerate(report["chairs"]):
        ax = axes[i][0]
        boxes = np.array(c["_raw_boxes"])
        idx = c["_raw_frame_idx"]
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        mx1, my1, mx2, my2 = c["mean_box"]
        mean_cx, mean_cy = (mx1 + mx2) / 2, (my1 + my2) / 2

        ax.plot(idx, cx - mean_cx, label="center x drift (px)", marker=".")
        ax.plot(idx, cy - mean_cy, label="center y drift (px)", marker=".")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_title(f"{c['chair_id']} — box center drift from its own mean")
        ax.set_xlabel("sampled frame #")
        ax.set_ylabel("drift (px)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[Margin] Saved plot -> {out}")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Measure chair bounding-box margin of error (jitter + movement)"
    )
    ap.add_argument("--source", default="0", help="webcam index (0) or video path")
    ap.add_argument("--weights", default="yolov8n.pt",
                     help="COCO-pretrained weights — NOT best_v4.pt (no 'chair' class)")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    ap.add_argument("--frames", type=int, default=150, help="frames to sample")
    ap.add_argument("--warmup", type=int, default=30, help="frames to discard while camera settles")
    ap.add_argument("--sample-every", type=int, default=2, help="run detection on 1 of every N frames read")
    ap.add_argument("--match-iou", type=float, default=0.50,
                     help="IoU to treat boxes across frames as the same physical chair")
    ap.add_argument("--min-area", type=int, default=1500, help="ignore chair boxes smaller than this")
    ap.add_argument("--min-ratio", type=float, default=0.50,
                     help="only report chairs seen in at least this fraction of sampled frames")
    ap.add_argument("--overlap-threshold", type=float, default=0.30,
                     help="value from config/settings.yaml, for the comparison line in the report")
    ap.add_argument("--phase", choices=["single", "before", "after"], default="single",
                     help="'before'/'after' = two-phase solo workflow with an automatic "
                          "movement comparison; 'single' = one baseline-jitter reading only")
    ap.add_argument("--trial-tag", default="",
                     help="label for this trial when using --phase after, e.g. 'small'/'typical'/'large' "
                          "nudge — logged to output/chair_bbox_trials.json for the cumulative margin-of-error report")
    ap.add_argument("--min-trials", type=int, default=3,
                     help="how many logged trials are required before the margin-of-error number is flagged reportable")
    ap.add_argument("--no-preview", action="store_true", help="run without the live preview window")
    ap.add_argument("--skip-confirm", action="store_true", help="skip the camera-confirmation preview")
    ap.add_argument("--save", default=None,
                     help="path to save the JSON report (default: output/chair_bbox_margin_<phase>.json)")
    ap.add_argument("--plot", default=None,
                     help="path to save the drift plot (default: output/chair_bbox_margin_<phase>.png, "
                          "'' to skip)")
    args = ap.parse_args()

    save_path = args.save or f"output/chair_bbox_margin_{args.phase}.json"
    plot_path = args.plot if args.plot is not None else f"output/chair_bbox_margin_{args.phase}.png"

    if not args.skip_confirm and not confirm_camera(args.source):
        print("[Aborted] Camera not confirmed.")
        return

    if args.phase == "before":
        print("[Margin] Phase 1/2: sampling the chair AT REST. "
              "Don't touch it yet — this is your baseline.")
    elif args.phase == "after":
        print("[Margin] Phase 2/2: sampling the chair AFTER your nudge. "
              "(Make sure you already ran --phase before and nudged the chair first.)")

    report = measure(
        source=args.source,
        weights=args.weights,
        confidence=args.conf,
        frames=args.frames,
        warmup=args.warmup,
        sample_every=args.sample_every,
        match_iou=args.match_iou,
        min_area=args.min_area,
        min_ratio=args.min_ratio,
        preview=not args.no_preview,
    )
    if report is None:
        return

    print_report(report, overlap_threshold=args.overlap_threshold)

    if save_path:
        save_json(report, save_path)
    if plot_path:
        save_plot(report, plot_path)

    if args.phase == "before":
        print("[Margin] Baseline saved. Now walk over, nudge/push the chair a realistic "
              f"amount, walk back, then run:\n"
              f"    python scripts/chair_bbox_margin.py --source {args.source} --phase after\n")
    elif args.phase == "after":
        before_path = Path("output/chair_bbox_margin_before.json")
        if before_path.exists():
            with open(before_path) as f:
                before_report = json.load(f)
            results = compare_phases(before_report, report)
            if results:
                history = log_trial(results, tag=args.trial_tag)
                print_margin_of_error(history, overlap_threshold=args.overlap_threshold,
                                       min_trials=args.min_trials)
        else:
            print(f"[Margin] No baseline found at {before_path} — run --phase before first "
                  f"if you want the before/after movement comparison.")


if __name__ == "__main__":
    main()
