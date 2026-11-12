"""
preflight.py
One-command check before a lab session. Catches setup problems in ~30 s
instead of halfway through a test. Changes nothing in the project.

Usage (project root, venv active):
    python scripts/preflight.py                # checks cameras 0 and 1
    python scripts/preflight.py --cams 0 1 2

Checks
------
  1. Python packages import (incl. pkg_resources / setuptools<81 for DeepSORT)
  2. Model weights from settings.yaml exist and run (one YOLO inference)
  3. DeepSORT tracker initialises (mobilenet embedder loads)
  4. Each camera opens, gives frames, resolution, real fps
     -> snapshots saved to output/preflight/camN.jpg so you can check framing
  5. Seat / shared-desk files: present? from today? (old ones = old layout)
  6. Free disk space for recordings
Result: PASS / WARN / FAIL per check, and a final verdict.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root (this file is in scripts/)
sys.path.insert(0, str(ROOT))

results: list[tuple[str, str, str]] = []   # (status, check, detail)


def report(status: str, check: str, detail: str = "") -> None:
    results.append((status, check, detail))
    mark = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
    print(f"{mark} {check}" + (f" - {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────

def check_env() -> None:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    report("PASS" if in_venv else "WARN", "Virtual environment",
           sys.prefix if in_venv else "not active - run .venv\\Scripts\\activate")

    for mod in ["cv2", "numpy", "yaml", "loguru", "pandas", "torch", "ultralytics",
                "deep_sort_realtime"]:
        try:
            importlib.import_module(mod)
            report("PASS", f"import {mod}")
        except Exception as e:
            report("FAIL", f"import {mod}", f"{type(e).__name__}: {e}")

    try:
        import pkg_resources  # noqa: F401  (DeepSORT's embedder needs it)
        report("PASS", "pkg_resources (needed by DeepSORT)")
    except Exception:
        report("FAIL", "pkg_resources (needed by DeepSORT)",
               'run: pip install "setuptools<81"')


def check_model(cfg):
    import numpy as np

    from src.detector import YOLOv8Detector
    weights = cfg.model.custom_weights or cfg.model.weights
    if not (ROOT / weights).exists() and not Path(weights).exists():
        report("FAIL", "Model weights", f"{weights} not found")
        return None
    try:
        det = YOLOv8Detector(weights=weights, confidence=cfg.model.confidence_threshold,
                             iou_threshold=cfg.model.iou_threshold,
                             device=cfg.model.device, imgsz=cfg.model.imgsz)
        dummy = np.zeros((480, 640, 3), np.uint8)
        det.detect(dummy)                      # warm-up
        t0 = time.perf_counter()
        det.detect(dummy)
        ms = (time.perf_counter() - t0) * 1000
        report("PASS", "YOLO loads and runs", f"{weights} on {cfg.model.device}, ~{ms:.0f} ms/frame")
        return det
    except Exception as e:
        report("FAIL", "YOLO loads and runs", f"{type(e).__name__}: {e}")
        return None


def check_tracker(cfg) -> None:
    try:
        from src.tracker import DeepSORTTracker
        DeepSORTTracker(max_age=cfg.tracker.max_age, n_init=cfg.tracker.n_init,
                        max_iou_distance=cfg.tracker.max_iou_distance,
                        embedder=cfg.tracker.embedder)
        report("PASS", "DeepSORT tracker initialises", f"embedder={cfg.tracker.embedder}")
    except Exception as e:
        report("FAIL", "DeepSORT tracker initialises", f"{type(e).__name__}: {e}")


def check_cameras(cams: list[str], det, out_dir: Path) -> None:
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam in cams:
        if cam.isdigit():
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(cam), backend)   # same backend as run_dual
        else:
            cap = cv2.VideoCapture(cam)
        if not cap.isOpened():
            report("FAIL", f"Camera {cam}", "cannot open (unplugged? in use by another program?)")
            continue

        frame = None
        for _ in range(30):                     # let exposure settle
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            report("FAIL", f"Camera {cam}", "opens but returns no frames")
            cap.release()
            continue

        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 2.0:
            ok, f = cap.read()
            if ok:
                n += 1
                frame = f
        fps = n / (time.perf_counter() - t0)
        cap.release()

        h, w = frame.shape[:2]
        mean = float(frame.mean())
        snap = out_dir / f"cam{cam}.jpg"
        people = ""
        if det is not None:
            try:
                dets = det.detect(frame)
                people = f", {len(dets)} person(s) detected"
                for d in dets:
                    x1, y1, x2, y2 = d.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            except Exception:
                pass
        cv2.imwrite(str(snap), frame)

        status = "PASS"
        notes = []
        if mean < 25:
            status, notes = "WARN", notes + ["image very dark - lens cover / lighting?"]
        if fps < 5:
            status, notes = "WARN", notes + ["low fps - try another USB port"]
        report(status, f"Camera {cam}",
               f"{w}x{h} @ {fps:.1f} fps{people}; snapshot {snap.relative_to(ROOT)}"
               + (" | " + "; ".join(notes) if notes else ""))


def check_seat_files(cfg, cams: list[str]) -> None:
    today = dt.date.today()
    out = ROOT / cfg.io.output_dir
    files = [out / f"seats_cam{i}.json" for i in range(len(cams))]
    files.append(ROOT / "config" / "shared_seats.yaml")
    for f in files:
        rel = f.relative_to(ROOT)
        if not f.exists():
            report("PASS", f"{rel}", "not present - you will draw / pair fresh (expected before setup)")
            continue
        mdate = dt.date.fromtimestamp(f.stat().st_mtime)
        if mdate == today:
            report("PASS", f"{rel}", "from today")
        else:
            report("WARN", f"{rel}", f"from {mdate} - old layout? delete it if cameras moved")


def check_disk() -> None:
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    status = "PASS" if free_gb >= 5 else ("WARN" if free_gb >= 2 else "FAIL")
    report(status, "Free disk space", f"{free_gb:.1f} GB (15-min x2 clips need roughly 1-3 GB)")


# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-session system check")
    ap.add_argument("--cams", nargs="+", default=["0", "1"])
    ap.add_argument("--config", default=str(ROOT / "config" / "settings.yaml"))
    args = ap.parse_args()

    print("\n=== Preflight check ===\n")
    check_env()
    if any(s == "FAIL" and c.startswith("import cv2") for s, c, _ in results):
        print("\nOpenCV missing - fix the environment first.")
        sys.exit(1)

    from config.config_loader import load_config
    cfg = load_config(args.config)

    det = check_model(cfg)
    check_tracker(cfg)
    check_cameras(args.cams, det, ROOT / cfg.io.output_dir / "preflight")
    check_seat_files(cfg, args.cams)
    check_disk()

    fails = sum(1 for s, _, _ in results if s == "FAIL")
    warns = sum(1 for s, _, _ in results if s == "WARN")
    print("\n=== Result ===")
    if fails:
        print(f"NOT READY - {fails} failure(s), {warns} warning(s). Fix the [FAIL] lines above.")
    elif warns:
        print(f"READY WITH WARNINGS - {warns} warning(s). Read the [WARN] lines.")
    else:
        print("READY - all checks passed.")
    print("Open output\\preflight\\cam*.jpg to check each camera's framing.\n")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
