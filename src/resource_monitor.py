"""
src/resource_monitor.py
Logs what a live pipeline run actually costs, once per second:

  * process CPU %  (of the whole machine, and in "cores" used)
  * system CPU %, process RAM (MB), GPU memory (MB, if CUDA)
  * pipeline loop rate and processed frames per second
  * milliseconds spent per second in each stage (detect / track / face),
    plus people tracked - the same quantities the cost formula predicts

Output
------
  output/resources_<time>.csv           one row per second
  output/resources_<time>_summary.json  averages for validate_cost_model.py
"""
from __future__ import annotations

import csv
import json
import os
import threading
import time
from pathlib import Path

from loguru import logger


class StageCounters:
    """Thread-safe accumulators the pipeline loop adds to."""

    FIELDS = ("loops", "processed", "detect_ms", "track_ms", "face_ms",
              "detect_n", "track_n", "people_sum", "people_n")

    def __init__(self):
        self._lock = threading.Lock()
        self._v = {k: 0.0 for k in self.FIELDS}

    def add(self, **kw) -> None:
        with self._lock:
            for k, x in kw.items():
                self._v[k] += x

    def snapshot_and_reset(self) -> dict:
        with self._lock:
            snap = dict(self._v)
            self._v = {k: 0.0 for k in self.FIELDS}
        return snap


class ResourceMonitor(threading.Thread):
    COLUMNS = ["t", "time_of_day", "proc_cpu_pct_machine", "proc_cpu_cores",
               "sys_cpu_pct", "rss_mb", "gpu_mem_mb",
               "loops_per_s", "processed_per_s_per_cam",
               "detect_ms_per_s", "track_ms_per_s", "face_ms_per_s",
               "compute_ms_per_s", "avg_people_per_cam",
               "detect_ms_per_frame", "track_ms_per_frame"]

    def __init__(self, counters: StageCounters, out_dir: str | Path, n_cams: int,
                 settings: dict, interval: float = 1.0, warmup_s: float = 5.0):
        super().__init__(daemon=True)
        self.counters = counters
        self.n_cams = max(1, n_cams)
        self.settings = settings
        self.interval = interval
        self.warmup_s = warmup_s
        self._stop_evt = threading.Event()   # NOT self._stop: that name is used inside threading.Thread
        self.rows: list[dict] = []

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = out_dir / f"resources_{stamp}.csv"
        self.summary_path = out_dir / f"resources_{stamp}_summary.json"

        try:
            import psutil
            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
            self._proc.cpu_percent(None)          # prime the counters
            psutil.cpu_percent(None)
        except Exception:
            self._psutil = None
            logger.warning("[Resources] psutil not available - CPU/RAM not logged "
                           "(pip install psutil)")
        self._cores = os.cpu_count() or 1

    # ── thread ──
    def run(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()
        logger.info(f"[Resources] Logging CPU/RAM + stage timings -> {self.csv_path}")

        t0 = time.perf_counter()
        last = t0
        self.counters.snapshot_and_reset()
        while not self._stop_evt.wait(self.interval):
            now = time.perf_counter()
            dt = now - last
            last = now
            c = self.counters.snapshot_and_reset()
            row = self._row(now - t0, dt, c)
            self.rows.append(row)
            with open(self.csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.COLUMNS).writerow(row)

    def _row(self, t: float, dt: float, c: dict) -> dict:
        proc_cpu = sys_cpu = rss = None
        if self._psutil:
            proc_cpu = self._proc.cpu_percent(None)          # % of ONE core (can be >100)
            sys_cpu = self._psutil.cpu_percent(None)
            rss = self._proc.memory_info().rss / 1e6
        gpu = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.memory_allocated() / 1e6
        except Exception:
            pass

        detect_s, track_s, face_s = c["detect_ms"] / dt, c["track_ms"] / dt, c["face_ms"] / dt
        return {
            "t": round(t, 1),
            "time_of_day": time.strftime("%H:%M:%S"),
            "proc_cpu_pct_machine": round(proc_cpu / self._cores, 1) if proc_cpu is not None else "",
            "proc_cpu_cores": round(proc_cpu / 100.0, 2) if proc_cpu is not None else "",
            "sys_cpu_pct": sys_cpu if sys_cpu is not None else "",
            "rss_mb": round(rss, 0) if rss is not None else "",
            "gpu_mem_mb": round(gpu, 0) if gpu is not None else "",
            "loops_per_s": round(c["loops"] / dt, 2),
            "processed_per_s_per_cam": round(c["processed"] / dt, 2),
            "detect_ms_per_s": round(detect_s, 1),
            "track_ms_per_s": round(track_s, 1),
            "face_ms_per_s": round(face_s, 1),
            "compute_ms_per_s": round(detect_s + track_s + face_s, 1),
            "avg_people_per_cam": round(c["people_sum"] / c["people_n"], 2) if c["people_n"] else 0.0,
            "detect_ms_per_frame": round(c["detect_ms"] / c["detect_n"], 1) if c["detect_n"] else "",
            "track_ms_per_frame": round(c["track_ms"] / c["track_n"], 1) if c["track_n"] else "",
        }

    # ── stop + summary ──
    def stop(self) -> Path | None:
        self._stop_evt.set()
        self.join(timeout=3)
        rows = [r for r in self.rows if r["t"] >= self.warmup_s]
        if not rows:
            logger.warning("[Resources] Run too short for a summary.")
            return None

        def mean(key):
            vals = [r[key] for r in rows if r[key] not in ("", None)]
            return round(sum(vals) / len(vals), 3) if vals else None

        def peak(key):
            vals = [r[key] for r in rows if r[key] not in ("", None)]
            return max(vals) if vals else None

        summary = {
            "settings": {**self.settings, "n_cams": self.n_cams, "cpu_logical_cores": self._cores},
            "seconds_measured": len(rows),
            "warmup_skipped_s": self.warmup_s,
            "mean": {k: mean(k) for k in self.COLUMNS if k not in ("t", "time_of_day")},
            "peak": {k: peak(k) for k in ("proc_cpu_pct_machine", "proc_cpu_cores",
                                          "sys_cpu_pct", "rss_mb", "gpu_mem_mb",
                                          "compute_ms_per_s")},
        }
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        m = summary["mean"]
        logger.info(
            f"[Resources] {len(rows)} s measured: process CPU {m['proc_cpu_pct_machine']}% of machine "
            f"({m['proc_cpu_cores']} cores), RAM {m['rss_mb']} MB, "
            f"compute {m['compute_ms_per_s']} ms/s -> {self.summary_path}")
        return self.summary_path
