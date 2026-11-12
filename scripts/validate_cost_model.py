"""
validate_cost_model.py
Check the cost formula against a real live run.

  PREDICTED : per-stage times measured offline by benchmark.py, plugged into
              the cost formula at the SAME rates the live run actually ran at
              (frames processed per second, people per camera, cameras).
  MEASURED  : what the live run actually spent, logged by
              `run_dual.py --log-resources`.

If the two agree (e.g. within ~10-20 %), the formula used for the
N-camera cost table is backed by a real run, not just assumed.

Usage
-----
    python scripts/validate_cost_model.py \
        --benchmark output/benchmark_<time>_summary.json \
        --resources output/resources_<time>_summary.json

Use the SAME settings for both runs (device, --imgsz, face on/off).
Writes output/cost_validation_<time>.md (tables ready for the report).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def pct_err(pred: float, meas: float) -> str:
    if not meas:
        return "–"
    return f"{(pred - meas) / meas * 100:+.0f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare cost-formula prediction with a live run")
    ap.add_argument("--benchmark", required=True, help="benchmark_<time>_summary.json")
    ap.add_argument("--resources", required=True, help="resources_<time>_summary.json")
    ap.add_argument("--headroom", type=float, default=0.7)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    bench = json.loads(Path(args.benchmark).read_text())
    res = json.loads(Path(args.resources).read_text())

    bs, rs = bench["settings"], res["settings"]
    st = bench["stats"]
    m, pk = res["mean"], res["peak"]

    # ── settings check ──
    warnings = []
    if str(bs.get("device")) != str(rs.get("device")):
        warnings.append(f"device differs: benchmark={bs.get('device')} live={rs.get('device')}")
    if int(bs.get("imgsz") or 0) != int(rs.get("imgsz") or 0):
        warnings.append(f"imgsz differs: benchmark={bs.get('imgsz')} live={rs.get('imgsz')}")
    if bool(bs.get("face_enabled")) != bool(rs.get("face_enabled")):
        warnings.append(f"face recognition differs: benchmark={bs.get('face_enabled')} "
                        f"live={rs.get('face_enabled')}")

    # ── live rates ──
    N = int(rs["n_cams"])
    P = float(m["processed_per_s_per_cam"] or 0)      # frames processed / s / camera
    people = float(m["avg_people_per_cam"] or 0)
    face_on = bool(rs.get("face_enabled"))

    # ── benchmark per-stage times ──
    t_det = st["detect"]["mean"]
    t_trk = st["track"]["mean"]
    t_face = st["face_per_call"]["mean"] if st["face_per_call"]["n"] else 0.0
    ttl = bs.get("face_cache_ttl_s") or 5.0
    face_calls_per_s = people * min(P, 1.0 / ttl) if face_on else 0.0

    pred = {
        "detect": N * P * t_det,
        "track": N * P * t_trk,
        "face": N * face_calls_per_s * t_face,
    }
    pred["total"] = sum(pred.values())
    meas = {
        "detect": m["detect_ms_per_s"] or 0.0,
        "track": m["track_ms_per_s"] or 0.0,
        "face": m["face_ms_per_s"] or 0.0,
        "total": m["compute_ms_per_s"] or 0.0,
    }

    budget = args.headroom * 1000.0
    pred_per_cam = pred["total"] / N if N else 0
    meas_per_cam = meas["total"] / N if N else 0
    cams_pred = math.floor(budget / pred_per_cam) if pred_per_cam else 0
    cams_meas = math.floor(budget / meas_per_cam) if meas_per_cam else 0

    # ── report ──
    L = [f"# Cost model validation — {time.strftime('%Y-%m-%d %H:%M')}", ""]
    if warnings:
        L += ["> **Settings mismatch** — fix before trusting the comparison:", ""]
        L += [f"> - {w}" for w in warnings] + [""]
    L += ["## Live run", "",
          f"- {N} camera(s), device `{rs.get('device')}`, imgsz {rs.get('imgsz')}, "
          f"detect every {rs.get('process_every')} frame(s), face recognition {'on' if face_on else 'off'}",
          f"- {res['seconds_measured']} s measured (first {res['warmup_skipped_s']:.0f} s skipped)",
          f"- Processed **{P:.2f} frames/s per camera**, avg **{people:.2f} people per camera**",
          ""]

    L += ["## Predicted vs measured compute (ms of work per second, all cameras)", "",
          "| Stage | Predicted (formula + benchmark) | Measured (live) | Error |",
          "|---|---:|---:|---:|"]
    for k, name in [("detect", "YOLO detect"), ("track", "DeepSORT track"),
                    ("face", "Face recognition"), ("total", "**Total**")]:
        L.append(f"| {name} | {pred[k]:.0f} | {meas[k]:.0f} | {pct_err(pred[k], meas[k])} |")
    L.append("")

    L += ["## Per-frame times (ms)", "",
          "| Stage | Benchmark (offline clip) | Live run | Error |",
          "|---|---:|---:|---:|"]
    for key, name in [("detect", "YOLO detect"), ("track", "DeepSORT track")]:
        live = m.get(f"{key}_ms_per_frame") or 0.0
        L.append(f"| {name} | {st[key]['mean']:.1f} | {live:.1f} | {pct_err(st[key]['mean'], live)} |")
    L.append("")

    L += ["## Capacity at the live run's rates", "",
          "| | Formula | Live measurement |",
          "|---|---:|---:|",
          f"| Compute per camera (ms/s) | {pred_per_cam:.0f} | {meas_per_cam:.0f} |",
          f"| Cameras per machine ({int(args.headroom * 100)}% of loop) | {cams_pred} | {cams_meas} |",
          ""]

    L += ["## Machine resources during the live run", "",
          "| Metric | Mean | Peak |", "|---|---:|---:|",
          f"| Process CPU (% of whole machine) | {m.get('proc_cpu_pct_machine')} | {pk.get('proc_cpu_pct_machine')} |",
          f"| Process CPU (cores busy) | {m.get('proc_cpu_cores')} | {pk.get('proc_cpu_cores')} |",
          f"| System CPU (%) | {m.get('sys_cpu_pct')} | {pk.get('sys_cpu_pct')} |",
          f"| Process RAM (MB) | {m.get('rss_mb')} | {pk.get('rss_mb')} |"]
    if m.get("gpu_mem_mb"):
        L.append(f"| GPU memory (MB) | {m.get('gpu_mem_mb')} | {pk.get('gpu_mem_mb')} |")
    L += ["", f"_Machine: {rs.get('cpu_logical_cores')} logical cores. "
              "Error = (predicted − measured) / measured._", ""]

    tot_err = abs((pred["total"] - meas["total"]) / meas["total"] * 100) if meas["total"] else None
    if tot_err is not None:
        verdict = ("agrees well" if tot_err <= 15 else
                   "agrees roughly" if tot_err <= 30 else "does NOT agree")
        L += [f"**Verdict:** the formula {verdict} with the live run "
              f"(total compute error {tot_err:.0f}%).", ""]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cost_validation_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out_path.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
