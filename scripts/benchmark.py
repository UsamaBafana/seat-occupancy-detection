"""
benchmark.py
Measure per-stage processing cost of the occupancy pipeline and project
the compute needed for an N-camera deployment.

It uses the SAME modules and settings as the real pipeline
(src/detector.py, src/tracker.py, src/face_recognition_module.py,
config/settings.yaml), but never writes to known_faces/ or the occupancy log.

Usage (from the project root)
-----------------------------
    python scripts/benchmark.py --source data/raw/lab_clip.mp4
    python scripts/benchmark.py --source data/raw/lab_clip.mp4 --no-face
    python scripts/benchmark.py --source data/raw/lab_clip.mp4 --imgsz 320 --device cuda
    python scripts/benchmark.py --source 0 --frames 300               # live webcam
    python scripts/benchmark.py --source clip.mp4 --target-fps 0.5 --cameras 1 4 10 25 50 100
    python scripts/benchmark.py --source clip.mp4 --machine-cost 1500 --machine-watts 120 --tariff 0.30

What it measures (milliseconds, per camera, per processed frame)
----------------------------------------------------------------
    decode   : cv2 read/decode of one frame
    detect   : YOLOv8 inference (+ crop extraction)
    track    : DeepSORT update (appearance embedding per detection)
    face     : ONE uncached DeepFace recognition call on one person crop

How it projects
---------------
Per camera, every second the machine must do:
    decode : stream_fps  x t_decode        (the stream is decoded continuously)
    detect : target_fps  x (t_detect + t_track)
    face   : face_calls_per_s x t_face
             where face_calls_per_s = avg_people x min(target_fps, 1/face_cache_ttl)
             (the real FaceRecogniser caches each track's identity for cache_ttl s)

    load_per_cam (ms of work per second) = decode + detect + face
    cameras_per_machine = floor(headroom x 1000 / load_per_cam)

The pipeline loop is single-threaded, so this is "cameras per process on
THIS machine". Running several processes / a GPU changes the numbers —
re-run the benchmark on each candidate machine.

Outputs (in output/)
--------------------
    benchmark_<ts>_frames.csv   per-frame raw timings
    benchmark_<ts>_summary.json all stats + projection table
    benchmark_<ts>_report.md    tables ready to paste into the report
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent   # project root (this file is in scripts/)
sys.path.insert(0, str(ROOT))

from config.config_loader import load_config  # noqa: E402
from src.detector import YOLOv8Detector  # noqa: E402
from src.tracker import DeepSORTTracker  # noqa: E402

# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _sync(device: str) -> None:
    """Make GPU timings honest (CUDA calls are async)."""
    if str(device).startswith("cuda"):
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    s = sorted(values)
    p95 = s[min(len(s) - 1, int(math.ceil(0.95 * len(s))) - 1)]
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "p50": statistics.median(s),
        "p95": p95,
        "max": s[-1],
    }


def _hardware_info(device: str) -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "device": device,
    }
    try:
        import os
        info["cpu_logical_cores"] = os.cpu_count()
    except Exception:
        pass
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def _rss_mb() -> float | None:
    try:
        import os

        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        return None


def _gpu_peak_mb(device: str) -> float | None:
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch
        return torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
#  Face recogniser (read-only — never auto-enrols during a benchmark)
# ─────────────────────────────────────────────────────────────

def _build_face(cfg):
    from src.face_recognition_module import FaceRecogniser
    fr_cfg = cfg.face_recognition
    face = FaceRecogniser(
        known_faces_dir=fr_cfg.known_faces_dir,
        model_name=fr_cfg.model,
        detector_backend=fr_cfg.detector_backend,
        distance_metric=fr_cfg.distance_metric,
        threshold=fr_cfg.threshold,
        min_face_size=fr_cfg.min_face_size,
        auto_enroll=False,  # IMPORTANT: don't write new faces to the gallery
        face_region_top_fraction=fr_cfg.get("face_region_top_fraction", 0.5),
    )
    return face


# ─────────────────────────────────────────────────────────────
#  Benchmark
# ─────────────────────────────────────────────────────────────

def run_benchmark(args) -> dict:
    cfg = load_config(args.config)

    device = args.device or cfg.model.device
    imgsz = args.imgsz or cfg.model.imgsz
    weights = args.weights or cfg.model.custom_weights or cfg.model.weights
    use_face = (not args.no_face) and bool(cfg.face_recognition.enabled)

    print(f"[bench] weights={weights}  device={device}  imgsz={imgsz}  face={use_face}")

    accel = args.accel
    ov_name = None
    if accel == "cpu":
        detector = YOLOv8Detector(
            weights=weights,
            confidence=cfg.model.confidence_threshold,
            iou_threshold=cfg.model.iou_threshold,
            device=device,
            imgsz=imgsz,
        )
    else:
        from src.accel import OpenVINODetector
        detector = OpenVINODetector(
            weights=weights,
            confidence=cfg.model.confidence_threshold,
            iou_threshold=cfg.model.iou_threshold,
            imgsz=imgsz,
            ov_device=args.ov_device,
        )
        ov_name = detector.device_name
    tracker = DeepSORTTracker(
        max_age=cfg.tracker.max_age,
        n_init=cfg.tracker.n_init,
        max_iou_distance=cfg.tracker.max_iou_distance,
        embedder=cfg.tracker.embedder,
    )
    if accel == "gpu":
        from src.accel import accelerate_tracker
        accelerate_tracker(tracker, args.ov_device)
    emb = getattr(tracker._tracker, "embedder", None)
    if args.track_device == "cpu" and emb is not None and getattr(emb, "gpu", False):
        emb.model.float().cpu()                 # keep DeepSORT's appearance model on the CPU
        emb.gpu, emb.half = False, False
    if args.face_device == "cpu" and use_face:
        try:                                     # hide the GPU from TensorFlow (DeepFace)
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
        except Exception as e:
            print(f"[bench] could not hide GPU from TensorFlow: {e}")
    cuda = str(device).startswith("cuda")
    if accel == "cpu":
        where = {"detect": "GPU (CUDA)" if cuda else "CPU (PyTorch)",
                 "track": "GPU (CUDA)" if (emb is not None and getattr(emb, "gpu", False)) else "CPU (PyTorch)"}
    else:
        where = {
            "hybrid": {"detect": f"{args.ov_device} (OpenVINO)", "track": "CPU (PyTorch)"},
            "gpu": {"detect": f"{args.ov_device} (OpenVINO)", "track": f"{args.ov_device} (OpenVINO)"},
        }[accel]
    where["face"] = "CPU (TensorFlow)"
    face = None
    face_cache_ttl = args.face_cache_ttl
    if use_face:
        face = _build_face(cfg)
        face_cache_ttl = face_cache_ttl or getattr(face, "cache_ttl", 5.0)
        try:
            import tensorflow as tf
            if tf.config.list_logical_devices("GPU"):
                where["face"] = "GPU (TensorFlow)"
        except Exception:
            pass
        if not face.is_available():
            print("[bench] WARNING: face recognition not available "
                  "(DeepFace missing or known_faces/ empty) — face cost will read ~0. "
                  "Enrol a few faces first for a realistic number.")

    if accel == "cpu" and cuda:
        run_label = "cuda_gpu" if all(v.startswith("GPU") for k, v in where.items() if k != "face" or use_face) else "cuda_hybrid"
    else:
        run_label = accel
    print(f"[bench] mode={run_label}: detect on {where['detect']}, track on {where['track']}, face on {where['face']}")

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"[bench] source {width}x{height} @ {src_fps:.1f} fps")

    rss_start = _rss_mb()

    # ── Warm-up (model load / JIT / cudnn autotune are not representative) ──
    for _ in range(args.warmup):
        ok, frame = cap.read()
        if not ok:
            break
        detector.detect(frame, extract_crops=True)

    rows = []
    face_times: list[float] = []
    face_ids: dict[str, int] = {}   # who each sampled face call returned
    frames_done = 0
    print(f"[bench] timing {args.frames} frames ...")
    t_wall0 = time.perf_counter()

    while frames_done < args.frames:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            if isinstance(src, str) and args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        detections = detector.detect(frame, extract_crops=True)
        _sync(device)
        t2 = time.perf_counter()

        tracks = tracker.update(detections, frame)
        _sync(device)
        t3 = time.perf_counter()

        # Face: time one UNCACHED call per person, but only every Nth frame
        # (it's slow; we only need a good average, not every sample).
        face_ms_this_frame = 0.0
        if face is not None and face.is_available() and frames_done % args.face_every == 0:
            for t in tracks[: args.face_max_per_frame]:
                if t.crop is None or t.crop.size == 0:
                    continue
                f0 = time.perf_counter()
                r = face.identify_frame(t.crop)
                _sync(device)
                dt = (time.perf_counter() - f0) * 1000
                name = getattr(r, "identity", "unknown") or "unknown"
                face_ids[name] = face_ids.get(name, 0) + 1
                face_times.append(dt)
                face_ms_this_frame += dt

        rows.append({
            "frame": frames_done,
            "decode_ms": (t1 - t0) * 1000,
            "detect_ms": (t2 - t1) * 1000,
            "track_ms": (t3 - t2) * 1000,
            "face_ms_sampled": face_ms_this_frame,
            "n_detections": len(detections),
            "n_tracks": len(tracks),
        })
        frames_done += 1
        if frames_done % 50 == 0:
            ids = (" | faces: " + ", ".join(f"{k} {v}" for k, v in face_ids.items())) if face_ids else ""
            print(f"  {frames_done}/{args.frames}{ids}")

    wall_s = time.perf_counter() - t_wall0
    cap.release()

    if not rows:
        raise RuntimeError("No frames were processed — check the source.")

    st = {
        "decode": _stats([r["decode_ms"] for r in rows]),
        "detect": _stats([r["detect_ms"] for r in rows]),
        "track": _stats([r["track_ms"] for r in rows]),
        "face_per_call": _stats(face_times),
    }
    avg_dets = statistics.fmean(r["n_detections"] for r in rows)
    avg_people = statistics.fmean(r["n_tracks"] for r in rows)
    max_people = max(r["n_tracks"] for r in rows)

    return {
        "rows": rows,
        "stats": st,
        "avg_detections": avg_dets,
        "avg_people": avg_people,
        "max_people": max_people,
        "face_identities": face_ids,
        "frames": frames_done,
        "wall_s": wall_s,
        "source": {"path": str(args.source), "width": width, "height": height, "fps": src_fps},
        "settings": {
            "weights": str(weights), "device": device, "imgsz": imgsz,
            "face_enabled": use_face,
            "face_model": cfg.face_recognition.model if use_face else None,
            "face_detector": cfg.face_recognition.detector_backend if use_face else None,
            "face_cache_ttl_s": face_cache_ttl if use_face else None,
            "embedder": cfg.tracker.embedder,
            "accel": accel,
            "run_label": run_label,
            "ov_device": args.ov_device if accel != "cpu" else None,
            "ov_device_name": ov_name,
            "where": where,
        },
        "memory": {
            "rss_start_mb": rss_start,
            "rss_end_mb": _rss_mb(),
            "gpu_peak_mb": _gpu_peak_mb(device),
        },
        "hardware": _hardware_info(device),
    }


# ─────────────────────────────────────────────────────────────
#  Projection
# ─────────────────────────────────────────────────────────────

def project(res: dict, args) -> dict:
    st = res["stats"]
    stat_key = "p95" if args.use_p95 else "mean"

    t_decode = st["decode"][stat_key]
    t_detect = st["detect"][stat_key]
    t_track = st["track"][stat_key]
    t_face = st["face_per_call"][stat_key] if st["face_per_call"]["n"] else 0.0

    people = args.people if args.people is not None else res["avg_people"]
    stream_fps = args.stream_fps
    target_fps = args.target_fps

    ttl = res["settings"]["face_cache_ttl_s"] or 0
    face_rate = (min(target_fps, 1.0 / ttl) if ttl else target_fps) if res["settings"]["face_enabled"] else 0.0
    face_calls_per_s = people * face_rate

    load_decode = stream_fps * t_decode
    load_detect = target_fps * (t_detect + t_track)
    load_face = face_calls_per_s * t_face
    load_per_cam = load_decode + load_detect + load_face  # ms of work / second

    budget = args.headroom * 1000.0
    cams_per_machine = max(0, int(budget // load_per_cam)) if load_per_cam > 0 else 0

    table = []
    for n in args.cameras:
        load = n * load_per_cam
        machines = math.ceil(load / budget) if load > 0 else 0
        row = {
            "cameras": n,
            "load_ms_per_s": load,
            "util_one_machine_pct": load / 10.0,  # % of one core-loop second
            "machines_needed": machines,
            "bandwidth_mbps": n * args.mbps_per_camera,
        }
        if args.machine_cost is not None:
            row["hardware_cost"] = machines * args.machine_cost
        if args.machine_watts is not None and args.tariff is not None:
            kwh_year = machines * args.machine_watts * 24 * 365 / 1000
            row["energy_kwh_year"] = kwh_year
            row["energy_cost_year"] = kwh_year * args.tariff
            if args.machine_cost is not None:
                row["total_cost_3yr"] = row["hardware_cost"] + 3 * row["energy_cost_year"]
        table.append(row)

    return {
        "statistic": stat_key,
        "assumptions": {
            "stream_fps": stream_fps,
            "target_fps": target_fps,
            "people_per_camera": people,
            "face_calls_per_s_per_cam": face_calls_per_s,
            "headroom": args.headroom,
            "mbps_per_camera": args.mbps_per_camera,
            "machine_cost": args.machine_cost,
            "machine_watts": args.machine_watts,
            "tariff_per_kwh": args.tariff,
        },
        "per_camera_ms_per_s": {
            "decode": load_decode,
            "detect_track": load_detect,
            "face": load_face,
            "total": load_per_cam,
        },
        "cameras_per_machine": cams_per_machine,
        "table": table,
    }


# ─────────────────────────────────────────────────────────────
#  Output
# ─────────────────────────────────────────────────────────────

def write_outputs(res: dict, proj: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"benchmark_{ts}_{res['settings'].get('run_label') or res['settings'].get('accel', 'cpu')}{'' if res['settings'].get('face_enabled') else '_noface'}"

    # Per-frame CSV
    import csv
    csv_path = Path(f"{base}_frames.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res["rows"][0].keys()))
        w.writeheader()
        w.writerows(res["rows"])

    # JSON
    json_path = Path(f"{base}_summary.json")
    summary = {k: v for k, v in res.items() if k != "rows"}
    summary["projection"] = proj
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Markdown report
    st, s, hw = res["stats"], res["settings"], res["hardware"]
    a, pc = proj["assumptions"], proj["per_camera_ms_per_s"]
    L = []
    L.append(f"# Compute benchmark — {time.strftime('%Y-%m-%d %H:%M')}\n")
    L.append("## Test setup\n")
    L.append(f"- Hardware: {hw.get('processor')} · {hw.get('cpu_logical_cores', '?')} logical cores · "
             f"{hw.get('ram_total_gb', '?')} GB RAM · GPU: {hw.get('gpu', 'none')}")
    L.append(f"- Device used: `{s['device']}` · YOLO weights `{s['weights']}` @ imgsz {s['imgsz']} · "
             f"tracker embedder `{s['embedder']}`")
    wh = s.get("where") or {}
    if wh:
        L.append(f"- Mode `{s.get('run_label') or s.get('accel', 'cpu')}`: YOLO on {wh['detect']} · DeepSORT model on {wh['track']} · "
                 f"face recognition on {wh['face']}"
                 + (f" · OpenVINO device: {s['ov_device_name']}" if s.get('ov_device_name') else ""))
    L.append(f"- Face recognition: {'%s + %s (cache %ss)' % (s['face_model'], s['face_detector'], s['face_cache_ttl_s']) if s['face_enabled'] else 'off'}")
    fid = res.get("face_identities") or {}
    if s["face_enabled"]:
        tot = sum(fid.values())
        L.append("- Face results (sampled calls): " + (", ".join(
            f"{k} {v} ({v / tot * 100:.0f}%)" for k, v in sorted(fid.items(), key=lambda kv: -kv[1]))
            if tot else "none — no face calls ran"))
    L.append(f"- Source: {res['source']['width']}x{res['source']['height']} @ {res['source']['fps']:.1f} fps, "
             f"{res['frames']} frames timed · avg {res['avg_people']:.2f} people tracked (max {res['max_people']})\n")

    L.append("## Per-stage time (ms, one camera, one frame)\n")
    L.append("| Stage | Mean | p50 | p95 | Max | Samples |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for name, key in [("Decode", "decode"), ("YOLO detect", "detect"),
                      ("DeepSORT track", "track"), ("Face (per person, uncached)", "face_per_call")]:
        x = st[key]
        L.append(f"| {name} | {x['mean']:.1f} | {x['p50']:.1f} | {x['p95']:.1f} | {x['max']:.1f} | {x['n']} |")
    L.append("")

    L.append(f"## Load per camera (using {proj['statistic']})\n")
    L.append(f"Assumes stream {a['stream_fps']} fps, occupancy update {a['target_fps']} fps, "
             f"{a['people_per_camera']:.2f} people/camera, {a['face_calls_per_s_per_cam']:.2f} face calls/s/camera.\n")
    L.append("| Component | ms of work per second |")
    L.append("|---|---:|")
    L.append(f"| Decode | {pc['decode']:.1f} |")
    L.append(f"| Detect + track | {pc['detect_track']:.1f} |")
    L.append(f"| Face recognition | {pc['face']:.1f} |")
    L.append(f"| **Total** | **{pc['total']:.1f}** |")
    L.append("")
    L.append(f"**≈ {proj['cameras_per_machine']} cameras per machine** like this one "
             f"(keeping {int((1 - a['headroom']) * 100)}% headroom, single pipeline process).\n")

    L.append("## Scaling\n")
    cols = ["Cameras", "Load (ms/s)", "Machines needed", "Bandwidth (Mbps)"]
    has_hw = a["machine_cost"] is not None
    has_en = a["machine_watts"] is not None and a["tariff_per_kwh"] is not None
    if has_hw:
        cols.append("Hardware cost")
    if has_en:
        cols += ["Energy (kWh/yr)", "Energy cost/yr"]
    if has_hw and has_en:
        cols.append("3-yr total")
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "---:|" * len(cols))
    for r in proj["table"]:
        cells = [str(r["cameras"]), f"{r['load_ms_per_s']:.0f}", str(r["machines_needed"]),
                 f"{r['bandwidth_mbps']:.0f}"]
        if has_hw:
            cells.append(f"{r['hardware_cost']:,.0f}")
        if has_en:
            cells += [f"{r['energy_kwh_year']:,.0f}", f"{r['energy_cost_year']:,.0f}"]
        if has_hw and has_en:
            cells.append(f"{r['total_cost_3yr']:,.0f}")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    m = res["memory"]
    L.append("## Memory\n")
    L.append(f"- Process RSS: {m['rss_start_mb'] or 0:.0f} MB after model load → {m['rss_end_mb'] or 0:.0f} MB at end")
    if m["gpu_peak_mb"]:
        L.append(f"- GPU peak allocated: {m['gpu_peak_mb']:.0f} MB")
    L.append("\n_Load = milliseconds of compute needed every second. 1000 ms/s = one machine fully busy._")

    md_path = Path(f"{base}_report.md")
    md_path.write_text("\n".join(L), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "md": md_path, "md_text": "\n".join(L)}


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark pipeline compute cost and project N-camera requirements.")
    p.add_argument("--source", required=True, help="Video file, RTSP url, or camera index (e.g. 0)")
    p.add_argument("--config", default=str(ROOT / "config" / "settings.yaml"))
    p.add_argument("--weights", default=None, help="Override model weights")
    p.add_argument("--device", default=None, help="cpu | cuda | mps (default: settings.yaml)")
    p.add_argument("--imgsz", type=int, default=None, help="Override inference size")
    p.add_argument("--no-face", action="store_true", help="Skip face recognition")
    p.add_argument("--accel", choices=["cpu", "hybrid", "gpu"], default="cpu",
                   help="cpu: all on CPU | hybrid: YOLO on Intel GPU | gpu: YOLO + DeepSORT model on Intel GPU (needs: pip install openvino)")
    p.add_argument("--track-device", choices=["auto", "cpu"], default="auto",
                   help="cpu: keep DeepSORT's appearance model on the CPU even if a CUDA GPU is present")
    p.add_argument("--face-device", choices=["auto", "cpu"], default="auto",
                   help="cpu: keep face recognition (TensorFlow) on the CPU even if a GPU is present")
    p.add_argument("--ov-device", default="GPU", help="OpenVINO device for --accel hybrid/gpu (GPU = Intel Iris Xe; CPU = OpenVINO on the CPU, for comparison)")

    p.add_argument("--frames", type=int, default=200, help="Frames to time")
    p.add_argument("--warmup", type=int, default=10, help="Warm-up frames (not timed)")
    p.add_argument("--loop", action="store_true", help="Loop the video if it ends early")
    p.add_argument("--face-every", type=int, default=5, help="Sample face cost every N frames")
    p.add_argument("--face-max-per-frame", type=int, default=5, help="Max people to face-test per sampled frame")
    p.add_argument("--face-cache-ttl", type=float, default=None, help="Seconds a face identity is cached (default: module's cache_ttl)")

    p.add_argument("--stream-fps", type=float, default=15.0, help="Camera stream fps (decode cost)")
    p.add_argument("--target-fps", type=float, default=1.0, help="Occupancy updates per second per camera")
    p.add_argument("--people", type=float, default=None, help="People per camera (default: measured average)")
    p.add_argument("--headroom", type=float, default=0.7, help="Max fraction of machine to use")
    p.add_argument("--use-p95", action="store_true", help="Project with p95 instead of mean timings")
    p.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100])
    p.add_argument("--mbps-per-camera", type=float, default=3.0, help="Network bitrate per camera stream")

    p.add_argument("--machine-cost", type=float, default=None, help="Price of one machine like the test one")
    p.add_argument("--machine-watts", type=float, default=None, help="Average power draw of one machine (W)")
    p.add_argument("--tariff", type=float, default=None, help="Electricity price per kWh")
    p.add_argument("--out", default=str(ROOT / "output"))
    return p.parse_args()


def main():
    args = parse_args()
    res = run_benchmark(args)
    proj = project(res, args)
    paths = write_outputs(res, proj, Path(args.out))
    print("\n" + paths["md_text"])
    print(f"\n[bench] wrote:\n  {paths['md']}\n  {paths['json']}\n  {paths['csv']}")


if __name__ == "__main__":
    main()
