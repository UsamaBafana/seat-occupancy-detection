# Compute benchmark — 2026-10-07 18:12

## Test setup

- Hardware: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel · 16 logical cores · 16.9 GB RAM · GPU: none
- Device used: `cpu` · YOLO weights `models/yolov8n.pt` @ imgsz 640 · tracker embedder `mobilenet`
- Face recognition: off
- Source: 1280x720 @ 30.0 fps, 1000 frames timed · avg 1.02 people tracked (max 2)

## Per-stage time (ms, one camera, one frame)

| Stage | Mean | p50 | p95 | Max | Samples |
|---|---:|---:|---:|---:|---:|
| Decode | 3.0 | 2.7 | 4.9 | 12.3 | 1000 |
| YOLO detect | 79.6 | 65.0 | 190.7 | 456.5 | 1000 |
| DeepSORT track | 39.3 | 28.8 | 106.3 | 199.8 | 1000 |
| Face (per person, uncached) | 0.0 | 0.0 | 0.0 | 0.0 | 0 |

## Load per camera (using mean)

Assumes stream 15.0 fps, occupancy update 1.0 fps, 1.02 people/camera, 0.00 face calls/s/camera.

| Component | ms of work per second |
|---|---:|
| Decode | 44.6 |
| Detect + track | 118.9 |
| Face recognition | 0.0 |
| **Total** | **163.5** |

**≈ 4 cameras per machine** like this one (keeping 30% headroom, single pipeline process).

## Scaling

| Cameras | Load (ms/s) | Machines needed | Bandwidth (Mbps) |
|---:|---:|---:|---:|
| 1 | 163 | 1 | 3 |
| 2 | 327 | 1 | 6 |
| 5 | 817 | 2 | 15 |
| 10 | 1635 | 3 | 30 |
| 20 | 3270 | 5 | 60 |
| 50 | 8174 | 12 | 150 |
| 100 | 16349 | 24 | 300 |

## Memory

- Process RSS: 485 MB after model load → 549 MB at end

_Load = milliseconds of compute needed every second. 1000 ms/s = one machine fully busy._