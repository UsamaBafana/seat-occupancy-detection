# Compute benchmark — 2026-10-07 18:34

## Test setup

- Hardware: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel · 16 logical cores · 16.9 GB RAM · GPU: none
- Device used: `cpu` · YOLO weights `models/yolov8n.pt` @ imgsz 640 · tracker embedder `mobilenet`
- Face recognition: VGG-Face + mtcnn (cache 5.0s)
- Source: 1280x720 @ 30.0 fps, 1000 frames timed · avg 1.02 people tracked (max 2)

## Per-stage time (ms, one camera, one frame)

| Stage | Mean | p50 | p95 | Max | Samples |
|---|---:|---:|---:|---:|---:|
| Decode | 3.8 | 3.2 | 7.8 | 17.6 | 1000 |
| YOLO detect | 116.6 | 92.6 | 227.2 | 1753.2 | 1000 |
| DeepSORT track | 56.4 | 47.4 | 120.0 | 1646.9 | 1000 |
| Face (per person, uncached) | 749.1 | 582.0 | 1075.1 | 18575.7 | 200 |

## Load per camera (using mean)

Assumes stream 15.0 fps, occupancy update 1.0 fps, 1.02 people/camera, 0.20 face calls/s/camera.

| Component | ms of work per second |
|---|---:|
| Decode | 57.7 |
| Detect + track | 172.9 |
| Face recognition | 153.0 |
| **Total** | **383.6** |

**≈ 1 cameras per machine** like this one (keeping 30% headroom, single pipeline process).

## Scaling

| Cameras | Load (ms/s) | Machines needed | Bandwidth (Mbps) |
|---:|---:|---:|---:|
| 1 | 384 | 1 | 3 |
| 2 | 767 | 2 | 6 |
| 5 | 1918 | 3 | 15 |
| 10 | 3836 | 6 | 30 |
| 20 | 7673 | 11 | 60 |
| 50 | 19181 | 28 | 150 |
| 100 | 38363 | 55 | 300 |

## Memory

- Process RSS: 485 MB after model load → 3103 MB at end

_Load = milliseconds of compute needed every second. 1000 ms/s = one machine fully busy._