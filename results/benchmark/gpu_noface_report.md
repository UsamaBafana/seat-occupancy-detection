# Compute benchmark — 2026-10-08 01:58

## Test setup

- Hardware: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel · 16 logical cores · 16.9 GB RAM · GPU: none
- Device used: `cpu` · YOLO weights `models/yolov8n.pt` @ imgsz 640 · tracker embedder `mobilenet`
- Mode `gpu`: YOLO on GPU (OpenVINO) · DeepSORT model on GPU (OpenVINO) · face recognition on CPU (TensorFlow) · OpenVINO device: Intel(R) Iris(R) Xe Graphics (iGPU)
- Face recognition: off
- Source: 1280x720 @ 30.0 fps, 1000 frames timed · avg 1.02 people tracked (max 2)

## Per-stage time (ms, one camera, one frame)

| Stage | Mean | p50 | p95 | Max | Samples |
|---|---:|---:|---:|---:|---:|
| Decode | 4.3 | 3.9 | 6.0 | 13.9 | 1000 |
| YOLO detect | 28.5 | 27.3 | 34.0 | 55.9 | 1000 |
| DeepSORT track | 50.4 | 22.8 | 35.1 | 23352.8 | 1000 |
| Face (per person, uncached) | 0.0 | 0.0 | 0.0 | 0.0 | 0 |

## Load per camera (using mean)

Assumes stream 15.0 fps, occupancy update 1.0 fps, 1.02 people/camera, 0.00 face calls/s/camera.

| Component | ms of work per second |
|---|---:|
| Decode | 64.2 |
| Detect + track | 78.9 |
| Face recognition | 0.0 |
| **Total** | **143.1** |

**≈ 4 cameras per machine** like this one (keeping 30% headroom, single pipeline process).

## Scaling

| Cameras | Load (ms/s) | Machines needed | Bandwidth (Mbps) |
|---:|---:|---:|---:|
| 1 | 143 | 1 | 3 |
| 2 | 286 | 1 | 6 |
| 5 | 715 | 2 | 15 |
| 10 | 1431 | 3 | 30 |
| 20 | 2862 | 5 | 60 |
| 50 | 7154 | 11 | 150 |
| 100 | 14308 | 21 | 300 |

## Memory

- Process RSS: 814 MB after model load → 929 MB at end

_Load = milliseconds of compute needed every second. 1000 ms/s = one machine fully busy._