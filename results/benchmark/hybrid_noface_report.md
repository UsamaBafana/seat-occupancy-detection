# Compute benchmark — 2026-10-08 01:56

## Test setup

- Hardware: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel · 16 logical cores · 16.9 GB RAM · GPU: none
- Device used: `cpu` · YOLO weights `models/yolov8n.pt` @ imgsz 640 · tracker embedder `mobilenet`
- Mode `hybrid`: YOLO on GPU (OpenVINO) · DeepSORT model on CPU (PyTorch) · face recognition on CPU (TensorFlow) · OpenVINO device: Intel(R) Iris(R) Xe Graphics (iGPU)
- Face recognition: off
- Source: 1280x720 @ 30.0 fps, 1000 frames timed · avg 1.02 people tracked (max 2)

## Per-stage time (ms, one camera, one frame)

| Stage | Mean | p50 | p95 | Max | Samples |
|---|---:|---:|---:|---:|---:|
| Decode | 4.0 | 3.8 | 5.1 | 20.0 | 1000 |
| YOLO detect | 26.6 | 25.6 | 33.9 | 65.7 | 1000 |
| DeepSORT track | 62.5 | 67.8 | 88.3 | 175.0 | 1000 |
| Face (per person, uncached) | 0.0 | 0.0 | 0.0 | 0.0 | 0 |

## Load per camera (using mean)

Assumes stream 15.0 fps, occupancy update 1.0 fps, 1.02 people/camera, 0.00 face calls/s/camera.

| Component | ms of work per second |
|---|---:|
| Decode | 59.4 |
| Detect + track | 89.0 |
| Face recognition | 0.0 |
| **Total** | **148.5** |

**≈ 4 cameras per machine** like this one (keeping 30% headroom, single pipeline process).

## Scaling

| Cameras | Load (ms/s) | Machines needed | Bandwidth (Mbps) |
|---:|---:|---:|---:|
| 1 | 148 | 1 | 3 |
| 2 | 297 | 1 | 6 |
| 5 | 742 | 2 | 15 |
| 10 | 1485 | 3 | 30 |
| 20 | 2969 | 5 | 60 |
| 50 | 7423 | 11 | 150 |
| 100 | 14845 | 22 | 300 |

## Memory

- Process RSS: 839 MB after model load → 872 MB at end

_Load = milliseconds of compute needed every second. 1000 ms/s = one machine fully busy._