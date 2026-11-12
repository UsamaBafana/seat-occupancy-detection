# Compute benchmark — 2026-10-08 02:02

## Test setup

- Hardware: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel · 16 logical cores · 16.9 GB RAM · GPU: none
- Device used: `cpu` · YOLO weights `models/yolov8n.pt` @ imgsz 640 · tracker embedder `mobilenet`
- Mode `hybrid`: YOLO on GPU (OpenVINO) · DeepSORT model on CPU (PyTorch) · face recognition on CPU (TensorFlow) · OpenVINO device: Intel(R) Iris(R) Xe Graphics (iGPU)
- Face recognition: VGG-Face + mtcnn (cache 5.0s)
- Face results (sampled calls): person_1 199 (100%), unknown 1 (0%)
- Source: 1280x720 @ 30.0 fps, 1000 frames timed · avg 1.02 people tracked (max 2)

## Per-stage time (ms, one camera, one frame)

| Stage | Mean | p50 | p95 | Max | Samples |
|---|---:|---:|---:|---:|---:|
| Decode | 3.4 | 3.2 | 4.3 | 15.9 | 1000 |
| YOLO detect | 25.0 | 24.0 | 31.6 | 76.8 | 1000 |
| DeepSORT track | 45.6 | 46.0 | 71.2 | 153.2 | 1000 |
| Face (per person, uncached) | 685.4 | 583.6 | 854.4 | 17915.6 | 200 |

## Load per camera (using mean)

Assumes stream 15.0 fps, occupancy update 1.0 fps, 1.02 people/camera, 0.20 face calls/s/camera.

| Component | ms of work per second |
|---|---:|
| Decode | 50.8 |
| Detect + track | 70.6 |
| Face recognition | 140.1 |
| **Total** | **261.6** |

**≈ 2 cameras per machine** like this one (keeping 30% headroom, single pipeline process).

## Scaling

| Cameras | Load (ms/s) | Machines needed | Bandwidth (Mbps) |
|---:|---:|---:|---:|
| 1 | 262 | 1 | 3 |
| 2 | 523 | 1 | 6 |
| 5 | 1308 | 2 | 15 |
| 10 | 2616 | 4 | 30 |
| 20 | 5232 | 8 | 60 |
| 50 | 13079 | 19 | 150 |
| 100 | 26158 | 38 | 300 |

## Memory

- Process RSS: 665 MB after model load → 3175 MB at end

_Load = milliseconds of compute needed every second. 1000 ms/s = one machine fully busy._