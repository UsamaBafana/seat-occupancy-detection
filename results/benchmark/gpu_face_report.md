# Compute benchmark — 2026-10-08 02:06

## Test setup

- Hardware: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel · 16 logical cores · 16.9 GB RAM · GPU: none
- Device used: `cpu` · YOLO weights `models/yolov8n.pt` @ imgsz 640 · tracker embedder `mobilenet`
- Mode `gpu`: YOLO on GPU (OpenVINO) · DeepSORT model on GPU (OpenVINO) · face recognition on CPU (TensorFlow) · OpenVINO device: Intel(R) Iris(R) Xe Graphics (iGPU)
- Face recognition: VGG-Face + mtcnn (cache 5.0s)
- Face results (sampled calls): person_1 199 (100%), unknown 1 (0%)
- Source: 1280x720 @ 30.0 fps, 1000 frames timed · avg 1.02 people tracked (max 2)

## Per-stage time (ms, one camera, one frame)

| Stage | Mean | p50 | p95 | Max | Samples |
|---|---:|---:|---:|---:|---:|
| Decode | 4.3 | 3.6 | 7.0 | 32.5 | 1000 |
| YOLO detect | 31.6 | 30.2 | 41.7 | 171.9 | 1000 |
| DeepSORT track | 23.9 | 23.3 | 36.9 | 751.6 | 1000 |
| Face (per person, uncached) | 827.7 | 807.0 | 1100.9 | 8122.3 | 200 |

## Load per camera (using mean)

Assumes stream 15.0 fps, occupancy update 1.0 fps, 1.02 people/camera, 0.20 face calls/s/camera.

| Component | ms of work per second |
|---|---:|
| Decode | 63.8 |
| Detect + track | 55.5 |
| Face recognition | 169.2 |
| **Total** | **288.5** |

**≈ 2 cameras per machine** like this one (keeping 30% headroom, single pipeline process).

## Scaling

| Cameras | Load (ms/s) | Machines needed | Bandwidth (Mbps) |
|---:|---:|---:|---:|
| 1 | 288 | 1 | 3 |
| 2 | 577 | 1 | 6 |
| 5 | 1442 | 3 | 15 |
| 10 | 2885 | 5 | 30 |
| 20 | 5769 | 9 | 60 |
| 50 | 14424 | 21 | 150 |
| 100 | 28847 | 42 | 300 |

## Memory

- Process RSS: 719 MB after model load → 3377 MB at end

_Load = milliseconds of compute needed every second. 1000 ms/s = one machine fully busy._