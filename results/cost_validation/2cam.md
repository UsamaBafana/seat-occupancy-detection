# Cost model validation — 2026-10-07 18:28

## Live run

- 2 camera(s), device `cpu`, imgsz 640, detect every 3 frame(s), face recognition off
- 327 s measured (first 5 s skipped)
- Processed **2.65 frames/s per camera**, avg **0.97 people per camera**

## Predicted vs measured compute (ms of work per second, all cameras)

| Stage | Predicted (formula + benchmark) | Measured (live) | Error |
|---|---:|---:|---:|
| YOLO detect | 423 | 350 | +21% |
| DeepSORT track | 208 | 190 | +10% |
| Face recognition | 0 | 0 | – |
| **Total** | 631 | 540 | +17% |

## Per-frame times (ms)

| Stage | Benchmark (offline clip) | Live run | Error |
|---|---:|---:|---:|
| YOLO detect | 79.6 | 66.8 | +19% |
| DeepSORT track | 39.3 | 37.0 | +6% |

## Capacity at the live run's rates

| | Formula | Live measurement |
|---|---:|---:|
| Compute per camera (ms/s) | 316 | 270 |
| Cameras per machine (70% of loop) | 2 | 2 |

## Machine resources during the live run

| Metric | Mean | Peak |
|---|---:|---:|
| Process CPU (% of whole machine) | 47.736 | 51.8 |
| Process CPU (cores busy) | 7.637 | 8.28 |
| System CPU (%) | 53.581 | 77.6 |
| Process RAM (MB) | 638.997 | 655.0 |

_Machine: 16 logical cores. Error = (predicted − measured) / measured._

**Verdict:** the formula agrees roughly with the live run (total compute error 17%).
