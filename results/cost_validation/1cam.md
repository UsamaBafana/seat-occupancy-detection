# Cost model validation — 2026-10-07 18:28

## Live run

- 1 camera(s), device `cpu`, imgsz 640, detect every 3 frame(s), face recognition off
- 186 s measured (first 5 s skipped)
- Processed **5.94 frames/s per camera**, avg **0.98 people per camera**

## Predicted vs measured compute (ms of work per second, all cameras)

| Stage | Predicted (formula + benchmark) | Measured (live) | Error |
|---|---:|---:|---:|
| YOLO detect | 473 | 424 | +11% |
| DeepSORT track | 233 | 211 | +10% |
| Face recognition | 0 | 0 | – |
| **Total** | 706 | 635 | +11% |

## Per-frame times (ms)

| Stage | Benchmark (offline clip) | Live run | Error |
|---|---:|---:|---:|
| YOLO detect | 79.6 | 75.4 | +6% |
| DeepSORT track | 39.3 | 38.7 | +2% |

## Capacity at the live run's rates

| | Formula | Live measurement |
|---|---:|---:|
| Compute per camera (ms/s) | 706 | 635 |
| Cameras per machine (70% of loop) | 0 | 1 |

## Machine resources during the live run

| Metric | Mean | Peak |
|---|---:|---:|
| Process CPU (% of whole machine) | 48.702 | 52.4 |
| Process CPU (cores busy) | 7.794 | 8.39 |
| System CPU (%) | 54.274 | 67.2 |
| Process RAM (MB) | 590.683 | 602.0 |

_Machine: 16 logical cores. Error = (predicted − measured) / measured._

**Verdict:** the formula agrees well with the live run (total compute error 11%).
