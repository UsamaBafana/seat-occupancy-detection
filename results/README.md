# Results

Saved outputs behind the figures in the final report. Video recordings, face
images and raw live logs are not included, because they show the participants.
The face identity in the benchmark files has been replaced with `person_1`.

## `poc_evaluation/` – proof-of-concept test (report Section 5.8, Table 12)

| File | Content |
|---|---|
| `ground_truth_events.csv` | Manually labelled sit-down / departure times per seat (clip time, seconds). |
| `ground_truth_per_second.csv` | The same labels expanded to one row per seat per second (354 seat-seconds). |
| `system_log_cliptime.csv` | The system's occupancy log for the evaluated run, mapped onto the video timeline. |
| `eval_overall.csv`, `eval_per_seat.csv`, `eval_confusion_matrix.csv` | Evaluation output (Table 12, all seat-seconds). |
| `seat_time_summary.csv` | Seconds each seat spent in each state during the run. |
| `resources_summary.json` | CPU, memory and per-stage time logged during the run (report Section 5.8). |

The Table 12 figures can be recomputed from these files:

```bash
python main.py evaluate --gt results/poc_evaluation/ground_truth_per_second.csv \
    --pred results/poc_evaluation/system_log_cliptime.csv --direction backward --tolerance 10
```

This check is also run automatically by `tests/test_metrics.py`.

## `benchmark/` – computation cost (report Section 5.9, Tables 13 and 14)

Output of `scripts/benchmark.py` on the test laptop (Intel Core i7-1260P, 16 GB RAM,
Intel Iris Xe integrated graphics), on a 1,000-frame 1280×720 clip with one occupant.

| Files | Configuration (Table 14 column) | Face recognition |
|---|---|---|
| `cpu_noface_*` | CPU only | off |
| `cpu_face_*` | CPU only | on |
| `hybrid_noface_*` | CPU + GPU (detection on the GPU) | off |
| `hybrid_face_*` | CPU + GPU (detection on the GPU) | on |
| `gpu_noface_*` | GPU (detection and DeepSORT appearance network on the GPU) | off |
| `gpu_face_*` | GPU (detection and DeepSORT appearance network on the GPU) | on |

Each run has a readable `_report.md` and a machine-readable `_summary.json`.

Note on `gpu_noface`: its saved mean DeepSORT time (50.4 ms) is raised by a single
23.4 s stall in one frame; the median is 22.8 ms. Table 14 reports 22.0 ms for this
configuration.

## `cost_validation/` – cost model against live runs

Output of `scripts/validate_cost_model.py`, comparing the compute predicted from the
benchmark with what one- and two-camera live runs actually used.
