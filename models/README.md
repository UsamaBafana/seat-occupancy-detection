# Model weights

| File | What it is | In the repository? |
|---|---|---|
| `best_v4.pt` | Custom person detector: YOLOv8n fine-tuned on four combined public occupancy datasets (report Section 5.2, Table 8). Set `model.custom_weights: "models/best_v4.pt"` in `config/settings.yaml` to use it. | Yes (about 6 MB) |
| `yolov8n.pt` | COCO-pretrained YOLOv8n from Ultralytics. Used for automatic chair/seat detection and, in the PoC test, for person detection (report Table 11). | No. Ultralytics downloads it automatically on first use. |
| `best_v1.pt` – `best_v3.pt` | Earlier training versions reported in Table 8. | No. Available from the author on request. |
| `yolov8n_openvino_model/` | OpenVINO export for Intel integrated-GPU runs. | No. Created automatically by `src/accel.py` the first time `scripts/benchmark.py --accel hybrid` or `--accel gpu` is run. |

If `yolov8n.pt` cannot be downloaded automatically, place it in this folder; `config/settings.yaml` expects `models/yolov8n.pt`.
