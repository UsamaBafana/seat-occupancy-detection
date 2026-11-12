"""
src/accel.py
Run parts of the pipeline on an Intel GPU (e.g. Iris Xe) through OpenVINO.

Used by benchmark.py --accel {cpu, hybrid, gpu}:
  cpu    : everything on the CPU with PyTorch (the original pipeline)
  hybrid : YOLO detection on the GPU, DeepSORT + face recognition on the CPU
  gpu    : YOLO detection AND the DeepSORT appearance model on the GPU,
           face recognition on the CPU (DeepFace/TensorFlow cannot use an
           Intel GPU without extra plugins)

Needs:  pip install openvino
The first run converts yolov8n.pt to OpenVINO format (a few seconds) and
saves it next to the weights as <name>_openvino_model/.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.detector import Detection


def ov_core():
    try:
        import openvino as ov
    except ImportError as e:
        raise ImportError("OpenVINO not installed. Run:  pip install openvino") from e
    return ov, ov.Core()


def available_devices() -> list[str]:
    _, core = ov_core()
    out = []
    for d in core.available_devices:
        try:
            out.append(f"{d}: {core.get_property(d, 'FULL_DEVICE_NAME')}")
        except Exception:
            out.append(d)
    return out


def check_device(device: str) -> str:
    """Fail early, with a clear message, if the OpenVINO device is missing."""
    _, core = ov_core()
    devs = core.available_devices
    base = device.split(".")[0]
    if base not in [d.split(".")[0] for d in devs]:
        raise RuntimeError(
            f"OpenVINO device '{device}' not found. Available: {devs}. "
            "For the Intel Iris Xe this should include 'GPU' - update the Intel "
            "graphics driver if it does not.")
    try:
        return core.get_property(device, "FULL_DEVICE_NAME")
    except Exception:
        return device


# ─────────────────────────────────────────────────────────────
#  YOLO on OpenVINO
# ─────────────────────────────────────────────────────────────

class OpenVINODetector:
    """Drop-in replacement for YOLOv8Detector that runs on an OpenVINO device."""

    PERSON_CLASS = "person"

    def __init__(self, weights: str, confidence: float = 0.45, iou_threshold: float = 0.5,
                 imgsz: int = 640, ov_device: str = "GPU", target_classes=None):
        ov, core = ov_core()
        self.confidence, self.iou_threshold, self.imgsz = confidence, iou_threshold, imgsz
        self.ov_device = ov_device
        self.device_name = check_device(ov_device)
        self.target_classes = target_classes or [self.PERSON_CLASS]

        w = Path(weights)
        xml = w.parent / f"{w.stem}_openvino_model" / f"{w.stem}.xml"
        from ultralytics import YOLO
        pt_model = YOLO(str(w))
        self.names = dict(pt_model.names)
        if not xml.exists():
            print(f"[accel] Converting {w} to OpenVINO (one-off) ...")
            if not hasattr(ov, "runtime"):      # newer OpenVINO removed ov.runtime, which older ultralytics uses
                import types
                ov.runtime = types.SimpleNamespace(save_model=ov.save_model, Core=ov.Core)
            out_dir = Path(pt_model.export(format="openvino", imgsz=imgsz, half=False, dynamic=False))
            xml = out_dir / f"{w.stem}.xml" if out_dir.is_dir() else out_dir
        self.target_ids = {cid for cid, n in self.names.items() if n in self.target_classes}
        model = core.read_model(str(xml))
        self.compiled = core.compile_model(model, ov_device, {"PERFORMANCE_HINT": "LATENCY"})
        self.request = self.compiled.create_infer_request()
        print(f"[accel] YOLO running on OpenVINO {ov_device} ({self.device_name})")

    def _letterbox(self, img):
        h, w = img.shape[:2]
        s = self.imgsz
        r = min(s / h, s / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (s - nw) / 2, (s - nh) / 2
        if (w, h) != (nw, nh):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, left, top

    def detect(self, frame: np.ndarray, extract_crops: bool = False) -> list[Detection]:
        img, r, padx, pady = self._letterbox(frame)
        x = img[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.request.infer({0: np.ascontiguousarray(x)})
        pred = next(iter(out.values()))[0]          # (4 + nc, N)
        pred = pred.T
        boxes, scores_all = pred[:, :4], pred[:, 4:]
        cls = scores_all.argmax(1)
        conf = scores_all[np.arange(len(cls)), cls]
        keep = conf >= self.confidence
        if self.target_ids:
            keep &= np.isin(cls, list(self.target_ids))
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
        if len(conf) == 0:
            return []
        xywh = boxes.copy()
        xywh[:, 0] -= xywh[:, 2] / 2
        xywh[:, 1] -= xywh[:, 3] / 2
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), self.confidence, self.iou_threshold)
        idx = np.array(idx).reshape(-1)
        h, w = frame.shape[:2]
        dets = []
        for i in idx:
            bx, by, bw, bh = xywh[i]
            x1 = int(np.clip((bx - padx) / r, 0, w)); y1 = int(np.clip((by - pady) / r, 0, h))
            x2 = int(np.clip((bx + bw - padx) / r, 0, w)); y2 = int(np.clip((by + bh - pady) / r, 0, h))
            crop = frame[y1:y2, x1:x2].copy() if (extract_crops and y2 > y1 and x2 > x1) else None
            dets.append(Detection(bbox=(x1, y1, x2, y2), confidence=float(conf[i]),
                                  class_id=int(cls[i]), class_name=self.names[int(cls[i])], crop=crop))
        return dets


# ─────────────────────────────────────────────────────────────
#  DeepSORT appearance model (MobileNetV2) on OpenVINO
# ─────────────────────────────────────────────────────────────

class _OVEmbedderModel:
    """Stands in for the torch MobileNetV2 inside deep_sort_realtime's embedder."""

    def __init__(self, torch_model, ov_device: str):
        import torch
        ov, core = ov_core()
        ex = torch.zeros(1, 3, 224, 224)
        self.dynamic = True
        try:
            m = ov.convert_model(torch_model.eval(), example_input=ex, input=[-1, 3, 224, 224])
            self.compiled = core.compile_model(m, ov_device)
        except Exception:
            self.dynamic = False
            m = ov.convert_model(torch_model.eval(), example_input=ex, input=[1, 3, 224, 224])
            self.compiled = core.compile_model(m, ov_device)
        self.request = self.compiled.create_infer_request()
        self._torch = torch

    def forward(self, batch):
        x = batch.detach().cpu().float().numpy()
        if self.dynamic:
            y = next(iter(self.request.infer({0: x}).values()))
        else:
            y = np.concatenate([next(iter(self.request.infer({0: x[i:i + 1]}).values())) for i in range(len(x))])
        return self._torch.from_numpy(np.array(y))

    __call__ = forward


def accelerate_tracker(tracker, ov_device: str = "GPU") -> str:
    """Move the DeepSORT appearance model of a DeepSORTTracker onto an OpenVINO device."""
    name = check_device(ov_device)
    emb = tracker._tracker.embedder
    if emb is None or not hasattr(emb, "model"):
        raise RuntimeError("Tracker has no MobileNet embedder to accelerate (embedder must be 'mobilenet').")
    emb.gpu, emb.half = False, False
    emb.model = _OVEmbedderModel(emb.model, ov_device)
    print(f"[accel] DeepSORT appearance model running on OpenVINO {ov_device} ({name})")
    return name
