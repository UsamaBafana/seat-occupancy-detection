"""
src/detector.py
YOLOv8-based person detector.

Wraps Ultralytics YOLO to produce clean Detection objects used
by the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

# ─────────────────────────────────────────────────────────────
#  Detection dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class Detection:
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2  (pixel coords)
    confidence: float
    class_id: int
    class_name: str

    # Optional crop for downstream use
    crop: Optional[np.ndarray] = None

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox
        return x1, y1, x2 - x1, y2 - y1

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


# ─────────────────────────────────────────────────────────────
#  Detector
# ─────────────────────────────────────────────────────────────

class YOLOv8Detector:
    """
    Wraps Ultralytics YOLOv8 for person detection.

    Parameters
    ----------
    weights       : Path to .pt file (pretrained or fine-tuned)
    confidence    : Minimum detection confidence
    iou_threshold : NMS IoU threshold
    device        : 'cpu', 'cuda', 'mps', or '' (auto)
    imgsz         : Inference image size
    target_classes: List of class names to keep (e.g. ['person'])
                    None = keep all classes
    """

    PERSON_CLASS = "person"

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.45,
        iou_threshold: float = 0.50,
        device: str = "cpu",
        imgsz: int = 640,
        target_classes: list[str] | None = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics") from e

        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.imgsz = imgsz
        self.target_classes = target_classes or [self.PERSON_CLASS]

        weights_path = Path(weights)
        logger.info(f"[Detector] Loading weights: {weights_path}")
        self.model = YOLO(str(weights_path))

        # Build set of target class IDs from the model's names
        self.target_class_ids: set[int] = set()
        for cid, cname in self.model.names.items():
            if cname in self.target_classes:
                self.target_class_ids.add(cid)

        logger.info(
            f"[Detector] Target classes: {self.target_classes} → IDs {self.target_class_ids}"
        )

    # ── Public API ────────────────────────────────────────────

    def detect(self, frame: np.ndarray, extract_crops: bool = False) -> list[Detection]:
        """
        Run inference on a single BGR frame.

        Returns a list of Detection objects filtered to target_classes.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        for box in result.boxes:
            cid = int(box.cls.item())
            if self.target_class_ids and cid not in self.target_class_ids:
                continue

            conf = float(box.conf.item())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

            crop = None
            if extract_crops:
                crop = self._safe_crop(frame, x1, y1, x2, y2)

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cid,
                    class_name=self.model.names[cid],
                    crop=crop,
                )
            )

        return detections

    def detect_batch(
        self, frames: list[np.ndarray]
    ) -> list[list[Detection]]:
        """Batch inference (more efficient for offline processing)."""
        all_results = self.model.predict(
            source=frames,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        output = []
        for result in all_results:
            frame_dets: list[Detection] = []
            if result.boxes:
                for box in result.boxes:
                    cid = int(box.cls.item())
                    if self.target_class_ids and cid not in self.target_class_ids:
                        continue
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    frame_dets.append(
                        Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_id=cid,
                            class_name=self.model.names[cid],
                        )
                    )
            output.append(frame_dets)
        return output

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _safe_crop(
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        pad: int = 10,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        x1c = max(0, x1 - pad)
        y1c = max(0, y1 - pad)
        x2c = min(w, x2 + pad)
        y2c = min(h, y2 + pad)
        return frame[y1c:y2c, x1c:x2c].copy()

    # ── Visualisation ─────────────────────────────────────────

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: list[Detection],
        color: tuple = (255, 160, 0),
    ) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv_color = color
            cv2_import()
            import cv2
            cv2.rectangle(frame, (x1, y1), (x2, y2), cv_color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cv_color, 1)
        return frame


def cv2_import():
    """Lazy cv2 import guard (already available in env)."""
    pass
