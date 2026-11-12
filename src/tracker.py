"""
src/tracker.py
DeepSORT-based multi-object tracker.

Wraps deep_sort_realtime to produce clean Track objects compatible
with the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

# ─────────────────────────────────────────────────────────────
#  Track dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float
    class_name: str = "person"
    age: int = 0                       # frames since confirmed
    is_confirmed: bool = True

    # Enriched downstream
    face_identity: Optional[str] = field(default=None, repr=False)
    crop: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def bottom_center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, y2


# ─────────────────────────────────────────────────────────────
#  Tracker
# ─────────────────────────────────────────────────────────────

class DeepSORTTracker:
    """
    Lightweight wrapper around deep_sort_realtime.

    Parameters
    ----------
    max_age         : Max frames to keep a track alive without a match
    n_init          : Consecutive detections needed to confirm a track
    max_iou_dist    : Max IoU distance for association
    embedder        : Appearance backbone ('mobilenet', 'torchreid', etc.)
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        embedder: str = "mobilenet",
    ):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as e:
            raise ImportError(
                "deep_sort_realtime not installed. Run: pip install deep-sort-realtime"
            ) from e

        logger.info(
            f"[Tracker] Initialising DeepSORT  max_age={max_age}  "
            f"n_init={n_init}  embedder={embedder}"
        )

        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder=embedder,
            half=False,
            bgr=True,
        )

        self._frame_count = 0

    # ── Public API ────────────────────────────────────────────

    def update(
        self,
        detections,               # list[Detection] from detector.py
        frame: np.ndarray,
    ) -> list[Track]:
        """
        Feed new detections and return current active tracks.

        Parameters
        ----------
        detections : list of detector.Detection
        frame      : current BGR frame (used for appearance embedding)

        Returns
        -------
        list[Track] — confirmed tracks with updated bboxes
        """
        self._frame_count += 1

        # Convert to DeepSORT format: ([x,y,w,h], confidence, class_name)
        raw_dets = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            raw_dets.append(
                ([x1, y1, x2 - x1, y2 - y1], det.confidence, det.class_name)
            )

        raw_tracks = self._tracker.update_tracks(raw_dets, frame=frame)

        tracks: list[Track] = []
        for rt in raw_tracks:
            if not rt.is_confirmed():
                continue

            ltrb = rt.to_ltrb()
            x1, y1, x2, y2 = (int(v) for v in ltrb)

            # Clamp to frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2].copy() if (y2 > y1 and x2 > x1) else None

            tracks.append(
                Track(
                    track_id=rt.track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=rt.det_conf if rt.det_conf is not None else 1.0,
                    class_name=rt.det_class or "person",
                    age=rt.age,
                    is_confirmed=True,
                    crop=crop,
                )
            )

        return tracks

    def reset(self) -> None:
        """Clear all tracks (call when switching video sources)."""
        self._tracker = self._tracker.__class__(
            max_age=self._tracker.max_age,
            n_init=self._tracker.n_init,
        )
        self._frame_count = 0

    # ── Visualisation ─────────────────────────────────────────

    @staticmethod
    def draw_tracks(
        frame: np.ndarray,
        tracks: list[Track],
        color: tuple = (0, 165, 255),
    ) -> np.ndarray:
        import cv2

        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_parts = [f"ID:{t.track_id}"]
            if t.face_identity:
                label_parts.append(t.face_identity)

            label = " | ".join(label_parts)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

            # Draw centre dot
            cx, cy = t.center
            cv2.circle(frame, (cx, cy), 4, color, -1)

        return frame
