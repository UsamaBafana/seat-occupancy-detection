"""
src/opportunistic_calibration.py
Gathers cam0<->cam1 correspondence points "for free" during normal
operation, instead of a dedicated calibration session — matches the
design described in the final report (Section 4.7): whenever a
manually-confirmed shared seat is settled-occupied by the same person
in both camera views at once, the person's own ground-contact position
in each camera's frame is one valid correspondence point pair. No
marker, calibration object, or dedicated session is needed.

Once enough pairs have accumulated (>= min_points), computes a
homography via cv2.findHomography and saves it in the SAME file format
calibrate_homography.py / auto_match_seats_homography.py already use,
so that tooling keeps working unchanged on top of this.

HONEST LIMITATION: with only one shared desk in the current test setup,
captured points cluster near that one physical location rather than
being spread across the room, which makes the resulting homography less
numerically robust than points gathered from multiple distinct desks
over time. The mechanism matches what the report describes; calibration
quality is expected to improve as more distinct shared-desk locations
are added.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


class OpportunisticCalibrator:
    def __init__(
        self,
        out_path: str | Path = "config/homography_cam1_to_cam0.json",
        min_points: int = 4,
        settle_ratio: float = 0.85,
        capture_interval_ticks: int = 30,
        max_stored_points: int = 60,
        cam0_prefix: str = "cam0",
        cam1_prefix: str = "cam1",
    ):
        self.out_path = Path(out_path)
        self.min_points = min_points
        self.settle_ratio = settle_ratio
        self.capture_interval_ticks = capture_interval_ticks
        self.max_stored_points = max_stored_points
        self.cam0_prefix = cam0_prefix
        self.cam1_prefix = cam1_prefix

        self.pts0: list[list[float]] = []
        self.pts1: list[list[float]] = []
        self._last_capture_tick: dict[str, int] = {}
        self._calibrated = False

        self._load_existing()

    def _load_existing(self) -> None:
        """Resume accumulating points across sessions — this is what
        makes it genuinely opportunistic rather than a one-shot event;
        points gathered today and points gathered next week both count."""
        if not self.out_path.exists():
            return
        try:
            data = json.loads(self.out_path.read_text())
            pairs = data.get("point_pairs")
            if pairs:
                self.pts0 = [list(p) for p in pairs.get("cam0", [])]
                self.pts1 = [list(p) for p in pairs.get("cam1", [])]
                self._calibrated = "homography_cam1_to_cam0" in data
                logger.info(f"[Calibration] Resumed {len(self.pts0)} existing point pair(s) "
                            f"from {self.out_path}")
        except Exception as exc:
            logger.debug(f"[Calibration] Could not load existing calibration file: {exc}")

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def maybe_capture(
        self,
        seat_id: str,
        point_cam0: tuple[float, float],
        point_cam1: tuple[float, float],
        occupancy_ratio: float,
        tick: int,
    ) -> None:
        """
        Call once per processed tick for any seat currently seen by both
        cameras at once (only meaningful for a manually-confirmed shared
        seat, where both cameras already agree it's one physical desk —
        see multi_pipeline.py's seat_camera_points collection). Captures
        a correspondence point pair if the seat looks settled-occupied
        and enough ticks have passed since the last capture for this
        seat, to avoid dumping near-duplicate points from one long sit.
        """
        if occupancy_ratio < self.settle_ratio:
            return

        last = self._last_capture_tick.get(seat_id, -10**9)
        if tick - last < self.capture_interval_ticks:
            return
        self._last_capture_tick[seat_id] = tick

        self.pts0.append([float(point_cam0[0]), float(point_cam0[1])])
        self.pts1.append([float(point_cam1[0]), float(point_cam1[1])])
        if len(self.pts0) > self.max_stored_points:
            self.pts0 = self.pts0[-self.max_stored_points:]
            self.pts1 = self.pts1[-self.max_stored_points:]

        logger.info(f"[Calibration] Captured correspondence point pair from '{seat_id}' "
                    f"({len(self.pts0)}/{self.min_points}+ collected)")

        self._save()
        if len(self.pts0) >= self.min_points:
            self._try_compute()

    def _try_compute(self) -> bool:
        src = np.array(self.pts1, dtype=np.float32)
        dst = np.array(self.pts0, dtype=np.float32)
        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if H is None:
            logger.debug("[Calibration] Not enough geometric spread yet to solve a homography.")
            return False

        inliers = int(mask.sum())
        self._calibrated = True
        logger.info(f"[Calibration] Homography computed from opportunistically-gathered "
                    f"points \u2014 {inliers}/{len(self.pts0)} inliers. Saved -> {self.out_path}")
        self._save(H=H, inliers=inliers)
        return True

    def _save(self, H: np.ndarray | None = None, inliers: int | None = None) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "point_pairs": {"cam0": self.pts0, "cam1": self.pts1},
            "total_points": len(self.pts0),
            "source": "opportunistic",
        }
        if H is not None:
            data["homography_cam1_to_cam0"] = H.tolist()
            data["inliers"] = inliers
        with open(self.out_path, "w") as f:
            json.dump(data, f, indent=2)
