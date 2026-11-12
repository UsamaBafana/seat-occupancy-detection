"""
src/pipeline.py
Core occupancy detection pipeline.

Orchestrates:
  1. Frame capture (webcam / video / RTSP)
  2. YOLOv8 person detection
  3. DeepSORT object tracking
  4. Face recognition (optional)
  5. Seat–person association
  6. Temporal inference (false-positive suppression)
  7. Logging and visualisation
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from config.config_loader import Config
from src.detector import YOLOv8Detector
from src.face_recognition_module import FaceRecogniser
from src.occupancy_logger import OccupancyLogger
from src.seat_manager import SeatManager
from src.temporal_inference import SeatState, TemporalInferenceEngine
from src.tracker import DeepSORTTracker, Track


def _is_video_file(source) -> bool:
    """True for a recorded clip path (not a camera index or a network stream)."""
    s = str(source).strip()
    return not s.isdigit() and not s.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))


def _clip_time(cap) -> float:
    """Position of the frame just read, in seconds from the start of the clip."""
    return cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0


# ─────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────

class OccupancyPipeline:
    """
    End-to-end occupancy detection pipeline.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._frame_id = 0
        self._running = False

        # ── Sub-systems ───────────────────────────────────────
        weights = cfg.model.custom_weights or cfg.model.weights
        self.detector = YOLOv8Detector(
            weights=weights,
            confidence=cfg.model.confidence_threshold,
            iou_threshold=cfg.model.iou_threshold,
            device=cfg.model.device,
            imgsz=cfg.model.imgsz,
        )

        self.tracker = DeepSORTTracker(
            max_age=cfg.tracker.max_age,
            n_init=cfg.tracker.n_init,
            max_iou_distance=cfg.tracker.max_iou_distance,
            embedder=cfg.tracker.embedder,
        )

        self.face_rec = FaceRecogniser(
            known_faces_dir=cfg.face_recognition.known_faces_dir,
            model_name=cfg.face_recognition.model,
            detector_backend=cfg.face_recognition.detector_backend,
            distance_metric=cfg.face_recognition.distance_metric,
            threshold=cfg.face_recognition.threshold,
            min_face_size=cfg.face_recognition.min_face_size,
        )
        self._face_rec_enabled = cfg.face_recognition.enabled

        self.seat_mgr = SeatManager(
            overlap_threshold=cfg.seats.overlap_threshold,
            lower_body_portion=getattr(cfg.seats, "lower_body_portion", 0.40),
        )

        self.temporal = TemporalInferenceEngine(
            window_size=cfg.temporal.window_size,
            occupied_threshold=cfg.temporal.occupied_threshold,
            cooldown_frames=cfg.temporal.cooldown_frames,
        )

        self.occ_logger = OccupancyLogger(
            output_dir=cfg.io.output_dir,
            log_interval_sec=cfg.io.log_interval_seconds,
        )

        # Video writer (initialised lazily)
        self._video_writer: Optional[cv2.VideoWriter] = None

        logger.info("[Pipeline] Initialised ✓")

    # ── Seat setup ───────────────────────────────────────────

    def setup_seats(
        self,
        frame: Optional[np.ndarray] = None,
        load_json: Optional[str | Path] = None,
    ) -> None:
        """
        Load or interactively define seat ROIs, then register with temporal engine.
        Priority: load_json > config rois > interactive drawing.
        """
        if load_json and Path(load_json).exists():
            self.seat_mgr.load_from_json(load_json)
            logger.info(f"[Pipeline] Loaded {len(self.seat_mgr.seats)} seats from {load_json}")

        elif self.cfg.seats.rois:
            self.seat_mgr.load_from_config(self.cfg.seats.rois)
            logger.info(f"[Pipeline] Loaded {len(self.seat_mgr.seats)} seats from config")

        elif frame is not None:
            logger.info("[Pipeline] No seats defined — launching interactive ROI drawing")
            self.seat_mgr.draw_seats_interactive(frame)
            json_out = Path(self.cfg.io.output_dir) / "seats.json"
            self.seat_mgr.save_to_json(json_out)
            logger.info(f"[Pipeline] Seats saved → {json_out}")

        else:
            logger.warning("[Pipeline] No seats defined and no frame provided for drawing.")

        # Register with temporal engine
        seat_ids = [s.seat_id for s in self.seat_mgr.seats]
        self.temporal.register_seats(seat_ids)
        logger.info(f"[Pipeline] Registered {len(seat_ids)} seats with temporal engine")

    # ── Main run loop ────────────────────────────────────────

    def run(self) -> None:
        """Open video source and run the pipeline until stopped."""
        source = self.cfg.io.source
        cap = cv2.VideoCapture(source if isinstance(source, str) else int(source))

        # Recorded clip -> log video time (s from clip start) instead of the
        # wall clock, so results line up with labels from label_ground_truth.py
        self._is_file = _is_video_file(source)
        self._video_t: Optional[float] = None
        if self._is_file:
            logger.info("[Pipeline] Video file source - logging clip time (s) as timestamps")

        if not cap.isOpened():
            raise RuntimeError(f"[Pipeline] Cannot open video source: {source}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"[Pipeline] Stream: {width}×{height} @ {fps:.1f} fps")

        # ── Camera warm-up — discard early frames so auto-exposure settles ──
        warmup_frames = getattr(self.cfg.io, "warmup_frames", 30)
        for _ in range(warmup_frames):
            cap.read()

        # Interactive seat setup on first (now properly exposed) frame
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("[Pipeline] Could not read first frame.")
        if self._is_file:
            self._video_t = _clip_time(cap)

        if not self.seat_mgr.seats:
            self.setup_seats(frame=first_frame)

        # Init video writer
        if self.cfg.io.save_annotated_video:
            out_path = Path(self.cfg.io.output_dir) / "annotated.mp4"
            self._video_writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            logger.info(f"[Pipeline] Saving annotated video → {out_path}")

        self._running = True
        self._frame_id = 0
        last_annotated = None
        skip = getattr(self.cfg.io, "process_every_n_frames", 1)

        # Process the first frame we already read
        frames_to_process = [first_frame]

        logger.info("[Pipeline] Running — press 'q' to stop")

        while self._running:
            if frames_to_process:
                frame = frames_to_process.pop(0)
            else:
                ret, frame = cap.read()
                if not ret:
                    logger.info("[Pipeline] End of stream.")
                    break

            if self._is_file:
                self._video_t = _clip_time(cap)

            self._frame_id += 1

            # ── Frame skip — show last frame, skip heavy processing ──
            if self._frame_id % skip != 0:
                if self.cfg.display.show_window and last_annotated is not None:
                    cv2.imshow(self.cfg.display.window_title, last_annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._running = False
                continue

            annotated = self.process_frame(frame)
            last_annotated = annotated

            if self._video_writer:
                self._video_writer.write(annotated)

            if self.cfg.display.show_window:
                cv2.namedWindow(self.cfg.display.window_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.cfg.display.window_title, 1280, 720)
                cv2.imshow(self.cfg.display.window_title, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("[Pipeline] Quit requested by user.")
                    self._running = False

        # Cleanup
        self.stop()
        cap.release()

    def stop(self) -> None:
        self._running = False
        if self._video_writer:
            self._video_writer.release()
        cv2.destroyAllWindows()

        summary_path = self.occ_logger.save_summary()
        logger.info(f"[Pipeline] Summary → {summary_path}")
        logger.info(
            f"[Pipeline] Session ended. Logged {self.occ_logger.record_count} records."
        )

    # ── Single-frame processing ───────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run the full pipeline on one frame and return the annotated frame.
        Note: _frame_id is incremented in the run loop, not here.
        """
        # ── 1. Detect persons ────────────────────────────────
        detections = self.detector.detect(frame, extract_crops=True)

        # ── 2. Track ─────────────────────────────────────────
        tracks = self.tracker.update(detections, frame)

        # ── 3. Face recognition ──────────────────────────────
        if self._face_rec_enabled and self.face_rec.is_available():
            for track in tracks:
                if track.crop is not None:
                    identity = self.face_rec.identify_track(track.track_id, track.crop)
                    track.face_identity = identity

        # ── 4. Seat–person association ────────────────────────
        occupied_seat_ids: set[str] = set()
        occupant_map: dict[str, str] = {}

        for track in tracks:
            matched_seats = self.seat_mgr.match_person_to_seat(track.bbox)
            for seat in matched_seats:
                occupied_seat_ids.add(seat.seat_id)
                if track.face_identity and track.face_identity != "unknown":
                    occupant_map[seat.seat_id] = track.face_identity

        # ── 5. Temporal inference ─────────────────────────────
        states = self.temporal.update(occupied_seat_ids, occupant_map)

        # ── 6. Update seat manager visual state ──────────────
        for seat in self.seat_mgr.seats:
            seat.current_state = states.get(seat.seat_id, SeatState.VACANT).value
            seat.occupant_id = occupant_map.get(seat.seat_id)

        # ── 7. Log ────────────────────────────────────────────
        if self.cfg.io.save_occupancy_log:
            confidences = {
                sid: buf.confidence
                for sid, buf in self.temporal._buffers.items()
            }
            self.occ_logger.log_frame(
                frame_id=self._frame_id,
                seat_states=states,
                confidences=confidences,
                occupants=occupant_map,
                timestamp=getattr(self, "_video_t", None),
            )

        # ── 8. Annotate frame ─────────────────────────────────
        annotated = self._annotate(frame.copy(), tracks, states)
        return annotated

    # ── Annotation ───────────────────────────────────────────

    def _annotate(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        states: dict[str, SeatState],
    ) -> np.ndarray:
        cfg_d = self.cfg.display

        # Seat ROIs with enhanced visuals
        if cfg_d.draw_seat_rois:
            frame = self._draw_seats_enhanced(frame, states)

        # Person tracks
        if cfg_d.draw_tracks:
            frame = DeepSORTTracker.draw_tracks(frame, tracks)

        # HUD overlay
        frame = self._draw_hud(frame, states)
        return frame

    def _draw_seats_enhanced(
        self,
        frame: np.ndarray,
        states: dict[str, SeatState],
    ) -> np.ndarray:
        overlay = frame.copy()

        # First pass — semi-transparent fill
        for seat in self.seat_mgr.seats:
            x1, y1, x2, y2 = seat.bbox
            state = states.get(seat.seat_id, SeatState.VACANT)
            is_occupied = (state == SeatState.OCCUPIED)
            fill_color = (0, 0, 180) if is_occupied else (0, 160, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)

        # Blend at 25% opacity
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Second pass — border, label, confidence bar
        for seat in self.seat_mgr.seats:
            x1, y1, x2, y2 = seat.bbox
            state = states.get(seat.seat_id, SeatState.VACANT)
            is_occupied = (state == SeatState.OCCUPIED)

            border_color = (0, 60, 255) if is_occupied else (0, 220, 80)
            status_text  = "OCCUPIED" if is_occupied else "VACANT"

            # Border
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

            # Label background + text
            label = f"{seat.seat_id}  {status_text}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), border_color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Confidence bar at bottom of ROI
            buf = self.temporal._buffers.get(seat.seat_id)
            if buf:
                bar_w = x2 - x1
                fill_w = int(bar_w * buf.occupancy_ratio)
                cv2.rectangle(frame, (x1, y2 - 6), (x2, y2), (50, 50, 50), -1)
                cv2.rectangle(frame, (x1, y2 - 6), (x1 + fill_w, y2), border_color, -1)

        return frame

    def _draw_hud(
        self,
        frame: np.ndarray,
        states: dict[str, SeatState],
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        occupied = self.temporal.occupied_count
        total    = self.temporal.total_seats
        vacant   = total - occupied
        util     = (occupied / total * 100) if total else 0

        # Semi-transparent top banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 56), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title
        cv2.putText(frame, "OFFICE OCCUPANCY MONITOR",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)

        # Stats row
        cv2.putText(frame, f"Seats: {total}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        occ_color = (0, 80, 255) if occupied > 0 else (200, 200, 200)
        cv2.putText(frame, f"Occupied: {occupied}",
                    (120, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, occ_color, 1)

        cv2.putText(frame, f"Vacant: {vacant}",
                    (260, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 80), 1)

        cv2.putText(frame, f"Utilisation: {util:.0f}%",
                    (370, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        cv2.putText(frame, f"Frame: {self._frame_id:05d}",
                    (w - 140, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        return frame