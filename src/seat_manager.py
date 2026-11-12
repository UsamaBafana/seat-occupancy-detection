"""
src/seat_manager.py
Manages seat Regions of Interest (ROIs).

Seats can be:
  1. Manually defined in config (list of bboxes)
  2. Interactively drawn on the first frame
  3. Auto-detected by running a general COCO YOLOv8 model and proposing one
     draft seat per detected chair (COCO class 56). Two flavours:
       - auto_detect_seats()      : one frame  (fast, but fragile)
       - auto_detect_seats_live() : sample several frames with a LIVE preview
                                    and keep only chairs seen consistently
     Either way, point it at an EMPTY scene — a seated person occludes the
     chair — and treat the output as drafts the operator reviews/nudges.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class SeatROI:
    seat_id: str
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    label: str = "unknown"            # optional friendly label

    # runtime state (not serialised)
    current_state: str = field(default="vacant", repr=False)
    occupant_id: Optional[str] = field(default=None, repr=False)

    # ── helpers ──────────────────────────────────────────────
    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def iou_with(self, other_bbox: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other_bbox

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0

        union = self.area + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    def overlap_fraction(self, other_bbox: tuple[int, int, int, int]) -> float:
        """Fraction of the person bbox that falls inside this seat ROI."""
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other_bbox

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        person_area = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / person_area

    def to_dict(self) -> dict:
        return {"seat_id": self.seat_id, "bbox": list(self.bbox), "label": self.label}


# ─────────────────────────────────────────────────────────────
#  Manager
# ─────────────────────────────────────────────────────────────

class SeatManager:
    """
    Holds all seat ROIs and provides serialisation / interactive drawing.
    """

    def __init__(self, overlap_threshold: float = 0.30):
        self.seats: list[SeatROI] = []
        self.overlap_threshold = overlap_threshold
        self._counter = 0
        # Last frame seen by a live auto-detect (handy for previewing results).
        self.last_frame: Optional[np.ndarray] = None

    # ── Loading ───────────────────────────────────────────────
    def load_from_config(self, rois_config: list[dict]) -> None:
        """Load seats defined in settings.yaml."""
        for item in rois_config:
            self.add_seat(
                seat_id=item.get("id", f"seat_{self._counter:02d}"),
                bbox=tuple(item["bbox"]),
                label=item.get("label", ""),
            )

    def load_from_json(self, path: str | Path) -> None:
        """Load seats from a saved JSON file."""
        path = Path(path)
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        for item in data:
            self.add_seat(
                seat_id=item["seat_id"],
                bbox=tuple(item["bbox"]),
                label=item.get("label", ""),
            )

    def save_to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([s.to_dict() for s in self.seats], f, indent=2)

    # ── Auto-detection: single frame ──────────────────────────
    def auto_detect_seats(
        self,
        frame: np.ndarray,
        detector=None,
        weights: str = "yolov8n.pt",
        confidence: float = 0.25,
        device: str = "cpu",
        imgsz: int = 640,
        min_area: int = 1500,
        dedup_iou: float = 0.70,
        label: str = "chair",
        replace: bool = False,
    ) -> list[SeatROI]:
        """
        Propose seat ROIs by detecting chairs on a SINGLE frame.

        Run this on an EMPTY frame: once someone sits down they occlude the
        chair, so chairs detect most reliably before anyone is seated.

        Fast, but fragile — one bad frame (glare, a passer-by) gives bad
        seats. For a steadier result use auto_detect_seats_live().

        Parameters
        ----------
        frame      : a BGR frame (ideally of empty desks)
        detector   : an existing YOLOv8Detector to reuse. If None, one is
                     built from `weights`. IMPORTANT: this must be a general
                     COCO model (e.g. yolov8n.pt) — NOT your person-only
                     best_v4.pt, which has no 'chair' class.
        weights    : weights to load if `detector` is None
        confidence : min detection confidence (chairs are often partly hidden,
                     so this defaults lower than the person detector)
        min_area   : drop boxes smaller than this many pixels (removes specks)
        dedup_iou  : merge near-duplicate boxes above this IoU
        label      : COCO class name to look for ('chair' == class id 56)
        replace    : if True, clear existing seats before adding the drafts

        Returns
        -------
        list[SeatROI] : the draft seats that were added (reading order)
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            print("[SeatManager] auto_detect_seats: empty frame — nothing to do.")
            return []

        detector = self._ensure_chair_detector(
            detector, weights, confidence, device, imgsz, label
        )
        if detector is None:
            return []

        boxes = self._chair_boxes(detector.detect(frame), min_area)
        boxes = self._dedup_boxes(boxes, iou_thresh=dedup_iou)
        boxes = self._reading_order(boxes)

        if replace:
            self.clear()
        added = [self.add_seat(bbox=b, label=label) for b in boxes]

        print(f"[SeatManager] Auto-detected {len(added)} chair(s) as draft seat ROIs.")
        if added:
            print("              Review/adjust with draw_seats_interactive() "
                  "before committing.")
        else:
            print("              No chairs found — try a clearer/emptier frame "
                  "or lower confidence.")
        return added

    # ── Auto-detection: live video (temporal voting + preview) ─
    def auto_detect_seats_live(
        self,
        source=0,
        detector=None,
        weights: str = "yolov8n.pt",
        confidence: float = 0.25,
        device: str = "cpu",
        imgsz: int = 640,
        frames: int = 30,
        warmup: int = 30,
        sample_every: int = 3,
        min_ratio: float = 0.60,
        match_iou: float = 0.50,
        min_area: int = 1500,
        dedup_iou: float = 0.70,
        label: str = "chair",
        preview: bool = True,
        replace: bool = False,
    ) -> list[SeatROI]:
        """
        Like auto_detect_seats, but watches LIVE video and keeps only the
        chairs that show up consistently. A still chair appears in (almost)
        every frame; a person walking through or a one-off glitch appears in
        only a few, so it is voted out.

        With preview=True a window shows the live camera the whole time, with
        candidate boxes drawn on it:
          • green  = currently passing the vote (will be kept)
          • orange = still tentative (seen, but not in enough frames yet)
        Press 'q' in the window to stop sampling early.

        Still point it at an EMPTY scene: temporal voting removes noise, but it
        cannot see a chair through a seated person (occluded in every frame).

        Parameters
        ----------
        source       : webcam index (0), video path, or rtsp:// URL
        frames       : number of frames to actually run detection on
        warmup       : frames to read and discard first, so the camera's
                       auto-exposure / white-balance can settle
        sample_every : run detection on 1 of every N frames read, to spread
                       the sample over a few seconds rather than one burst
        min_ratio    : keep a chair only if seen in at least this fraction of
                       the sampled frames (0.60 = 60%)
        match_iou    : two boxes across frames are the same chair if their IoU
                       is at least this
        preview      : show the live camera window during detection
        (other params behave as in auto_detect_seats)

        Returns
        -------
        list[SeatROI] : the confirmed draft seats (reading order)
        """
        detector = self._ensure_chair_detector(
            detector, weights, confidence, device, imgsz, label
        )
        if detector is None:
            return []

        cap = cv2.VideoCapture(
            int(source) if str(source).isdigit() else source,
            cv2.CAP_DSHOW if str(source).isdigit() else 0,
        )
        if not cap.isOpened():
            print(f"[SeatManager] Auto-detect (live): cannot open source: {source}")
            return []

        win = "Auto-detecting seats (live)"
        if preview:
            cv2.namedWindow(win)

        # Warm up — throw away early frames while the camera settles, but show
        # them so the window appears immediately rather than going blank.
        for _ in range(max(0, warmup)):
            ok, frame = cap.read()
            if preview and ok:
                view = frame.copy()
                self._banner(view, "Warming up camera...")
                cv2.imshow(win, view)
                cv2.waitKey(1)

        # Each "track" gathers boxes that refer to the same physical chair.
        # frames_seen is a set of frame indices, so two detections of one chair
        # in a single frame don't double-count.
        tracks: list[dict] = []
        sampled = 0
        read_idx = 0
        last_frame = None
        stopped_early = False

        print(f"[SeatManager] Live auto-detect: sampling {frames} frame(s) "
              f"(1 in every {sample_every}). Press 'q' in the window to stop.")

        while sampled < frames:
            ok, frame = cap.read()
            if not ok:
                break
            read_idx += 1

            # Run detection only on sampled frames (keeps it fast), but show
            # every frame so the live feed stays smooth.
            if read_idx % max(1, sample_every) == 0:
                sampled += 1
                last_frame = frame
                for box in self._chair_boxes(detector.detect(frame), min_area):
                    best, best_iou = None, 0.0
                    for t in tracks:
                        iou = self._box_iou(box, t["mean"])
                        if iou > best_iou:
                            best, best_iou = t, iou
                    if best is not None and best_iou >= match_iou:
                        best["boxes"].append(box)
                        best["frames_seen"].add(sampled)
                        best["mean"] = self._mean_box(best["boxes"])
                    else:
                        tracks.append({"boxes": [box], "frames_seen": {sampled}, "mean": box})

                if sampled % 5 == 0:
                    print(f"  ... {sampled}/{frames} frames sampled, "
                          f"{len(tracks)} candidate chair(s)")

            if preview:
                view = frame.copy()
                for t in tracks:
                    x1, y1, x2, y2 = t["mean"]
                    strong = sampled > 0 and len(t["frames_seen"]) / sampled >= min_ratio
                    color = (0, 200, 80) if strong else (0, 165, 255)  # green / orange
                    cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
                self._banner(
                    view,
                    f"Sampling {sampled}/{frames}  |  candidates: {len(tracks)}  "
                    f"|  green=keep orange=tentative  |  q=stop",
                )
                cv2.imshow(win, view)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    stopped_early = True
                    print("[SeatManager] Live auto-detect: stopped early by user.")
                    break

        cap.release()
        self.last_frame = last_frame

        if sampled == 0:
            if preview:
                cv2.destroyWindow(win)
            print("[SeatManager] Auto-detect (live): no frames captured.")
            return []

        # Diagnostic: list every candidate and how often it was seen, so you
        # can see exactly why each was kept or dropped (and tune min_ratio).
        keep_pct = int(min_ratio * 100)
        print(f"[SeatManager] Candidate summary  ({sampled} frames sampled, "
              f"keep if seen >= {keep_pct}%):")
        if not tracks:
            print("    (no chair candidates at all)")
        for t in sorted(tracks, key=lambda t: len(t["frames_seen"]), reverse=True):
            ratio = len(t["frames_seen"]) / sampled
            status = "KEEP" if ratio >= min_ratio else "drop"
            print(f"    [{status}] seen {ratio * 100:5.0f}%   box={t['mean']}")

        # Keep chairs that appeared in at least `min_ratio` of sampled frames.
        confirmed = [
            self._mean_box(t["boxes"])
            for t in tracks
            if len(t["frames_seen"]) / sampled >= min_ratio
        ]
        confirmed = self._dedup_boxes(confirmed, iou_thresh=dedup_iou)
        confirmed = self._reading_order(confirmed)

        if replace:
            self.clear()
        added = [self.add_seat(bbox=b, label=label) for b in confirmed]

        note = " (stopped early)" if stopped_early else ""
        print(f"[SeatManager] Live auto-detect: {len(added)} chair(s) confirmed "
              f"(seen in >= {int(min_ratio * 100)}% of {sampled} sampled frames){note}.")
        if added:
            print("              Review/adjust with draw_seats_interactive() "
                  "before committing.")
        else:
            print("              Nothing confirmed — lower min_ratio/confidence, "
                  "or check the scene is empty and well lit.")

        # Final freeze-frame so you can see the committed seats before it closes.
        if preview:
            if last_frame is not None:
                final = self.draw_on_frame(last_frame.copy())
                self._banner(final, f"Done: {len(added)} seat(s). Press any key to close.")
                cv2.imshow(win, final)
                cv2.waitKey(0)
            cv2.destroyWindow(win)

        return added

    # ── Auto-detection helpers ────────────────────────────────
    @staticmethod
    def _banner(img: np.ndarray, text: str) -> None:
        """Draw a dark top strip with a line of status text (in place)."""
        w = img.shape[1]
        cv2.rectangle(img, (0, 0), (w, 30), (20, 20, 20), -1)
        cv2.putText(img, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (240, 240, 240), 1)

    @staticmethod
    def _ensure_chair_detector(detector, weights, confidence, device, imgsz, label):
        """
        Return a detector that can find `label`, building one from COCO weights
        if needed. Returns None (and prints why) if the model has no such class
        — this stops a person-only model from silently labelling people as
        chairs.
        """
        if detector is None:
            from src.detector import YOLOv8Detector  # lazy import avoids cycles
            detector = YOLOv8Detector(
                weights=weights,
                confidence=confidence,
                device=device,
                imgsz=imgsz,
                target_classes=[label],   # 'chair' resolves to COCO id 56
            )

        if not getattr(detector, "target_class_ids", None):
            print(
                f"[SeatManager] Auto-detect aborted: model has no '{label}' class.\n"
                f"              Use COCO-pretrained weights (e.g. yolov8n.pt),\n"
                f"              not your person-only fine-tune (best_v4.pt)."
            )
            return None
        return detector

    @staticmethod
    def _chair_boxes(detections, min_area: int) -> list[tuple[int, int, int, int]]:
        """Pull sensibly sized integer boxes out of a list of Detections."""
        boxes = []
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            if (x2 - x1) * (y2 - y1) >= min_area:
                boxes.append((x1, y1, x2, y2))
        return boxes

    @staticmethod
    def _box_iou(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _mean_box(
        boxes: list[tuple[int, int, int, int]],
    ) -> tuple[int, int, int, int]:
        """Average a group of boxes into one representative box."""
        n = len(boxes)
        return (
            sum(b[0] for b in boxes) // n,
            sum(b[1] for b in boxes) // n,
            sum(b[2] for b in boxes) // n,
            sum(b[3] for b in boxes) // n,
        )

    @classmethod
    def _dedup_boxes(
        cls,
        boxes: list[tuple[int, int, int, int]],
        iou_thresh: float = 0.70,
    ) -> list[tuple[int, int, int, int]]:
        """Drop boxes that nearly duplicate one already kept (high IoU)."""
        kept: list[tuple[int, int, int, int]] = []
        # Bigger boxes first so we keep the more complete chair box.
        for box in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
            if all(cls._box_iou(box, k) < iou_thresh for k in kept):
                kept.append(box)
        return kept

    @staticmethod
    def _reading_order(
        boxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        """
        Sort boxes top-to-bottom, then left-to-right, so seat_01, seat_02 …
        follow a sensible order. Rows are grouped by vertical bands roughly
        one chair-height apart.
        """
        if not boxes:
            return []
        heights = [y2 - y1 for (_, y1, _, y2) in boxes]
        band = max(20, int(np.median(heights) * 0.6))

        def key(b: tuple[int, int, int, int]):
            x1, y1, x2, y2 = b
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            return (cy // band, cx)

        return sorted(boxes, key=key)

    # ── Mutation ─────────────────────────────────────────────
    def add_seat(
        self,
        bbox: tuple,
        seat_id: str | None = None,
        label: str = "",
    ) -> SeatROI:
        self._counter += 1
        if seat_id is None:
            seat_id = f"seat_{self._counter:02d}"
        seat = SeatROI(seat_id=seat_id, bbox=tuple(int(v) for v in bbox), label=label)
        self.seats.append(seat)
        return seat

    def clear(self) -> None:
        self.seats.clear()
        self._counter = 0

    # ── Queries ───────────────────────────────────────────────
    def get_seat(self, seat_id: str) -> SeatROI | None:
        return next((s for s in self.seats if s.seat_id == seat_id), None)

    def match_person_to_seat(
        self, person_bbox: tuple[int, int, int, int]
    ) -> list[SeatROI]:
        """Return all seats that this person bbox overlaps sufficiently."""
        matched = []
        for seat in self.seats:
            if seat.overlap_fraction(person_bbox) >= self.overlap_threshold:
                matched.append(seat)
        return matched

    @property
    def occupancy_summary(self) -> dict[str, str]:
        return {s.seat_id: s.current_state for s in self.seats}

    @property
    def occupied_count(self) -> int:
        return sum(1 for s in self.seats if s.current_state == "occupied")

    @property
    def vacant_count(self) -> int:
        return sum(1 for s in self.seats if s.current_state == "vacant")

    # ── Interactive Drawing ───────────────────────────────────
    def draw_seats_interactive(self, frame: np.ndarray) -> None:
        """
        Opens a window and lets the user draw bounding boxes for each seat.
        Any seats already present (e.g. auto-detected drafts) are shown so
        you can keep them, undo from the end with 'u', or add more by drawing.
        Press 'q' when done, 'u' to undo last seat.
        """
        clone = frame.copy()
        drawing = False
        ix, iy = -1, -1
        temp_frame = clone.copy()

        def _draw_existing(img: np.ndarray) -> np.ndarray:
            out = img.copy()
            for seat in self.seats:
                x1, y1, x2, y2 = seat.bbox
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 100), 2)
                cv2.putText(
                    out, seat.seat_id, (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 2,
                )
            return out

        def on_mouse(event, x, y, flags, param):
            nonlocal drawing, ix, iy, temp_frame
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp_frame = _draw_existing(clone.copy())
                cv2.rectangle(temp_frame, (ix, iy), (x, y), (255, 100, 0), 2)
                cv2.imshow("Draw Seat ROIs", temp_frame)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                x1, y1 = min(ix, x), min(iy, y)
                x2, y2 = max(ix, x), max(iy, y)
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    self.add_seat(bbox=(x1, y1, x2, y2))
                temp_frame = _draw_existing(clone.copy())
                cv2.imshow("Draw Seat ROIs", temp_frame)

        cv2.namedWindow("Draw Seat ROIs")
        cv2.setMouseCallback("Draw Seat ROIs", on_mouse)

        print("\n[SeatManager] Draw seat ROIs with left-click drag.")
        print("  'u'  = undo last seat")
        print("  'q'  = finish\n")

        while True:
            display = _draw_existing(clone.copy())
            hint = f"Seats defined: {len(self.seats)}  |  'u'=undo  'q'=done"
            cv2.putText(display, hint, (10, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Draw Seat ROIs", display)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("u") and self.seats:
                self.seats.pop()
                self._counter -= 1

        cv2.destroyWindow("Draw Seat ROIs")
        print(f"[SeatManager] {len(self.seats)} seats defined.")

    # ── Visualisation ─────────────────────────────────────────
    def draw_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Overlay seat ROIs with current occupancy state."""
        for seat in self.seats:
            x1, y1, x2, y2 = seat.bbox
            color = (0, 60, 220) if seat.current_state == "occupied" else (0, 200, 80)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_parts = [seat.seat_id]
            if seat.current_state == "occupied" and seat.occupant_id:
                label_parts.append(seat.occupant_id)
            label = "  ".join(label_parts)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame