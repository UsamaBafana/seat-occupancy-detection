"""
src/temporal_inference.py
Temporal state machine for each seat.

Reduces false positives by requiring consistent evidence across
a sliding window of frames before flipping seat state.

State machine per seat:
  vacant  ──[window fill above threshold]──► occupied
  occupied ──[window drop below threshold]──► vacant

A "cooldown" period prevents rapid toggling.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger

# ─────────────────────────────────────────────────────────────
#  Enums & constants
# ─────────────────────────────────────────────────────────────

class SeatState(str, Enum):
    VACANT   = "vacant"
    OCCUPIED = "occupied"
    UNKNOWN  = "unknown"


# ─────────────────────────────────────────────────────────────
#  Per-seat temporal buffer
# ─────────────────────────────────────────────────────────────

@dataclass
class SeatTemporalBuffer:
    seat_id: str
    window_size: int        = 15
    occupied_threshold: float = 0.60    # fraction of window that must be occupied
    cooldown_frames: int    = 20        # min frames between state transitions

    # Runtime state
    _buffer: deque = field(default_factory=deque, repr=False)
    _state: SeatState = field(default=SeatState.UNKNOWN, repr=False)
    _frames_since_transition: int = field(default=0, repr=False)
    _last_transition_time: float = field(default_factory=time.time, repr=False)

    # Occupant tracking
    _last_occupant_id: Optional[str] = field(default=None, repr=False)
    _state_history: list[tuple[float, SeatState]] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._buffer = deque(maxlen=self.window_size)

    # ── Update ────────────────────────────────────────────────

    def update(self, person_present: bool, occupant_id: Optional[str] = None) -> SeatState:
        """
        Feed a single frame observation.

        Parameters
        ----------
        person_present : True if a person was detected overlapping this seat
        occupant_id    : Face identity if available, else None

        Returns
        -------
        The current inferred SeatState.
        """
        self._buffer.append(1 if person_present else 0)
        self._frames_since_transition += 1

        if occupant_id:
            self._last_occupant_id = occupant_id

        new_state = self._compute_state()
        if new_state != self._state and new_state != SeatState.UNKNOWN:
            if self._frames_since_transition >= self.cooldown_frames:
                old = self._state
                self._state = new_state
                self._frames_since_transition = 0
                self._last_transition_time = time.time()
                self._state_history.append((time.time(), new_state))
                logger.debug(
                    f"[Temporal] {self.seat_id}: {old.value} → {new_state.value}  "
                    f"(buffer={self.occupancy_ratio:.2f})"
                )
        elif self._state == SeatState.UNKNOWN and len(self._buffer) >= self.window_size // 2:
            # Bootstrap from partial window
            self._state = new_state if new_state != SeatState.UNKNOWN else SeatState.VACANT

        return self._state

    # ── Properties ───────────────────────────────────────────

    @property
    def state(self) -> SeatState:
        return self._state

    @property
    def occupancy_ratio(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(self._buffer) / len(self._buffer)

    @property
    def last_occupant(self) -> Optional[str]:
        return self._last_occupant_id

    @property
    def is_occupied(self) -> bool:
        return self._state == SeatState.OCCUPIED

    @property
    def confidence(self) -> float:
        """Confidence that the current state label is correct (0–1)."""
        r = self.occupancy_ratio
        if self._state == SeatState.OCCUPIED:
            return r
        elif self._state == SeatState.VACANT:
            return 1.0 - r
        return 0.5

    # ── Internal ─────────────────────────────────────────────

    def _compute_state(self) -> SeatState:
        if len(self._buffer) < max(3, self.window_size // 3):
            return SeatState.UNKNOWN
        r = self.occupancy_ratio
        if r >= self.occupied_threshold:
            return SeatState.OCCUPIED
        elif r <= (1.0 - self.occupied_threshold):
            return SeatState.VACANT
        return SeatState.UNKNOWN   # ambiguous — hold current state

    def reset(self) -> None:
        self._buffer.clear()
        self._state = SeatState.UNKNOWN
        self._frames_since_transition = 0
        self._last_occupant_id = None

    def summary(self) -> dict:
        return {
            "seat_id":         self.seat_id,
            "state":           self._state.value,
            "confidence":      round(self.confidence, 3),
            "occupancy_ratio": round(self.occupancy_ratio, 3),
            "last_occupant":   self._last_occupant_id or "",
            "window_size":     len(self._buffer),
        }


# ─────────────────────────────────────────────────────────────
#  Inference engine (manages all seats)
# ─────────────────────────────────────────────────────────────

class TemporalInferenceEngine:
    """
    Manages one SeatTemporalBuffer per seat and provides a batch update API.
    """

    def __init__(
        self,
        window_size: int = 15,
        occupied_threshold: float = 0.60,
        cooldown_frames: int = 20,
    ):
        self.window_size = window_size
        self.occupied_threshold = occupied_threshold
        self.cooldown_frames = cooldown_frames

        self._buffers: dict[str, SeatTemporalBuffer] = {}

    # ── Seat management ───────────────────────────────────────

    def register_seat(self, seat_id: str) -> None:
        if seat_id not in self._buffers:
            self._buffers[seat_id] = SeatTemporalBuffer(
                seat_id=seat_id,
                window_size=self.window_size,
                occupied_threshold=self.occupied_threshold,
                cooldown_frames=self.cooldown_frames,
            )

    def register_seats(self, seat_ids: list[str]) -> None:
        for sid in seat_ids:
            self.register_seat(sid)

    def remove_seat(self, seat_id: str) -> None:
        self._buffers.pop(seat_id, None)

    # ── Update ────────────────────────────────────────────────

    def update(
        self,
        occupied_seat_ids: set[str],
        occupant_map: dict[str, str] | None = None,
    ) -> dict[str, SeatState]:
        """
        Update all registered seats with the current frame's observations.

        Parameters
        ----------
        occupied_seat_ids : set of seat IDs that have a person overlapping them
        occupant_map      : optional {seat_id: face_identity} for recognised persons

        Returns
        -------
        dict {seat_id: SeatState}
        """
        if occupant_map is None:
            occupant_map = {}

        results: dict[str, SeatState] = {}
        for seat_id, buf in self._buffers.items():
            person_present = seat_id in occupied_seat_ids
            occupant_id = occupant_map.get(seat_id)
            results[seat_id] = buf.update(person_present, occupant_id)

        return results

    # ── Queries ───────────────────────────────────────────────

    def get_state(self, seat_id: str) -> SeatState:
        buf = self._buffers.get(seat_id)
        return buf.state if buf else SeatState.UNKNOWN

    def get_all_states(self) -> dict[str, SeatState]:
        return {sid: buf.state for sid, buf in self._buffers.items()}

    def get_summary(self) -> list[dict]:
        return [buf.summary() for buf in self._buffers.values()]

    @property
    def occupied_seats(self) -> list[str]:
        return [sid for sid, buf in self._buffers.items() if buf.is_occupied]

    @property
    def vacant_seats(self) -> list[str]:
        return [sid for sid, buf in self._buffers.items() if not buf.is_occupied]

    @property
    def occupied_count(self) -> int:
        return len(self.occupied_seats)

    @property
    def total_seats(self) -> int:
        return len(self._buffers)

    def reset_all(self) -> None:
        for buf in self._buffers.values():
            buf.reset()
