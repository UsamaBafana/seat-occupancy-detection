"""
src/occupancy_logger.py
Logs seat occupancy state to CSV and in-memory history.

Output schema:
  timestamp, frame_id, seat_id, state, confidence, occupant_id
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class OccupancyLogger:
    """
    Writes occupancy events to a rolling CSV file and maintains
    an in-memory DataFrame for live analysis.

    Parameters
    ----------
    output_dir       : Directory where the CSV will be written
    log_interval_sec : Min seconds between consecutive log entries for same seat
    """

    COLUMNS = ["timestamp", "frame_id", "seat_id", "state", "confidence", "occupant_id"]

    def __init__(self, output_dir: str | Path = "output/", log_interval_sec: float = 5.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval_sec = log_interval_sec

        self._csv_path = self.output_dir / f"occupancy_{int(time.time())}.csv"
        self._records: list[dict] = []
        self._last_logged: dict[str, float] = {}   # {seat_id: last_log_time}
        self._last_state: dict[str, str] = {}       # {seat_id: last_logged_state}

        # Write CSV header
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()

        logger.info(f"[Logger] Occupancy log → {self._csv_path}")

    # ── Public API ────────────────────────────────────────────

    def log_frame(
        self,
        frame_id: int,
        seat_states: dict,          # {seat_id: SeatState}
        confidences: dict[str, float] | None = None,
        occupants: dict[str, str] | None = None,
        timestamp: float | None = None,
    ) -> None:
        """
        Log occupancy states for all seats in this frame.
        Throttled by log_interval_sec unless state changed.

        timestamp : optional time to record instead of the wall clock.
                    When processing a recorded clip, pass the position in
                    the video (seconds from the start of the clip) so the
                    log lines up with ground truth labelled on that clip.
        """
        now = time.time() if timestamp is None else float(timestamp)
        if confidences is None:
            confidences = {}
        if occupants is None:
            occupants = {}

        rows_to_write = []

        for seat_id, state in seat_states.items():
            state_str = state.value if hasattr(state, "value") else str(state)

            last_time = self._last_logged.get(seat_id, 0.0)
            last_state = self._last_state.get(seat_id, "")

            state_changed = state_str != last_state
            interval_elapsed = (now - last_time) >= self.log_interval_sec

            if state_changed or interval_elapsed:
                row = {
                    "timestamp": now,
                    "frame_id": frame_id,
                    "seat_id": seat_id,
                    "state": state_str,
                    "confidence": round(confidences.get(seat_id, 0.0), 3),
                    "occupant_id": occupants.get(seat_id, ""),
                }
                self._records.append(row)
                rows_to_write.append(row)
                self._last_logged[seat_id] = now
                self._last_state[seat_id] = state_str

        if rows_to_write:
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writerows(rows_to_write)

    def log_event(
        self,
        frame_id: int,
        seat_id: str,
        state: str,
        confidence: float = 0.0,
        occupant_id: str = "",
    ) -> None:
        """Log a single seat event unconditionally."""
        row = {
            "timestamp": time.time(),
            "frame_id": frame_id,
            "seat_id": seat_id,
            "state": state,
            "confidence": round(confidence, 3),
            "occupant_id": occupant_id,
        }
        self._records.append(row)
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(row)

    # ── Analysis helpers ──────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(columns=self.COLUMNS)
        df = pd.DataFrame(self._records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    def occupancy_duration(self) -> pd.DataFrame:
        """
        Compute total occupied / vacant duration per seat (seconds).
        """
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values(["seat_id", "timestamp"])
        results = []

        for seat_id, group in df.groupby("seat_id"):
            group = group.reset_index(drop=True)
            for i in range(len(group) - 1):
                duration = (
                    group.loc[i + 1, "timestamp"] - group.loc[i, "timestamp"]
                ).total_seconds()
                results.append({
                    "seat_id": seat_id,
                    "state": group.loc[i, "state"],
                    "duration_sec": duration,
                })

        return pd.DataFrame(results)

    def save_summary(self, path: Optional[str | Path] = None) -> Path:
        """Write a human-readable summary CSV."""
        path = Path(path) if path else self.output_dir / "summary.csv"
        dur_df = self.occupancy_duration()
        if dur_df.empty:
            logger.warning("[Logger] No data to summarise yet.")
            return path

        summary = (
            dur_df.groupby(["seat_id", "state"])["duration_sec"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        summary.to_csv(path, index=False)
        logger.info(f"[Logger] Summary saved → {path}")
        return path

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    @property
    def record_count(self) -> int:
        return len(self._records)
