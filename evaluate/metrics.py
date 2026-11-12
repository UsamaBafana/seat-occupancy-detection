"""
evaluate/metrics.py
Seat-occupancy evaluation against manually labelled ground truth.

Reads the system's occupancy log and a ground-truth CSV (same schema), pairs each
ground-truth row with a system log row for the same seat, and reports accuracy, precision, recall, F1, specificity and
false-positive rate overall and per seat, plus a confusion matrix. A positive is
the "occupied" state.

Two pairing modes:
  nearest  : closest log row within --tolerance seconds (pilot, report Section 5.3)
  backward : most recent state the system had reported at that moment
             (PoC test, report Section 5.8; use a tolerance longer than
             io.log_interval_seconds, e.g. 10 s)

Used by:
    python main.py evaluate --pred <log.csv> --gt <ground_truth.csv>
    python main.py evaluate --pred <log.csv> --gt <ground_truth.csv> --direction backward --tolerance 10
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# ─────────────────────────────────────────────────────────────
#  Core metrics
# ─────────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    positive_label: str = "occupied",
) -> dict[str, float]:
    """
    Binary classification metrics for occupied vs vacant.

    Returns
    -------
    dict with: accuracy, precision, recall, f1, specificity, fpr
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    tn = np.sum((y_pred != positive_label) & (y_true != positive_label))
    fp = np.sum((y_pred == positive_label) & (y_true != positive_label))
    fn = np.sum((y_pred != positive_label) & (y_true == positive_label))

    n = len(y_true)
    accuracy    = (tp + tn) / n if n > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr         = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy":    round(accuracy,    4),
        "precision":   round(precision,   4),
        "recall":      round(recall,      4),
        "f1":          round(f1,          4),
        "specificity": round(specificity, 4),
        "fpr":         round(fpr,         4),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "n_samples": int(n),
    }


def confusion_matrix_values(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    labels: list[str] = None,
) -> np.ndarray:
    """Return a 2x2 (or NxN) confusion matrix as a numpy array."""
    from sklearn.metrics import confusion_matrix
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))
    return confusion_matrix(y_true, y_pred, labels=labels)


# ─────────────────────────────────────────────────────────────
#  Evaluator (reads prediction and GT CSVs)
# ─────────────────────────────────────────────────────────────

class OccupancyEvaluator:
    """
    Loads ground-truth and prediction CSV files and computes metrics.

    CSV format: timestamp, frame_id, seat_id, state, confidence, occupant_id
    (frame_id is carried through but not used for matching — see module
    docstring. It only needs to be present in predictions, and is written
    as a placeholder 0 by record_ground_truth.py for ground truth.)

    Parameters
    ----------
    tolerance_sec : max time gap (seconds) allowed when pairing a ground
                    truth row with its nearest prediction row for the
                    same seat. Rows with no prediction inside this
                    window are dropped rather than mismatched.
    """

    def __init__(
        self,
        ground_truth_csv: str | Path,
        prediction_csv: str | Path,
        positive_label: str = "occupied",
        tolerance_sec: float = 2.0,
        direction: str = "nearest",
    ):
        self.gt_path   = Path(ground_truth_csv)
        self.pred_path = Path(prediction_csv)
        self.positive_label = positive_label
        self.tolerance_sec = tolerance_sec
        # "nearest"  : closest prediction within tolerance (original behaviour)
        # "backward" : the most recent prediction at or before each label -
        #              i.e. "what did the system say the seat was at that
        #              moment". Use this with clip-time logs, since the
        #              pipeline only logs on change / every log_interval s.
        self.direction = direction

        self.gt_df   = self._load(self.gt_path,   "ground_truth")
        self.pred_df = self._load(self.pred_path,  "predictions")
        self.merged  = self._merge()

    # ── Load & merge ─────────────────────────────────────────

    @staticmethod
    def _load(path: Path, name: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"{name} CSV not found: {path}")
        df = pd.read_csv(path)
        required = {"seat_id", "state", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} CSV missing columns: {missing}")
        return df

    def _merge(self) -> pd.DataFrame:
        """
        Join ground truth and predictions by NEAREST timestamp within
        self.tolerance_sec, grouped by seat_id.
        """
        gt = self.gt_df[["timestamp", "seat_id", "state"]].rename(
            columns={"state": "gt_state"}
        ).copy()
        pred = self.pred_df[["timestamp", "seat_id", "state", "confidence"]].rename(
            columns={"state": "pred_state"}
        ).copy()

        # same dtype on both sides (whole-second timestamps would load as int)
        gt["timestamp"] = gt["timestamp"].astype(float)
        pred["timestamp"] = pred["timestamp"].astype(float)

        gt = gt.sort_values("timestamp").reset_index(drop=True)
        pred = pred.sort_values("timestamp").reset_index(drop=True)

        merged = pd.merge_asof(
            gt, pred,
            on="timestamp",
            by="seat_id",
            direction=self.direction,
            tolerance=self.tolerance_sec,
        )
        merged = merged.dropna(subset=["pred_state"]).reset_index(drop=True)

        if merged.empty:
            logger.warning(
                "[Evaluator] Merge produced no rows — check that ground truth "
                "and predictions were logged during an overlapping time window, "
                "and that tolerance_sec is wide enough."
            )
        return merged

    # ── Metrics ───────────────────────────────────────────────

    def overall_metrics(self) -> dict[str, float]:
        """Metrics across all seats and frames."""
        return compute_classification_metrics(
            self.merged["gt_state"].tolist(),
            self.merged["pred_state"].tolist(),
            positive_label=self.positive_label,
        )

    def per_seat_metrics(self) -> pd.DataFrame:
        """Metrics broken down by seat_id."""
        expected_cols = ["seat_id", "accuracy", "precision", "recall", "f1",
                         "specificity", "fpr", "tp", "tn", "fp", "fn", "n_samples"]
        if self.merged.empty:
            return pd.DataFrame(columns=expected_cols)

        rows = []
        for seat_id, group in self.merged.groupby("seat_id"):
            m = compute_classification_metrics(
                group["gt_state"].tolist(),
                group["pred_state"].tolist(),
                positive_label=self.positive_label,
            )
            m["seat_id"] = seat_id
            rows.append(m)
        df = pd.DataFrame(rows)
        cols = ["seat_id"] + [c for c in df.columns if c != "seat_id"]
        return df[cols]

    def confusion_matrix(self) -> np.ndarray:
        labels = [self.positive_label, "vacant"]
        return confusion_matrix_values(
            self.merged["gt_state"].tolist(),
            self.merged["pred_state"].tolist(),
            labels=labels,
        )

    def temporal_accuracy(self, window_sec: float = 60.0) -> pd.DataFrame:
        """
        Rolling accuracy over time windows.
        """
        if "timestamp" not in self.merged.columns:
            logger.warning("[Evaluator] No timestamp column for temporal analysis.")
            return pd.DataFrame()

        self.merged["correct"] = (
            self.merged["gt_state"] == self.merged["pred_state"]
        ).astype(int)

        self.merged["timestamp"] = pd.to_datetime(self.merged["timestamp"], unit="s")
        self.merged = self.merged.sort_values("timestamp")
        self.merged.set_index("timestamp", inplace=True)
        rolling = self.merged["correct"].rolling(f"{int(window_sec)}s").mean()
        self.merged.reset_index(inplace=True)
        return rolling

    # ── Report ────────────────────────────────────────────────

    def print_report(self) -> None:
        m = self.overall_metrics()

        if m["n_samples"] == 0:
            print("\n" + "═" * 48)
            print("  NO MATCHED SAMPLES — this is not a 0% score,")
            print("  it means ground truth and predictions never lined")
            print("  up within tolerance_sec. Check the timestamp ranges")
            print("  in both CSVs overlap, and widen tolerance_sec if")
            print("  they're close but not close enough.")
            print("═" * 48 + "\n")
            return

        print("\n" + "═" * 48)
        print("  OCCUPANCY DETECTION — EVALUATION REPORT")
        print("═" * 48)
        print(f"  Samples evaluated : {m['n_samples']}")
        print(f"  Accuracy          : {m['accuracy']:.1%}")
        print(f"  Precision         : {m['precision']:.1%}")
        print(f"  Recall            : {m['recall']:.1%}")
        print(f"  F1 Score          : {m['f1']:.1%}")
        print(f"  Specificity       : {m['specificity']:.1%}")
        print(f"  False Positive Rate: {m['fpr']:.1%}")
        print("\n  Confusion Matrix (rows=GT, cols=Pred)")
        print(f"  TP={m['tp']}  FP={m['fp']}")
        print(f"  FN={m['fn']}  TN={m['tn']}")
        print("═" * 48 + "\n")

    def save_report(self, output_dir: str | Path = "output/") -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Overall
        overall = self.overall_metrics()
        pd.DataFrame([overall]).to_csv(output_dir / "eval_overall.csv", index=False)

        # Per seat
        self.per_seat_metrics().to_csv(output_dir / "eval_per_seat.csv", index=False)

        # Confusion matrix
        cm = self.confusion_matrix()
        pd.DataFrame(
            cm,
            index=["GT_occupied", "GT_vacant"],
            columns=["Pred_occupied", "Pred_vacant"],
        ).to_csv(output_dir / "eval_confusion_matrix.csv")

        logger.info(f"[Evaluator] Reports saved → {output_dir}")

    def plot_confusion_matrix(
        self,
        save_path: Optional[str | Path] = None,
        show: bool = True,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not installed — skipping plot.")
            return

        cm = self.confusion_matrix()
        labels = [self.positive_label, "vacant"]

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"Pred {l}" for l in labels],
            yticklabels=[f"GT {l}" for l in labels],
            ax=ax,
        )
        ax.set_title("Occupancy Detection — Confusion Matrix")
        plt.tight_layout()

        if save_path:
            plt.savefig(str(save_path), dpi=150)
            logger.info(f"[Evaluator] Confusion matrix plot → {save_path}")
        if show:
            plt.show()
        plt.close(fig)

