"""Tests for the evaluation metrics (report Sections 5.3 and 5.8)."""
from pathlib import Path

import pytest

from evaluate.metrics import OccupancyEvaluator, compute_classification_metrics

POC = Path(__file__).resolve().parent.parent / "results" / "poc_evaluation"


def test_classification_metrics_on_a_small_example():
    y_true = ["occupied", "occupied", "vacant", "vacant", "occupied"]
    y_pred = ["occupied", "vacant", "vacant", "occupied", "occupied"]
    m = compute_classification_metrics(y_true, y_pred)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["specificity"] == pytest.approx(0.5)


def test_poc_results_are_reproduced_from_saved_logs():
    """Re-computes Table 12 (all seat-seconds) of the final report from results/."""
    ev = OccupancyEvaluator(
        POC / "ground_truth_per_second.csv",
        POC / "system_log_cliptime.csv",
        tolerance_sec=10,
        direction="backward",
    )
    m = ev.overall_metrics()
    assert m["n_samples"] == 354
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (79, 23, 16, 236)
    assert m["f1"] == pytest.approx(0.802, abs=1e-3)
