"""
sweep_homography_params.py
Parameter sweep for cv2.findHomography, using point pairs you've ALREADY
clicked (from a saved calibrate_homography.py output) — no camera, no
re-clicking. This is the "reportable experiment" for the FYP writeup:
instead of asserting one fixed method/threshold worked, it shows how
inlier count and reprojection error change across a grid of parameters,
so you can justify the choice with numbers instead of trial and error.

What it measures, per parameter combination:
  - inliers          : how many of your clicked pairs RANSAC/LMEDS kept
  - mean/max error    : reprojection error (px) of EVERY pair through the
                        resulting H — including pairs marked outliers, so
                        you can see whether "outliers" were genuine bad
                        clicks (huge error) or just borderline (small error)

Usage:
  python scripts/sweep_homography_params.py
  python scripts/sweep_homography_params.py --homography config/homography_cam1_to_cam0.json
  python scripts/sweep_homography_params.py --thresholds 2 3 5 8 10 15 --out output/homography_sweep.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

METHODS = {
    "ransac": cv2.RANSAC,
    "lmeds": cv2.LMEDS,
    "leastsq": 0,
}


def load_point_pairs(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    pairs = data.get("point_pairs")
    if not pairs or not pairs.get("cam0") or not pairs.get("cam1"):
        raise SystemExit(
            f"[!] {path} has no point_pairs saved — this sweep needs a homography JSON "
            f"produced by calibrate_homography.py (it saves the clicked points alongside H)."
        )
    dst = np.array(pairs["cam0"], dtype=np.float32)   # cam0 = destination space
    src = np.array(pairs["cam1"], dtype=np.float32)   # cam1 = source space
    if len(src) < 4:
        raise SystemExit(f"[!] Only {len(src)} point pair(s) in {path} — need at least 4.")
    return src, dst


def fit_and_score(src: np.ndarray, dst: np.ndarray, method: str,
                   reproj_threshold: float, confidence: float) -> dict | None:
    kwargs = {}
    if method == "ransac":
        kwargs = dict(ransacReprojThreshold=reproj_threshold, confidence=confidence, maxIters=2000)
    elif method == "lmeds":
        kwargs = dict(confidence=confidence)

    H, mask = cv2.findHomography(src, dst, method=METHODS[method], **kwargs)
    if H is None:
        return None

    if method == "leastsq":
        # No outlier concept for plain least-squares — OpenCV returns a
        # meaningless all-zero mask here, not "0 inliers". Every point is used.
        inliers = len(src)
    else:
        inliers = int(mask.sum()) if mask is not None else len(src)
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)

    return {
        "method": method,
        "reproj_threshold": reproj_threshold if method == "ransac" else None,
        "confidence": confidence if method in ("ransac", "lmeds") else None,
        "inliers": inliers,
        "total_points": len(src),
        "mean_error_px": round(float(errors.mean()), 2),
        "max_error_px": round(float(errors.max()), 2),
        "per_pair_error_px": [round(float(e), 1) for e in errors],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep cv2.findHomography parameters over already-clicked points")
    ap.add_argument("--homography", default="config/homography_cam1_to_cam0.json",
                     help="a homography JSON saved by calibrate_homography.py (needs point_pairs)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[2, 3, 5, 8, 10, 15],
                     help="ransacReprojThreshold values to try (px)")
    ap.add_argument("--confidences", type=float, nargs="+", default=[0.99],
                     help="confidence values to try (usually one value is enough)")
    ap.add_argument("--out", default=None,
                     help="optional path to save the full sweep results as JSON")
    args = ap.parse_args()

    src, dst = load_point_pairs(args.homography)
    print(f"[Sweep] Loaded {len(src)} point pair(s) from {args.homography}\n")

    results: list[dict] = []

    # RANSAC across every threshold x confidence combination.
    for conf in args.confidences:
        for thr in args.thresholds:
            r = fit_and_score(src, dst, "ransac", thr, conf)
            if r:
                results.append(r)

    # LMEDS and plain least-squares don't take a threshold — one row each.
    for method in ("lmeds", "leastsq"):
        r = fit_and_score(src, dst, method, 0.0, args.confidences[0])
        if r:
            results.append(r)

    header = f"{'method':<9}{'threshold':<11}{'confidence':<12}{'inliers':<10}{'mean err (px)':<15}{'max err (px)':<12}"
    print(header)
    print("-" * len(header))
    for r in results:
        thr = f"{r['reproj_threshold']:.0f}" if r["reproj_threshold"] is not None else "-"
        conf = f"{r['confidence']:.3f}" if r["confidence"] is not None else "-"
        print(f"{r['method']:<9}{thr:<11}{conf:<12}"
              f"{r['inliers']}/{r['total_points']:<8}"
              f"{r['mean_error_px']:<15}{r['max_error_px']:<12}")

    best = min(results, key=lambda r: r["mean_error_px"])
    print(f"\n[Sweep] Lowest mean reprojection error: {best['method']}"
          + (f" (threshold={best['reproj_threshold']:.0f}px)" if best["reproj_threshold"] else "")
          + f" -> mean {best['mean_error_px']}px, max {best['max_error_px']}px, "
          + f"{best['inliers']}/{best['total_points']} inliers")
    print("[Sweep] Note: fewer inliers with a lower mean error isn't automatically 'better' — "
          "it may just be discarding real points as outliers. Look at per_pair_error_px "
          "(saved with --out) to check whether dropped points were genuine mis-clicks "
          "(large error) or borderline ones (small error) before picking a threshold.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Sweep] Full results saved -> {out_path}")


if __name__ == "__main__":
    main()
