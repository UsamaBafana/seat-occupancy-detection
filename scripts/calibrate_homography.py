"""
calibrate_homography.py
Estimate the homography between two overlapping cameras from clicked points.

A frame from each camera is shown side by side; click at least four matching
ground-plane points (floor corners, chair wheels, desk corners) in the same
order in both views. The homography that maps camera 1 onto camera 0 is then
estimated with cv2.findHomography (RANSAC, LMEDS or plain least squares) and
saved, together with the clicked points and the reprojection error, for
auto_match_seats_homography.py and sweep_homography_params.py.

Usage (from the project root):
  python scripts/calibrate_homography.py --cam0-source 0 --cam1-source 1
  python scripts/calibrate_homography.py --cam0-source 0 --cam1-source 1 --method lmeds

Output: config/homography_cam1_to_cam0.json (set with --out).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def grab_frame(source: str, warmup: int = 30):
    if not source.isdigit():
        img = cv2.imread(source)
        if img is not None:
            return img
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source,
                            cv2.CAP_DSHOW if source.isdigit() else 0)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    frame = None
    for _ in range(max(1, warmup)):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise RuntimeError(f"Could not read a frame from: {source}")
    return frame


METHODS = {
    "ransac": cv2.RANSAC,
    "lmeds": cv2.LMEDS,
    "leastsq": 0,   # plain least-squares fit, no outlier rejection
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Click-calibrate a homography between two cameras")
    ap.add_argument("--cam0-source", default="0")
    ap.add_argument("--cam1-source", default="1")
    ap.add_argument("--out", default="config/homography_cam1_to_cam0.json")
    ap.add_argument("--min-points", type=int, default=4)
    ap.add_argument("--method", choices=list(METHODS), default="ransac",
                     help="fit method: ransac (robust to a few bad clicks, default), "
                          "lmeds (robust, no threshold needed), leastsq (uses every point, no outlier rejection)")
    ap.add_argument("--reproj-threshold", type=float, default=5.0,
                     help="(ransac only) max reprojection error in px for a point pair to count as an inlier — "
                          "was hardcoded to 5.0 before; raise it if too many real points get rejected, "
                          "lower it to be stricter about mis-clicks")
    ap.add_argument("--confidence", type=float, default=0.995,
                     help="(ransac/lmeds) desired probability the estimated homography is correct — "
                          "higher runs more RANSAC iterations")
    ap.add_argument("--max-iters", type=int, default=2000,
                     help="(ransac only) max RANSAC iterations")
    args = ap.parse_args()

    frame0 = grab_frame(args.cam0_source)
    frame1 = grab_frame(args.cam1_source)

    h = min(frame0.shape[0], frame1.shape[0])

    def resize(v):
        scale = h / v.shape[0]
        return cv2.resize(v, (int(v.shape[1] * scale), h))

    view0, view1 = resize(frame0), resize(frame1)
    offset_x = view0.shape[1]
    combined = np.hstack([view0, view1])

    scale0 = frame0.shape[1] / view0.shape[1]
    scale1 = frame1.shape[1] / view1.shape[1]
    vscale0 = frame0.shape[0] / view0.shape[0]
    vscale1 = frame1.shape[0] / view1.shape[0]

    pts0: list[tuple[float, float]] = []
    pts1: list[tuple[float, float]] = []
    pending_cam0: tuple[float, float] | None = None

    win_name = "Click matching points: cam0 then same point in cam1 - 'u'=undo 'q'=finish"
    colors = [(0, 220, 120), (0, 160, 255), (255, 120, 0), (200, 0, 200),
              (0, 220, 220), (255, 255, 0), (150, 80, 255), (80, 200, 80)]

    def redraw():
        display = combined.copy()
        for i, (p0, p1) in enumerate(zip(pts0, pts1)):
            color = colors[i % len(colors)]
            d0 = (int(p0[0] / scale0), int(p0[1] / vscale0))
            d1 = (int(p1[0] / scale1) + offset_x, int(p1[1] / vscale1))
            cv2.circle(display, d0, 6, color, -1)
            cv2.circle(display, d1, 6, color, -1)
            cv2.putText(display, str(i + 1), (d0[0] + 8, d0[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(display, str(i + 1), (d1[0] + 8, d1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if pending_cam0 is not None:
            d0 = (int(pending_cam0[0] / scale0), int(pending_cam0[1] / vscale0))
            cv2.circle(display, d0, 8, (0, 0, 255), 2)
        cv2.imshow(win_name, display)

    def on_mouse(event, x, y, flags, param):
        nonlocal pending_cam0
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x < offset_x:
            pending_cam0 = (x * scale0, y * vscale0)
            print("  cam0 point selected — now click the SAME physical point in cam1's view (right)")
        else:
            if pending_cam0 is None:
                print("  Click a point in cam0 (left) first.")
                return
            local_x = x - offset_x
            orig = (local_x * scale1, y * vscale1)
            pts0.append(pending_cam0)
            pts1.append(orig)
            print(f"  Pair {len(pts0)} recorded.")
            pending_cam0 = None
        redraw()

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)
    print(f"\n[Calibrate] Click matching points: floor corners, desk corners, wall edges — "
          f"anything visible and identifiable in BOTH views. Need at least {args.min_points}.")
    print("[Calibrate] Spread points out — avoid clicking them all in a straight line, "
          "that makes the math unstable.")
    print("[Calibrate] Click cam0 (left) point, then the SAME physical point in cam1 (right). "
          "'u'=undo last pair, 'q'=finish.\n")
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u") and pts0:
            pts0.pop()
            pts1.pop()
            redraw()

    cv2.destroyAllWindows()

    if len(pts0) < args.min_points:
        print(f"[Calibrate] Only {len(pts0)} pair(s) — need at least {args.min_points}. Nothing saved.")
        return

    src = np.array(pts1, dtype=np.float32)
    dst = np.array(pts0, dtype=np.float32)

    method_flag = METHODS[args.method]
    kwargs = {}
    if args.method == "ransac":
        kwargs = dict(ransacReprojThreshold=args.reproj_threshold,
                      confidence=args.confidence, maxIters=args.max_iters)
    elif args.method == "lmeds":
        kwargs = dict(confidence=args.confidence)

    H, mask = cv2.findHomography(src, dst, method=method_flag, **kwargs)

    if H is None:
        print("[Calibrate] Could not compute a homography from these points — try again with "
              "clearer, more spread-out reference points (avoid points all in a line).")
        return

    inliers = int(mask.sum()) if mask is not None else len(pts0)
    print(f"[Calibrate] Homography computed ({args.method}). "
          f"{inliers}/{len(pts0)} point pairs used as inliers.")

    # Per-point reprojection error, so the report can cite a concrete number
    # instead of just "it worked" — how far each cam1 point lands from its
    # matching cam0 point once mapped through H.
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    print("[Calibrate] Reprojection error per pair (px): "
          + ", ".join(f"{e:.1f}" for e in errors))
    print(f"[Calibrate]   mean={errors.mean():.1f}px  max={errors.max():.1f}px")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "homography_cam1_to_cam0": H.tolist(),
            "point_pairs": {"cam0": pts0, "cam1": pts1},
            "inliers": inliers,
            "total_points": len(pts0),
            "method": args.method,
            "reproj_threshold": args.reproj_threshold if args.method == "ransac" else None,
            "confidence": args.confidence if args.method in ("ransac", "lmeds") else None,
            "reprojection_error_px": {"per_pair": errors.tolist(),
                                      "mean": float(errors.mean()),
                                      "max": float(errors.max())},
        }, f, indent=2)

    print(f"[Calibrate] Saved -> {out_path}")


if __name__ == "__main__":
    main()
