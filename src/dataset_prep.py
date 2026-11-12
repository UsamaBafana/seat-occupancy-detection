"""
src/dataset_prep.py
Dataset preparation utilities.

Helps you:
  1. Split a folder of images into train/val/test sets
  2. Verify label files match images
  3. Preview annotations on images
  4. Generate a data.yaml from scratch
  5. Download sample public datasets (COCO person subset)

Usage:
  python -m src.dataset_prep split  --src data/raw/images  --ratio 0.7 0.2 0.1
  python -m src.dataset_prep verify --data data/
  python -m src.dataset_prep preview --img data/images/train/img001.jpg --lbl data/labels/train/img001.txt
  python -m src.dataset_prep yaml   --classes person --out data/data.yaml
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
from loguru import logger

# ── Split ─────────────────────────────────────────────────────

def split_dataset(
    src_images: str | Path,
    src_labels: str | Path | None = None,
    output_dir: str | Path = "data/",
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
) -> None:
    """
    Split images (and matching labels) into train/val/test.

    Expects labels to have the same stem as images with .txt extension.
    If src_labels is None, assumes labels are in same dir with .txt suffix.
    """
    src_images = Path(src_images)
    if src_labels is None:
        src_labels = src_images

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in src_images.iterdir() if p.suffix.lower() in exts)

    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])

    splits = {
        "train": images[:n_train],
        "val":   images[n_train: n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    output_dir = Path(output_dir)
    for split_name, split_imgs in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            shutil.copy2(img_path, img_out / img_path.name)
            lbl_path = Path(src_labels) / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_out / lbl_path.name)
            else:
                # Create empty label file (background image)
                (lbl_out / (img_path.stem + ".txt")).touch()

        logger.info(f"[DataPrep] {split_name:5s}: {len(split_imgs)} images → {img_out}")

    logger.info(f"[DataPrep] Split complete. Total: {n} images.")


# ── Verify ────────────────────────────────────────────────────

def verify_dataset(data_dir: str | Path = "data/") -> dict[str, int]:
    """
    Check that every image has a corresponding label file.
    Reports mismatches.
    """
    data_dir = Path(data_dir)
    stats: dict[str, int] = {}
    issues = []

    for split in ("train", "val", "test"):
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split
        if not img_dir.exists():
            continue

        images = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}}
        labels = {p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt"} if lbl_dir.exists() else set()

        missing_labels = images - labels
        orphan_labels  = labels - images

        stats[split] = len(images)
        for stem in sorted(missing_labels):
            issues.append(f"  {split}: missing label for {stem}.jpg")
        for stem in sorted(orphan_labels):
            issues.append(f"  {split}: orphan label {stem}.txt (no image)")

    if issues:
        print(f"\n[DataPrep] {len(issues)} issue(s) found:")
        for issue in issues[:20]:
            print(issue)
        if len(issues) > 20:
            print(f"  ... and {len(issues)-20} more")
    else:
        print("[DataPrep] Dataset verified — no issues found ✓")

    print("\n  Counts:")
    for split, count in stats.items():
        print(f"    {split:5s}: {count}")

    return stats


# ── Preview annotations ───────────────────────────────────────

def preview_annotation(
    img_path: str | Path,
    lbl_path: str | Path,
    class_names: list[str] | None = None,
) -> None:
    """Display an image with its YOLO annotations overlaid."""
    img = cv2.imread(str(img_path))
    if img is None:
        logger.error(f"Cannot read image: {img_path}")
        return

    h, w = img.shape[:2]

    lbl_path = Path(lbl_path)
    if not lbl_path.exists():
        logger.warning(f"Label file not found: {lbl_path}")
    else:
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 100), 2)
                label = class_names[cid] if class_names and cid < len(class_names) else str(cid)
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 1)

    cv2.imshow("Annotation Preview", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Generate data.yaml ────────────────────────────────────────

def generate_yaml(
    classes: list[str],
    data_dir: str = "data/",
    output_path: str | Path = "data/data.yaml",
) -> None:
    import yaml
    cfg = {
        "path":  data_dir,
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(classes),
        "names": {i: name for i, name in enumerate(classes)},
    }
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    logger.info(f"[DataPrep] data.yaml written → {output_path}")


# ── CLI ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset preparation utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_split = sub.add_parser("split")
    p_split.add_argument("--src",    required=True)
    p_split.add_argument("--labels", default=None)
    p_split.add_argument("--out",    default="data/")
    p_split.add_argument("--ratio",  nargs=3, type=float, default=[0.7, 0.2, 0.1])

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--data", default="data/")

    p_prev = sub.add_parser("preview")
    p_prev.add_argument("--img", required=True)
    p_prev.add_argument("--lbl", required=True)
    p_prev.add_argument("--classes", nargs="+", default=["person"])

    p_yaml = sub.add_parser("yaml")
    p_yaml.add_argument("--classes", nargs="+", default=["person"])
    p_yaml.add_argument("--dir",     default="data/")
    p_yaml.add_argument("--out",     default="data/data.yaml")

    args = parser.parse_args()

    if args.cmd == "split":
        split_dataset(args.src, args.labels, args.out, tuple(args.ratio))
    elif args.cmd == "verify":
        verify_dataset(args.data)
    elif args.cmd == "preview":
        preview_annotation(args.img, args.lbl, args.classes)
    elif args.cmd == "yaml":
        generate_yaml(args.classes, args.dir, args.out)


if __name__ == "__main__":
    main()
