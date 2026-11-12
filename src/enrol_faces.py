"""
src/enrol_faces.py
Utility to enrol known faces into the recognition gallery.

Two modes:
  1. From a directory of images      → --source path/to/images/ --name Alice
  2. Live capture from webcam        → --webcam --name Alice

Each enrolled person gets a sub-directory under known_faces/:
  known_faces/
    Alice/
      Alice_000.jpg
      Alice_001.jpg
    Bob/
      Bob_000.jpg

Usage examples:
  python -m src.enrol_faces --name Alice --source photos/alice/
  python -m src.enrol_faces --name Bob   --webcam --count 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from loguru import logger


def enrol_from_directory(
    name: str,
    source_dir: str | Path,
    gallery_dir: str | Path = "known_faces/",
) -> int:
    """Copy / convert images from a directory into the gallery. Returns count."""
    source_dir = Path(source_dir)
    gallery_dir = Path(gallery_dir)
    person_dir = gallery_dir / name
    person_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in source_dir.iterdir() if p.suffix.lower() in exts]

    if not images:
        logger.error(f"No images found in {source_dir}")
        return 0

    count = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        out_path = person_dir / f"{name}_{count:03d}.jpg"
        cv2.imwrite(str(out_path), img)
        count += 1
        logger.info(f"  Enrolled: {out_path}")

    logger.info(f"[Enrol] {count} images enrolled for {name!r}")
    return count


def enrol_from_webcam(
    name: str,
    gallery_dir: str | Path = "known_faces/",
    count: int = 5,
    camera_id: int = 0,
    interval_ms: int = 500,
) -> int:
    """
    Capture `count` face images from the webcam.
    Press SPACE to capture, 'q' to stop early.
    """
    gallery_dir = Path(gallery_dir)
    person_dir = gallery_dir / name
    person_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_id}")
        return 0

    captured = 0
    existing = len(list(person_dir.iterdir()))

    print(f"\n[Enrol] Press SPACE to capture, 'q' to finish.  Target: {count} photos.")
    print(f"        Saving to: {person_dir}\n")

    while captured < count:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(
            display,
            f"Enrolling: {name}  [{captured}/{count}]  SPACE=capture  q=quit",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 120), 2,
        )
        cv2.imshow("Enrol Faces", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            out_path = person_dir / f"{name}_{existing + captured:03d}.jpg"
            cv2.imwrite(str(out_path), frame)
            logger.info(f"  Captured: {out_path}")
            captured += 1
            cv2.waitKey(interval_ms)

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"[Enrol] {captured} images enrolled for {name!r}")
    return captured


def list_gallery(gallery_dir: str | Path = "known_faces/") -> None:
    """Print a summary of enrolled identities."""
    gallery_dir = Path(gallery_dir)
    if not gallery_dir.exists():
        print(f"Gallery directory not found: {gallery_dir}")
        return

    identities = sorted(
        d for d in gallery_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not identities:
        print("No identities enrolled yet.")
        return

    print(f"\n Gallery: {gallery_dir}")
    print("─" * 40)
    for person_dir in identities:
        imgs = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        print(f"  {person_dir.name:<20} {len(imgs):>3} image(s)")
    print("─" * 40 + "\n")


# ── CLI ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrol faces into the recognition gallery")
    parser.add_argument("--name",    type=str, required=False, help="Person name")
    parser.add_argument("--source",  type=str, default=None,   help="Source image directory")
    parser.add_argument("--webcam",  action="store_true",       help="Capture from webcam")
    parser.add_argument("--count",   type=int, default=5,       help="Number of webcam captures")
    parser.add_argument("--gallery", type=str, default="known_faces/", help="Gallery directory")
    parser.add_argument("--list",    action="store_true",       help="List enrolled identities")
    args = parser.parse_args()

    if args.list:
        list_gallery(args.gallery)
        return

    if not args.name:
        parser.error("--name is required unless using --list")

    if args.source:
        enrol_from_directory(args.name, args.source, args.gallery)
    elif args.webcam:
        enrol_from_webcam(args.name, args.gallery, count=args.count)
    else:
        parser.error("Provide --source <dir> or --webcam")


if __name__ == "__main__":
    main()
