"""
src/face_recognition_module.py
Face recognition layer using DeepFace.

Responsibilities:
  • Enrol known faces from a directory
  • Identify faces cropped from tracked persons
  • Maintain a per-track identity cache to avoid re-running every frame
  • Optionally AUTO-ENROL unrecognised faces under a pseudonymous id
    (fid_1, fid_2, ...) instead of pre-enrolling named people up front —
    opt-in via auto_enroll=True, off by default.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# ─────────────────────────────────────────────────────────────
#  Recognition result
# ─────────────────────────────────────────────────────────────

class RecognitionResult:
    def __init__(
        self,
        identity: str,
        distance: float,
        model: str,
        verified: bool,
    ):
        self.identity = identity          # name or "unknown"
        self.distance = distance
        self.model = model
        self.verified = verified

    def __repr__(self):
        return (
            f"RecognitionResult(identity={self.identity!r}, "
            f"distance={self.distance:.3f}, verified={self.verified})"
        )


# ─────────────────────────────────────────────────────────────
#  Recogniser
# ─────────────────────────────────────────────────────────────

class FaceRecogniser:
    """
    Wraps DeepFace for face recognition against a gallery of known faces.

    Parameters
    ----------
    known_faces_dir   : Directory containing sub-dirs named after each person,
                        each containing one or more face images.
                        Structure:
                          known_faces/
                            Alice/
                              alice_01.jpg
                            Bob/
                              bob_01.jpg
    model_name        : DeepFace model ('VGG-Face', 'Facenet', 'ArcFace', …)
    detector_backend  : Face detector ('opencv', 'ssd', 'mtcnn', 'retinaface')
    distance_metric   : 'cosine', 'euclidean', 'euclidean_l2'
    threshold         : Distance threshold — below = match
    min_face_size     : Minimum face dimension in pixels to attempt recognition
    cache_ttl         : Seconds to cache a per-track identity (avoids re-running)
    auto_enroll       : If True, a face that doesn't match anyone in the
                        gallery (including previously auto-enrolled people)
                        is automatically saved under a new pseudonymous id
                        (fid_1, fid_2, ...) instead of being labelled
                        'unknown'. No names are ever attached — just a
                        persistent anonymous id, so the same person is
                        recognised as "the same fid" on return, without a
                        name-to-face mapping being stored anywhere.
                        Off by default; existing named-enrollment behaviour
                        is unchanged unless this is explicitly turned on.
    auto_enroll_prefix: Prefix for auto-enrolled ids (default "fid").
    """

    UNKNOWN = "unknown"

    def __init__(
        self,
        known_faces_dir: str | Path = "known_faces/",
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        distance_metric: str = "cosine",
        threshold: float = 0.40,
        min_face_size: int = 40,
        cache_ttl: float = 5.0,
        auto_enroll: bool = False,
        auto_enroll_prefix: str = "fid",
        face_region_top_fraction: float = 0.5,
    ):
        self.known_faces_dir = Path(known_faces_dir)
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.min_face_size = min_face_size
        self.cache_ttl = cache_ttl
        self.auto_enroll = auto_enroll
        self.auto_enroll_prefix = auto_enroll_prefix
        self.face_region_top_fraction = face_region_top_fraction

        # Per-track identity cache: {track_id: (identity, timestamp)}
        self._cache: dict[int, tuple[str, float]] = {}

        self._deepface_available = False
        self._db_built = False

        self._try_import_deepface()

        if self._deepface_available and self.known_faces_dir.exists():
            self._verify_gallery()

        # Resume auto-enroll numbering from whatever's already on disk,
        # so restarting the pipeline doesn't collide with earlier fid_N's.
        self._next_fid = self._scan_existing_fid_counter()
        if self.auto_enroll:
            logger.info(
                f"[FaceRec] Auto-enroll ON — next new face will be "
                f"'{self.auto_enroll_prefix}_{self._next_fid}'"
            )

    # ── Setup ─────────────────────────────────────────────────

    def _try_import_deepface(self) -> None:
        try:
            import deepface  # noqa: F401
            self._deepface_available = True
            logger.info(f"[FaceRec] DeepFace loaded. Model={self.model_name}")
        except ImportError:
            logger.warning(
                "[FaceRec] DeepFace not installed — face recognition disabled.\n"
                "          Run: pip install deepface"
            )

    def _verify_gallery(self) -> None:
        """Count enrolled identities and warn if gallery is empty."""
        if not self.known_faces_dir.exists():
            logger.warning(f"[FaceRec] Gallery dir not found: {self.known_faces_dir}")
            return

        identities = [
            d for d in self.known_faces_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        total_imgs = sum(
            len(list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.jpeg")))
            for d in identities
        )

        if not identities:
            logger.warning(
                "[FaceRec] No enrolled identities found in gallery.\n"
                f"          Add sub-directories to: {self.known_faces_dir}"
            )
        else:
            logger.info(
                f"[FaceRec] Gallery: {len(identities)} identities, {total_imgs} images"
            )
            self._db_built = True

    def _scan_existing_fid_counter(self) -> int:
        """Find the highest existing fid_N folder so restarts don't collide."""
        if not self.known_faces_dir.exists():
            return 1
        pattern = re.compile(rf"^{re.escape(self.auto_enroll_prefix)}_(\d+)$")
        highest = 0
        for d in self.known_faces_dir.iterdir():
            if d.is_dir():
                m = pattern.match(d.name)
                if m:
                    highest = max(highest, int(m.group(1)))
        return highest + 1

    def _gallery_has_identities(self) -> bool:
        if not self.known_faces_dir.exists():
            return False
        return any(
            d.is_dir() and not d.name.startswith(".")
            for d in self.known_faces_dir.iterdir()
        )

    def _extract_face_region(self, crop: np.ndarray) -> np.ndarray:
        """
        track.crop is the FULL PERSON bounding box (body + background),
        not a tight face crop — DeepFace has to find a face somewhere
        inside that larger, noisier image every single call, which is
        inconsistent frame-to-frame with the lightweight 'opencv' detector
        backend even for the same real person. Cropping to the top portion
        (where the head actually is) gives DeepFace's internal detector a
        far easier, more consistent target — used both when matching AND
        when saving a new enrollment, so gallery and query images are
        cropped the same way.
        """
        if crop is None or crop.size == 0:
            return crop
        h = crop.shape[0]
        cutoff = max(1, int(h * self.face_region_top_fraction))
        return crop[:cutoff, :]

    # ── Public API ────────────────────────────────────────────

    def is_available(self) -> bool:
        # In auto-enroll mode, the recogniser is "available" as soon as
        # DeepFace itself is importable — it doesn't need a pre-existing
        # gallery, since it builds one as people are seen.
        if self.auto_enroll:
            return self._deepface_available
        return self._deepface_available and self._db_built

    def identify_track(
        self,
        track_id: int,
        person_crop: np.ndarray,
    ) -> str:
        """
        Return the identity for a track.
        Uses cache if within TTL, otherwise runs recognition.

        If auto_enroll=True, an unrecognised face is enrolled on the spot
        under a new pseudonymous id instead of being returned as
        'unknown' — see class docstring.
        """
        if self.auto_enroll:
            return self._identify_or_enroll(track_id, person_crop)

        if not self.is_available():
            return self.UNKNOWN

        # Check cache
        now = time.time()
        if track_id in self._cache:
            identity, ts = self._cache[track_id]
            if now - ts < self.cache_ttl:
                return identity

        # Run recognition
        result = self._recognise(person_crop)
        identity = result.identity if result.verified else self.UNKNOWN
        self._cache[track_id] = (identity, now)
        return identity

    def _identify_or_enroll(self, track_id: int, person_crop: np.ndarray) -> str:
        """Auto-enroll branch used when self.auto_enroll is True."""
        now = time.time()
        if track_id in self._cache:
            identity, ts = self._cache[track_id]
            if now - ts < self.cache_ttl:
                return identity

        if person_crop is None or person_crop.size == 0:
            return self.UNKNOWN
        h, w = person_crop.shape[:2]
        if h < self.min_face_size or w < self.min_face_size:
            return self.UNKNOWN

        # Cold start: nobody enrolled yet, skip straight to enrolling —
        # DeepFace.find has nothing to compare against regardless.
        if self._gallery_has_identities():
            result = self._recognise(person_crop)
        else:
            result = RecognitionResult(self.UNKNOWN, 1.0, self.model_name, False)

        if result.verified:
            identity = result.identity
        else:
            identity = f"{self.auto_enroll_prefix}_{self._next_fid}"
            face_region = self._extract_face_region(person_crop)
            if self.enrol(identity, face_region):
                logger.info(f"[FaceRec] New face auto-enrolled -> {identity}")
                self._next_fid += 1
            else:
                identity = self.UNKNOWN

        self._cache[track_id] = (identity, now)
        return identity

    def identify_frame(self, person_crop: np.ndarray) -> RecognitionResult:
        """One-shot recognition without caching (for offline use)."""
        if not self.is_available():
            return RecognitionResult(self.UNKNOWN, 1.0, self.model_name, False)
        return self._recognise(person_crop)

    def enrol(self, name: str, image: np.ndarray) -> bool:
        """
        Save a face image to the gallery for a given person.
        Returns True on success.
        """
        person_dir = self.known_faces_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)
        idx = len(list(person_dir.iterdir()))
        save_path = person_dir / f"{name}_{idx:03d}.jpg"
        ok = cv2.imwrite(str(save_path), image)
        if ok:
            logger.info(f"[FaceRec] Enrolled {name} → {save_path}")
            self._db_built = True
        return ok

    def clear_cache(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._cache.clear()
        else:
            self._cache.pop(track_id, None)

    # ── Internal ─────────────────────────────────────────────

    def _recognise(self, crop: np.ndarray) -> RecognitionResult:
        """Run DeepFace.find against the known gallery."""
        from deepface import DeepFace

        crop = self._extract_face_region(crop)

        if crop is None or crop.size == 0:
            return RecognitionResult(self.UNKNOWN, 1.0, self.model_name, False)

        h, w = crop.shape[:2]
        if h < self.min_face_size or w < self.min_face_size:
            return RecognitionResult(self.UNKNOWN, 1.0, self.model_name, False)

        try:
            results = DeepFace.find(
                img_path=crop,
                db_path=str(self.known_faces_dir),
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=False,
                silent=True,
            )

            if not results or results[0].empty:
                logger.info(
                    "[FaceRec] no candidates returned by DeepFace.find "
                    "(gallery empty, or no face detected in the crop)"
                )
                return RecognitionResult(self.UNKNOWN, 1.0, self.model_name, False)

            # Best match
            df = results[0]
            dist_col = f"{self.model_name}_{self.distance_metric}"
            if dist_col not in df.columns:
                # Try to find any distance column
                dist_cols = [c for c in df.columns if self.distance_metric in c.lower()]
                dist_col = dist_cols[0] if dist_cols else df.columns[-1]

            df = df.sort_values(dist_col)
            best = df.iloc[0]
            distance = float(best[dist_col])
            verified = distance <= self.threshold

            # Extract name from path: known_faces/Alice/alice_01.jpg → Alice
            identity_path = Path(best["identity"])
            identity = identity_path.parent.name

            # Left visible at INFO level on purpose — this is what you tune
            # threshold against. If distance for the SAME real person keeps
            # coming back consistently above self.threshold, raise the
            # threshold to match reality rather than guessing a value.
            logger.info(
                f"[FaceRec] nearest='{identity}' distance={distance:.3f} "
                f"threshold={self.threshold} -> {'MATCH' if verified else 'no match'}"
            )

            return RecognitionResult(
                identity=identity,
                distance=distance,
                model=self.model_name,
                verified=verified,
            )

        except Exception as exc:
            logger.debug(f"[FaceRec] Recognition error (non-fatal): {exc}")
            return RecognitionResult(self.UNKNOWN, 1.0, self.model_name, False)