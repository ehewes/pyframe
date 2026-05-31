from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import cv2
import numpy as np

from .errors import MediaDecodeError, UnsupportedMediaError

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
ANIMATION_EXTS = {".gif", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}

# Fallback frame rate when a container reports no usable FPS (common for GIFs).
DEFAULT_FPS = 10.0


class MediaKind(str, Enum):
    IMAGE = "image"
    ANIMATION = "animation"  # gif or video


@dataclass
class Frame:
    index: int
    timestamp: float  # seconds from start
    image: "np.ndarray"
    motion_score: float = 0.0

    def to_pil(self):
        from PIL import Image

        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


def media_kind(source: str | os.PathLike) -> MediaKind:
    ext = os.path.splitext(str(source))[1].lower()
    if ext in IMAGE_EXTS:
        return MediaKind.IMAGE
    if ext in ANIMATION_EXTS:
        return MediaKind.ANIMATION
    raise UnsupportedMediaError(
        f"Unsupported file type: {ext or '(none)'}. "
        f"Supported: {', '.join(sorted(IMAGE_EXTS | ANIMATION_EXTS))}"
    )


def _read_image(source: str | os.PathLike) -> "np.ndarray":
    img = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    # Fallback to PIL for formats OpenCV's build may not handle (e.g. some webp).
    try:
        from PIL import Image

        with Image.open(source) as pil:
            rgb = pil.convert("RGB")
            return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise MediaDecodeError(f"Could not read image {source}: {exc}") from exc


def iter_frames(source: str | os.PathLike) -> Iterator[Frame]:
    if not os.path.exists(source):
        raise FileNotFoundError(f"File not found: {source}")

    kind = media_kind(source)
    if kind is MediaKind.IMAGE:
        yield Frame(index=0, timestamp=0.0, image=_read_image(source), motion_score=0.0)
        return

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise MediaDecodeError(f"Could not open {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = DEFAULT_FPS

    prev_gray = None
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (64, 64))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                motion = 0.0
            else:
                motion = float(np.sum(cv2.absdiff(gray, prev_gray)))
            prev_gray = gray
            yield Frame(index=index, timestamp=index / fps, image=frame, motion_score=motion)
            index += 1
    finally:
        cap.release()
