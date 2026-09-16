"""Decoding media into frames, in two separable halves.

Sampling decisions need only per-frame metadata (index, timestamp, motion); backends
need only the pixels of the frames finally selected. Nothing needs both at once, so
decoding is split: `iter_frame_meta` walks the whole file holding one frame at a time,
and `iter_frames_at` re-walks it to materialise just the selected frames. That keeps
peak memory proportional to the sample rather than to the length of the media.

`iter_frames` (everything, with pixels) remains for callers that want it, and is built
on the same private generators so the two halves cannot drift apart.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

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


class FrameLike(Protocol):
    """What the samplers read. Frame and FrameMeta both satisfy it structurally."""

    index: int
    timestamp: float
    motion_score: float


@dataclass(frozen=True, slots=True)
class FrameMeta:
    """A frame's position and motion, without its pixels.

    Tens of bytes against several megabytes for a decoded Frame, which is what lets the
    sampling pass hold the whole timeline at once.
    """

    index: int
    timestamp: float  # seconds from start
    motion_score: float = 0.0


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


def _motion_gray(bgr: "np.ndarray") -> "np.ndarray":
    return cv2.cvtColor(cv2.resize(bgr, (64, 64)), cv2.COLOR_BGR2GRAY)


def _motion_score(gray: "np.ndarray", prev_gray) -> float:
    if prev_gray is None:
        return 0.0
    return float(np.sum(cv2.absdiff(gray, prev_gray)))


def _require_file(source: str | os.PathLike) -> None:
    if not os.path.exists(source):
        raise FileNotFoundError(f"File not found: {source}")


def _iter_video(source: str | os.PathLike) -> Iterator[tuple[FrameMeta, "np.ndarray"]]:
    """Every frame of a cv2-decodable file as (meta, BGR). One frame alive at a time;
    the caller decides what to keep. Single source of truth for index, timestamp and
    motion, so the metadata pass and the pixel pass cannot disagree."""
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
            gray = _motion_gray(frame)
            motion = _motion_score(gray, prev_gray)
            prev_gray = gray
            yield FrameMeta(index=index, timestamp=index / fps, motion_score=motion), frame
            index += 1
    finally:
        cap.release()


def _open_bytes(data: bytes):
    """Open in-memory media and probe its frame count. Shared by all three bytes
    entry points so the 'no video bytes' message stays identical across them."""
    import io

    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data))
        return img, getattr(img, "n_frames", 1)
    except Exception as exc:
        raise MediaDecodeError(
            f"could not decode bytes in memory: {exc} "
            "(video bytes are not supported by scan_bytes; use the path-based API)"
        ) from exc


def _iter_bytes(data: bytes) -> Iterator[tuple[FrameMeta, "np.ndarray"]]:
    """The in-memory counterpart of _iter_video, via Pillow. GIF timestamps accumulate
    per-frame durations rather than dividing by a container fps."""
    img, n_frames = _open_bytes(data)

    if n_frames <= 1:
        rgb = np.asarray(img.convert("RGB"))
        yield FrameMeta(index=0, timestamp=0.0), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return

    prev_gray = None
    timestamp = 0.0
    for index in range(n_frames):
        img.seek(index)
        rgb = np.asarray(img.convert("RGB"))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = _motion_gray(frame)
        motion = _motion_score(gray, prev_gray)
        prev_gray = gray
        yield FrameMeta(index=index, timestamp=timestamp, motion_score=motion), frame
        timestamp += (img.info.get("duration") or 100) / 1000.0  # per-frame GIF duration (ms)


def _as_frame(meta: FrameMeta, image: "np.ndarray") -> Frame:
    return Frame(
        index=meta.index,
        timestamp=meta.timestamp,
        image=image,
        motion_score=meta.motion_score,
    )


def _ordered_unique(metas: Sequence[FrameLike]) -> list[FrameLike]:
    by_index = {m.index: m for m in metas}
    return [by_index[i] for i in sorted(by_index)]


def iter_frames(source: str | os.PathLike) -> Iterator[Frame]:
    """Every frame with its pixels. Holds one frame at a time itself, but `list()` of it
    is proportional to the length of the media; prefer iter_frame_meta + iter_frames_at
    when you only need a sample."""
    _require_file(source)

    if media_kind(source) is MediaKind.IMAGE:
        yield Frame(index=0, timestamp=0.0, image=_read_image(source), motion_score=0.0)
        return

    for meta, image in _iter_video(source):
        yield _as_frame(meta, image)


def iter_frame_meta(source: str | os.PathLike) -> Iterator[FrameMeta]:
    """Pass one: the whole timeline, no pixels retained. Raises what iter_frames raises,
    at the same point, except that a still image is not decoded until it is fetched."""
    _require_file(source)

    if media_kind(source) is MediaKind.IMAGE:
        yield FrameMeta(index=0, timestamp=0.0)
        return

    for meta, _image in _iter_video(source):
        yield meta


def iter_frames_at(source: str | os.PathLike, metas: Sequence[FrameLike]) -> Iterator[Frame]:
    """Pass two: re-decode and yield only the frames named by `metas`, in index order.

    Skips unwanted frames with cap.grab() rather than seeking. CAP_PROP_POS_FRAMES is an
    approximate keyframe seek on long-GOP codecs, VFR containers and GIF, so it can land
    on a neighbouring frame; this pass must return exactly the frame pass one measured.
    Do not replace the walk with a seek.

    timestamp and motion_score are carried from `metas`, never recomputed: motion is a
    diff against the previous *decoded* frame, so recomputing it here would measure
    across the skipped gap instead.
    """
    wanted = _ordered_unique(metas)
    if not wanted:
        return

    _require_file(source)
    if media_kind(source) is MediaKind.IMAGE:
        image = _read_image(source)
        for meta in wanted:
            yield _as_frame(meta, image)
        return

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise MediaDecodeError(f"Could not open {source}")

    pending = iter(wanted)
    target = next(pending)
    index = 0
    try:
        while True:
            if index != target.index:
                if not cap.grab():
                    break
                index += 1
                continue
            ok, image = cap.read()
            if not ok:
                break
            yield _as_frame(target, image)
            index += 1
            try:
                target = next(pending)
            except StopIteration:
                return
    finally:
        cap.release()

    raise MediaDecodeError(
        f"{source}: second decode pass ended at frame {index}, before the selected "
        f"frame {target.index}. The file may decode non-deterministically."
    )


def iter_frames_from_bytes(data: bytes) -> Iterator[Frame]:
    """Decode a GIF / static image from memory (no disk). Pillow only; for video
    bytes use the path-based API. Motion + timestamps match iter_frames."""
    for meta, image in _iter_bytes(data):
        yield _as_frame(meta, image)


def iter_frame_meta_from_bytes(data: bytes) -> Iterator[FrameMeta]:
    """Pass one for the in-memory path."""
    for meta, _image in _iter_bytes(data):
        yield meta


def iter_frames_from_bytes_at(data: bytes, metas: Sequence[FrameLike]) -> Iterator[Frame]:
    """Pass two for the in-memory path. Pillow replays the GIF from frame 0, so this
    walks forward and keeps the matches rather than seeking."""
    wanted = {m.index: m for m in metas}
    if not wanted:
        return

    for meta, image in _iter_bytes(data):
        target = wanted.pop(meta.index, None)
        if target is None:
            continue
        yield _as_frame(target, image)
        if not wanted:
            return

    raise MediaDecodeError(
        f"second decode pass ended before the selected frame(s) {sorted(wanted)}"
    )
