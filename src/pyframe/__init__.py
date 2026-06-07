from __future__ import annotations

from .backends import Backend, clear_backend_cache, load_backend
from .config import Config, PrescreenConfig
from .errors import (
    BackendUnavailableError,
    MediaDecodeError,
    PyFrameError,
    UnsupportedMediaError,
)
from .image_utils import merge_images_to_grid, merge_to_grid
from .media import Frame, MediaKind, iter_frames, iter_frames_from_bytes, media_kind
from .pipe import Pipe, scan, scan_bytes
from .results import Label, ScanResult, Severity, Verdict
from .scanner import Scanner

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("pyframe-gif-video-image-moderation")
    except PackageNotFoundError:
        __version__ = "0.3.0"
except Exception:
    __version__ = "0.3.0"

__all__ = [
    "Pipe",
    "scan",
    "scan_bytes",
    "Scanner",
    "Config",
    "PrescreenConfig",
    "ScanResult",
    "Verdict",
    "Label",
    "Severity",
    "Backend",
    "load_backend",
    "clear_backend_cache",
    "Frame",
    "MediaKind",
    "iter_frames",
    "iter_frames_from_bytes",
    "media_kind",
    "merge_to_grid",
    "merge_images_to_grid",
    "PyFrameError",
    "UnsupportedMediaError",
    "MediaDecodeError",
    "BackendUnavailableError",
    "__version__",
]


def __getattr__(name):
    if name == "LocalBackend":
        from .backends.local import LocalBackend

        return LocalBackend
    if name == "RekognitionBackend":
        from .backends.aws import RekognitionBackend

        return RekognitionBackend
    if name == "video_to_gif":
        from .video import video_to_gif

        return video_to_gif
    raise AttributeError(f"module 'pyframe' has no attribute {name!r}")
