from __future__ import annotations

import os

from .errors import BackendUnavailableError


def video_to_gif(video_path, output_path=None, fps: int = 15, resize_width=None) -> str:
    try:
        from moviepy import VideoFileClip
    except ImportError as exc:
        raise BackendUnavailableError(
            "video", "pip install 'pyframe-gif-video-image-moderation[video]'"
        ) from exc

    if output_path is None:
        stem = os.path.splitext(os.path.basename(str(video_path)))[0]
        output_path = f"{stem}.gif"

    clip = VideoFileClip(str(video_path))
    try:
        if resize_width:
            # moviepy 2.x renamed resize -> resized
            clip = clip.resized(width=resize_width) if hasattr(clip, "resized") else clip.resize(width=resize_width)
        clip.write_gif(output_path, fps=fps)
    finally:
        clip.close()
    return output_path
