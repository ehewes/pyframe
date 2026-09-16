"""The decode path must hold a sample, not a timeline.

These assert peak *live Frame count*, not bytes. Counting objects is exact and platform
independent, where tracemalloc may not observe cv2's buffers at all and ru_maxrss is a
high water mark reported in bytes on macOS and kilobytes on Linux. Each bound is paired
with a positive control that fails if the tracker ever stops working, because a memory
test that silently measures nothing is worse than no test.
"""

import weakref

import numpy as np
import pytest
from PIL import Image

from pyframe import media as media_mod
from pyframe.backends.base import Backend
from pyframe.config import Config, PrescreenConfig
from pyframe.scanner import Scanner

N_FRAMES = 240


@pytest.fixture
def long_gif(tmp_path):
    path = tmp_path / "long.gif"
    imgs = [
        Image.fromarray(np.full((32, 32, 3), (i * 7) % 256, np.uint8)) for i in range(N_FRAMES)
    ]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=20, loop=0)
    return str(path)


class FrameTracker:
    """Counts live Frames by wrapping the constructor media.py resolves at call time."""

    def __init__(self, monkeypatch):
        self.live = self.peak = self.total = 0
        real = media_mod.Frame

        def factory(*args, **kwargs):
            frame = real(*args, **kwargs)
            self.live += 1
            self.total += 1
            self.peak = max(self.peak, self.live)
            weakref.finalize(frame, self._released)
            return frame

        monkeypatch.setattr(media_mod, "Frame", factory)

    def _released(self):
        self.live -= 1


class FlatBackend(Backend):
    name = "fake"
    cost_per_image = 0.0
    default_min_confidence = 0.5

    def __init__(self, score=0.0, name="fake"):
        self.name = name
        self.score = score

    def _score(self, image):
        return self.score, [], None


def test_iter_frames_holds_the_whole_clip(monkeypatch, long_gif):
    # Positive control. If this does not peak at the full length then the tracker is
    # broken and every bound below is vacuous.
    tracker = FrameTracker(monkeypatch)

    frames = list(media_mod.iter_frames(long_gif))

    assert len(frames) == N_FRAMES
    assert tracker.peak == N_FRAMES


def test_meta_pass_holds_one_frame_at_a_time(monkeypatch, long_gif):
    tracker = FrameTracker(monkeypatch)

    metas = list(media_mod.iter_frame_meta(long_gif))

    assert len(metas) == N_FRAMES  # the whole timeline is still measured
    assert tracker.total == 0  # and not one Frame was built to do it


def test_single_pass_decodes_only_the_sample(monkeypatch, long_gif):
    tracker = FrameTracker(monkeypatch)
    config = Config(backend=FlatBackend(), max_frames=10)

    result = Scanner.from_config(config).scan(long_gif)

    assert result.frames_total == N_FRAMES  # every frame was considered
    assert tracker.total == 10  # only the selected ones were decoded
    assert tracker.peak <= 12


def test_cascade_streams_the_screen_pass(monkeypatch, long_gif):
    # The screen set is screen_fps x duration, not max_frames, so this is the pass that
    # would still blow up if only the single-pass path had been bounded.
    tracker = FrameTracker(monkeypatch)
    config = Config(
        backend=FlatBackend(name="aws"),
        screen_backend=FlatBackend(name="local"),
        max_frames=10,
        prescreen=PrescreenConfig(enabled=True, screen_fps=2.0),
    )

    result = Scanner.from_config(config).scan(long_gif)

    assert result.frames_screened > 1
    assert result.escalated is False
    assert tracker.peak <= 4


def test_cascade_escalation_stays_within_its_budget(monkeypatch, long_gif):
    tracker = FrameTracker(monkeypatch)
    config = Config(
        backend=FlatBackend(score=0.99, name="aws"),
        screen_backend=FlatBackend(score=0.99, name="local"),
        max_frames=10,
        frames_per_batch=2,
        prescreen=PrescreenConfig(enabled=True, screen_fps=2.0, max_escalations=2),
    )

    result = Scanner.from_config(config).scan(long_gif)

    assert result.escalated is True
    # Every screened frame flagged, yet only max_escalations x frames_per_batch are
    # ever held at once.
    assert tracker.peak <= 6


def test_scan_bytes_is_bounded_too(monkeypatch, long_gif):
    with open(long_gif, "rb") as fh:
        data = fh.read()
    tracker = FrameTracker(monkeypatch)
    config = Config(backend=FlatBackend(), max_frames=10)

    result = Scanner.from_config(config).scan_bytes(data)

    assert result.frames_total == N_FRAMES
    assert tracker.total == 10
    assert tracker.peak <= 12
