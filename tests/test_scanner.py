import time

import numpy as np

from pyframe.backends.base import Backend
from pyframe.config import Config, PrescreenConfig
from pyframe.media import Frame, MediaKind
from pyframe.results import Severity
from pyframe.scanner import Scanner


class FakeBackend(Backend):
    # Scores a frame by its brightest pixel (0..1), so tests control "nsfw-ness"
    # by how bright a frame is. Using max (not mean) means a merged grid that
    # contains a bright frame still scores high, like a real detector would.
    def __init__(self, name="fake", cost=0.0):
        self.name = name
        self.cost_per_image = cost

    def _score(self, image):
        return float(np.asarray(image).max()) / 255.0, [], None


def _frames(values, fps=10.0):
    out = []
    for i, v in enumerate(values):
        out.append(Frame(index=i, timestamp=i / fps, image=np.full((8, 8, 3), v, np.uint8)))
    return out


def _scanner(precise, screen=None, **prescreen):
    cfg = Config(min_confidence=0.8, prescreen=PrescreenConfig(screen_fps=100.0, **prescreen))
    return Scanner(precise, screen=screen, config=cfg)


def test_single_pass_flags_bright_frame():
    frames = _frames([10] * 9 + [250])
    scanner = _scanner(FakeBackend(cost=0.001))
    result = scanner._single_pass("clip.gif", MediaKind.ANIMATION, frames, time.perf_counter())
    assert result.is_nsfw
    assert result.verdict is Severity.NSFW
    assert result.cost_usd > 0


def test_cascade_short_circuits_clean_media():
    frames = _frames([10] * 20)
    scanner = _scanner(FakeBackend("aws", cost=0.001), screen=FakeBackend("local"), enabled=True)
    result = scanner._cascade("clip.gif", MediaKind.ANIMATION, frames, time.perf_counter())
    assert not result.is_nsfw
    assert result.escalated is False
    assert result.frames_classified == 0
    assert result.cost_usd == 0  # never touched the precise backend


def test_cascade_escalates_top_suspicious_as_merged():
    frames = _frames([10] * 20)
    frames[12].image[:] = 250  # one suspicious frame
    scanner = _scanner(FakeBackend("aws", cost=0.001), screen=FakeBackend("local"), enabled=True)
    result = scanner._cascade("clip.gif", MediaKind.ANIMATION, frames, time.perf_counter())
    assert result.is_nsfw
    assert result.escalated is True
    assert 0 < result.frames_classified <= 2  # merged grids, capped
    assert "local" in result.backends_used and "aws" in result.backends_used


def test_cascade_caps_aws_calls_at_max_escalations():
    frames = _frames([250] * 40)  # every frame flags
    scanner = _scanner(FakeBackend("aws", 0.001), screen=FakeBackend("local"), enabled=True, max_escalations=2)
    result = scanner._cascade("c.gif", MediaKind.ANIMATION, frames, time.perf_counter())
    assert result.escalated is True
    assert result.frames_classified <= 2  # hard cap, regardless of how many frames flag


def test_cascade_pads_to_full_grid_when_one_frame_flagged():
    frames = _frames([10] * 20)
    frames[5].image[:] = 250  # only one suspicious frame
    scanner = _scanner(FakeBackend("aws", 0.001), screen=FakeBackend("local"), enabled=True, max_escalations=1)
    result = scanner._cascade("c.gif", MediaKind.ANIMATION, frames, time.perf_counter())
    assert result.escalated is True
    assert result.frames_classified == 1  # one merged grid (the flagged frame plus a neighbor)
    assert result.is_nsfw


def test_scan_bytes_decodes_in_memory():
    import io

    from PIL import Image

    from pyframe.media import iter_frames_from_bytes

    # distinct fills so PIL's GIF optimizer doesn't collapse identical frames
    pil = [Image.fromarray(np.full((16, 16, 3), v, np.uint8)) for v in (10, 60, 250, 120, 30)]
    buf = io.BytesIO()
    pil[0].save(buf, format="GIF", save_all=True, append_images=pil[1:], duration=80, loop=0)
    data = buf.getvalue()

    decoded = list(iter_frames_from_bytes(data))
    assert len(decoded) == 5  # decoded from memory, no disk
    assert any(f.motion_score > 0 for f in decoded)

    scanner = _scanner(FakeBackend("aws", 0.001), screen=FakeBackend("local"), enabled=True)
    result = scanner.scan_bytes(data, label="x.gif")
    assert result.media_kind == "animation"
    assert result.frames_total == 5


def test_cascade_fail_open_escalates_on_error():
    class BrokenScreen(Backend):
        name = "local"
        cost_per_image = 0.0

        def _score(self, image):
            raise RuntimeError("decode failed")

    frames = _frames([10] * 12)
    scanner = _scanner(FakeBackend("aws", cost=0.001), screen=BrokenScreen(), enabled=True, fail_open=True)
    result = scanner._cascade("clip.gif", MediaKind.ANIMATION, frames, time.perf_counter())
    assert result.frames_classified > 0  # errors were escalated, not silently cleared


def test_motion_sampler_always_includes_time_coverage_floor():
    from pyframe.sampling import DenseUniformSampler

    # 40 frames at 10 fps (4s). Motion is concentrated in the first 10 frames; the
    # rest are static. Pure motion bucketing over-picks the busy head and leaves whole
    # static time slices unsampled. The recall floor must still cover them.
    frames = [
        Frame(index=i, timestamp=i / 10.0, image=np.zeros((8, 8, 3), np.uint8),
              motion_score=100.0 if i < 10 else 0.0)
        for i in range(40)
    ]
    cfg = Config(sampler="motion", max_frames=10, prescreen=PrescreenConfig(screen_fps=2.0))
    scanner = Scanner(FakeBackend(), config=cfg)

    selected = scanner._motion_select_with_floor(frames)
    floor_idx = {f.index for f in DenseUniformSampler(2.0).select(frames)}
    selected_idx = {f.index for f in selected}

    assert floor_idx <= selected_idx  # time-coverage floor is never dropped for motion
    assert len(selected) <= cfg.max_frames  # cost cap still respected
    assert [f.index for f in selected] == sorted(selected_idx)  # returned in index order
