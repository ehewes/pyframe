import time

import numpy as np
import pytest

from pyframe.backends.base import Backend
from pyframe.config import Config, PrescreenConfig
from pyframe.errors import MediaDecodeError
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


def test_all_frames_failing_reports_error_not_clean():
    class BrokenPrecise(Backend):
        name = "aws"
        cost_per_image = 0.001

        def _score(self, image):
            raise RuntimeError("credentials expired")

    frames = _frames([10] * 6)
    scanner = _scanner(BrokenPrecise())
    result = scanner._single_pass("clip.gif", MediaKind.ANIMATION, frames, time.perf_counter())

    assert result.verdict is Severity.ERROR
    assert result.is_nsfw is False  # an error is not a positive finding...
    assert result.max_score == 0.0  # ...but the CLI must not read it as a clean bill either


def test_short_circuited_cascade_keeps_is_nsfw_and_verdict_in_lockstep():
    # escalate_threshold above min_confidence: the screen scores high enough to be NSFW
    # but not high enough to escalate, so nothing is ever classified. is_nsfw used to be
    # computed only from the classified frames, so it read False while verdict read nsfw
    # -- and the CLI gates on is_nsfw.
    frames = _frames([220] * 10)  # 220/255 = 0.86, over min_confidence 0.8
    scanner = _scanner(
        FakeBackend("aws", 0.001), screen=FakeBackend("local"),
        enabled=True, escalate_threshold=0.95,
    )
    result = scanner._cascade("clip.gif", MediaKind.ANIMATION, frames, time.perf_counter())

    assert result.escalated is False
    assert result.frames_classified == 0
    assert result.verdict is Severity.NSFW
    assert result.is_nsfw is True


def test_media_that_decodes_to_nothing_raises_rather_than_reporting_clean():
    scanner = _scanner(FakeBackend())

    with pytest.raises(MediaDecodeError):
        scanner._scan_frames("clip.gif", MediaKind.ANIMATION, [], time.perf_counter())


def test_ensure_min_frames_fills_by_suspicion_not_motion():
    # Screen scores are 0..1 while motion_score is a pixel-diff sum reaching ~1e6, so a
    # single flat sort key ranked any moving frame above a screened frame that had
    # nearly flagged.
    frames = _frames([10] * 3)
    frames[1].motion_score = 0.0  # screened, scored just under the gate
    frames[2].motion_score = 1_000_000.0  # never screened, merely busy
    scores = {0: 0.9, 1: 0.4}

    selected = _scanner(FakeBackend())._ensure_min_frames([frames[0]], frames, scores, 2)

    assert [f.index for f in selected] == [0, 1]


def test_max_escalations_below_one_is_rejected():
    # A non-positive budget does not disable escalation, it uncaps it: SuspicionSampler
    # returns every frame when budget <= 0.
    cfg = Config(backend=FakeBackend(), prescreen=PrescreenConfig(enabled=True, max_escalations=0))

    with pytest.raises(ValueError, match="max_escalations"):
        Scanner.from_config(cfg)


def test_max_escalations_of_one_still_caps_at_one_call():
    frames = _frames([250] * 40)  # every frame flags
    scanner = _scanner(
        FakeBackend("aws", 0.001), screen=FakeBackend("local"), enabled=True, max_escalations=1
    )
    result = scanner._cascade("c.gif", MediaKind.ANIMATION, frames, time.perf_counter())

    assert result.frames_classified == 1


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
