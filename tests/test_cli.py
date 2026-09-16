import json
import sys

import numpy as np
import pytest
from PIL import Image

from pyframe import cli
from pyframe import scanner as scanner_mod
from pyframe.backends.base import Backend


class ScriptedBackend(Backend):
    # Scores every frame the same, or fails on every frame, so a CLI test can pin one
    # exact verdict without downloading a model.
    cost_per_image = 0.0
    default_min_confidence = 0.5

    def __init__(self, score=0.0, fail_with=None, name="fake"):
        self.name = name
        self.score = score
        self.fail_with = fail_with

    def _score(self, image):
        if self.fail_with is not None:
            raise RuntimeError(self.fail_with)
        return self.score, [], None


@pytest.fixture
def gif(tmp_path):
    path = tmp_path / "clip.gif"
    imgs = [Image.fromarray(np.full((32, 32, 3), v, np.uint8)) for v in (10, 60, 120)]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=80, loop=0)
    return str(path)


def _run(monkeypatch, argv, backend=None):
    """Drive cli.main() end to end. Omit `backend` to let the real load_backend run."""
    if backend is not None:
        monkeypatch.setattr(scanner_mod, "load_backend", lambda *a, **k: backend)
    monkeypatch.setattr(sys, "argv", ["pyframe", *argv])
    return cli.main()


def test_clean_media_exits_zero(monkeypatch, gif):
    assert _run(monkeypatch, [gif], ScriptedBackend(score=0.01)) == 0


def test_nsfw_media_exits_one(monkeypatch, gif):
    assert _run(monkeypatch, [gif], ScriptedBackend(score=0.99)) == 1


def test_errored_scan_exits_four(monkeypatch, capsys, gif):
    # Every frame failing means nothing was cleared. Exiting 0 here would make
    # `pyframe upload.gif || reject` accept every upload while the backend is down.
    rc = _run(monkeypatch, [gif], ScriptedBackend(fail_with="credentials expired"))

    assert rc == 4
    assert "credentials expired" in capsys.readouterr().err


def test_fail_on_never_exits_zero_even_on_error(monkeypatch, gif):
    # The explicit "don't gate me, I just want the JSON" escape hatch.
    assert _run(monkeypatch, [gif, "--fail-on", "never"], ScriptedBackend(fail_with="boom")) == 0


def test_fail_on_never_exits_zero_on_nsfw(monkeypatch, gif):
    assert _run(monkeypatch, [gif, "--fail-on", "never"], ScriptedBackend(score=0.99)) == 0


def test_fail_on_uncertain_gates_the_middle_band(monkeypatch, gif):
    # 0.4 sits under the 0.5 threshold but over uncertain_threshold 0.3.
    assert _run(monkeypatch, [gif, "--fail-on", "uncertain"], ScriptedBackend(score=0.4)) == 1
    assert _run(monkeypatch, [gif], ScriptedBackend(score=0.4)) == 0


def test_missing_file_exits_two(monkeypatch, tmp_path):
    assert _run(monkeypatch, [str(tmp_path / "nope.gif")], ScriptedBackend()) == 2


def test_unsupported_type_exits_two(monkeypatch, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")

    assert _run(monkeypatch, [str(path)], ScriptedBackend()) == 2


def test_unknown_backend_exits_two_without_a_traceback(monkeypatch, capsys, gif):
    rc = _run(monkeypatch, [gif, "--backend", "bogus"])  # deliberately unpatched

    assert rc == 2
    assert "bogus" in capsys.readouterr().err


def test_max_escalations_of_zero_is_rejected(monkeypatch, capsys, gif):
    # 0 used to mean "no cap": the suspicion sampler keeps every frame on a non-positive
    # budget, so the flag whose job is bounding spend removed the bound instead.
    rc = _run(monkeypatch, [gif, "--prescreen", "--max-escalations", "0"], ScriptedBackend())

    assert rc == 2
    assert "max_escalations" in capsys.readouterr().err


def test_json_output_carries_the_documented_keys(monkeypatch, capsys, gif):
    rc = _run(monkeypatch, [gif, "--json"], ScriptedBackend(score=0.01))
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    for key in (
        "source", "media_kind", "verdict", "is_nsfw", "max_score", "worst_frame",
        "frames", "backends_used", "frames_total", "frames_screened",
        "frames_classified", "cost_usd", "prescreen_used", "escalated", "windows",
        "elapsed_s",
    ):
        assert key in payload, key


def test_json_verdict_and_is_nsfw_agree(monkeypatch, capsys, gif):
    _run(monkeypatch, [gif, "--json"], ScriptedBackend(score=0.99))
    payload = json.loads(capsys.readouterr().out)

    assert payload["verdict"] == "nsfw"
    assert payload["is_nsfw"] is True


def test_batch_exit_code_is_the_worst_across_files(monkeypatch, gif, tmp_path):
    rc = _run(monkeypatch, [gif, str(tmp_path / "nope.gif")], ScriptedBackend(score=0.99))

    assert rc == 2  # bad input (2) outranks nsfw (1)
