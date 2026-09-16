import sys


def test_base_import_has_no_heavy_deps():
    import pyframe

    assert pyframe.__version__
    assert callable(pyframe.scan)
    assert hasattr(pyframe, "Pipe")
    # The light base install must import without the optional backends present.
    assert "torch" not in sys.modules
    assert "boto3" not in sys.modules


def test_public_api_surface():
    import pyframe

    for name in ("Pipe", "scan", "Scanner", "ScanResult", "Verdict", "Backend", "load_backend"):
        assert hasattr(pyframe, name), name


def test_frame_still_constructs_positionally():
    # Frame is exported and its field order is public, so the metadata split must not
    # have reordered or defaulted its way into a breaking constructor.
    import numpy as np

    import pyframe

    frame = pyframe.Frame(3, 1.5, np.zeros((2, 2, 3), np.uint8))

    assert (frame.index, frame.timestamp, frame.motion_score) == (3, 1.5, 0.0)
    assert frame.image.shape == (2, 2, 3)


def test_two_pass_decode_names_are_exported():
    import pyframe

    for name in (
        "FrameMeta",
        "FrameLike",
        "iter_frame_meta",
        "iter_frames_at",
        "iter_frame_meta_from_bytes",
        "iter_frames_from_bytes_at",
    ):
        assert hasattr(pyframe, name), name
        assert name in pyframe.__all__, name


def test_unsupported_media_raises():
    import pytest

    import pyframe

    with pytest.raises(pyframe.UnsupportedMediaError):
        pyframe.media_kind("file.txt")
