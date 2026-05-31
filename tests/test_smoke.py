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


def test_unsupported_media_raises():
    import pytest

    import pyframe

    with pytest.raises(pyframe.UnsupportedMediaError):
        pyframe.media_kind("file.txt")
