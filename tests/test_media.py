import numpy as np
import pytest
from PIL import Image

from pyframe.errors import MediaDecodeError, UnsupportedMediaError
from pyframe.media import (
    ANIMATION_EXTS,
    IMAGE_EXTS,
    MediaKind,
    iter_frames,
    iter_frames_from_bytes,
    media_kind,
)


def _write_gif(path, fills=(10, 60, 250, 120, 30), size=32, duration=80):
    imgs = [Image.fromarray(np.full((size, size, 3), v, np.uint8)) for v in fills]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
    return str(path)


# The cv2.VideoCapture path had no coverage at all, so nothing checked that the OpenCV
# major version CI actually resolves can still decode the format the package is named
# after. `opencv-python-headless>=4.8` currently resolves to OpenCV 5.


def test_iter_frames_decodes_a_gif_from_disk(tmp_path):
    frames = list(iter_frames(_write_gif(tmp_path / "clip.gif")))

    assert [f.index for f in frames] == [0, 1, 2, 3, 4]
    assert frames[0].image.shape == (32, 32, 3)
    assert frames[0].image.dtype == np.uint8


def test_iter_frames_timestamps_start_at_zero_and_increase(tmp_path):
    stamps = [f.timestamp for f in iter_frames(_write_gif(tmp_path / "clip.gif"))]

    assert stamps[0] == 0.0
    assert all(later > earlier for earlier, later in zip(stamps, stamps[1:]))


def test_iter_frames_scores_motion_against_the_previous_frame(tmp_path):
    frames = list(iter_frames(_write_gif(tmp_path / "clip.gif")))

    assert frames[0].motion_score == 0.0  # nothing to diff against
    assert any(f.motion_score > 0 for f in frames[1:])


def test_iter_frames_reads_a_still_image(tmp_path):
    path = tmp_path / "still.png"
    Image.fromarray(np.full((16, 24, 3), 200, np.uint8)).save(path)

    frames = list(iter_frames(str(path)))

    assert len(frames) == 1
    assert (frames[0].index, frames[0].timestamp, frames[0].motion_score) == (0, 0.0, 0.0)
    assert frames[0].image.shape == (16, 24, 3)


def test_iter_frames_on_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_frames(str(tmp_path / "nope.gif")))


def test_iter_frames_on_a_gif_that_is_not_a_gif(tmp_path):
    path = tmp_path / "fake.gif"
    path.write_text("this is not a gif")

    with pytest.raises(MediaDecodeError):
        list(iter_frames(str(path)))


def test_iter_frames_from_bytes_on_garbage():
    with pytest.raises(MediaDecodeError):
        list(iter_frames_from_bytes(b"not an image"))


@pytest.mark.parametrize("ext", sorted(IMAGE_EXTS))
def test_media_kind_maps_every_image_extension(ext):
    assert media_kind(f"x{ext}") is MediaKind.IMAGE


@pytest.mark.parametrize("ext", sorted(ANIMATION_EXTS))
def test_media_kind_maps_every_animation_extension(ext):
    assert media_kind(f"x{ext}") is MediaKind.ANIMATION


def test_media_kind_ignores_case():
    assert media_kind("X.GIF") is MediaKind.ANIMATION


@pytest.mark.parametrize("name", ["x.txt", "x.pdf", "noextension"])
def test_media_kind_rejects_unknown_types(name):
    with pytest.raises(UnsupportedMediaError):
        media_kind(name)
