import numpy as np

from pyframe.media import Frame, FrameMeta
from pyframe.sampling import (
    DenseUniformSampler,
    MotionBucketSampler,
    SuspicionSampler,
    group_flagged_into_windows,
)


def _frames(motions, fps=10.0):
    return [
        Frame(index=i, timestamp=i / fps, image=np.zeros((4, 4, 3), np.uint8), motion_score=m)
        for i, m in enumerate(motions)
    ]


def test_motion_bucket_keeps_highest_per_bucket():
    frames = _frames([1, 9, 2, 8, 3, 7])
    chosen = MotionBucketSampler().select(frames, budget=3)
    assert [f.index for f in chosen] == [1, 3, 5]


def test_motion_bucket_returns_all_when_under_budget():
    frames = _frames([1, 2])
    assert MotionBucketSampler().select(frames, budget=10) == frames


def test_dense_sampler_respects_target_fps():
    frames = _frames([0] * 20)  # 20 frames over ~1.9s -> ~10 fps source
    selected = DenseUniformSampler(target_fps=2.0).select(frames)
    assert [f.index for f in selected] == [0, 5, 10, 15, 19]  # stride 5, plus the tail


def test_dense_sampler_always_covers_the_tail():
    # frames[::stride] stops at 15 here, leaving 16-19 outside the recall floor. An NSFW
    # event that runs to the end of the clip would have no sampled frame to catch it.
    frames = _frames([0] * 20)
    selected = DenseUniformSampler(target_fps=2.0).select(frames)
    assert selected[-1].index == 19


def test_dense_sampler_does_not_duplicate_an_aligned_tail():
    frames = _frames([0] * 21)  # 21 frames, stride 5 -> 0,5,10,15,20 already ends on 20
    selected = DenseUniformSampler(target_fps=2.0).select(frames)
    assert [f.index for f in selected] == [0, 5, 10, 15, 20]


def test_dense_sampler_keeps_all_when_no_duration():
    frames = [Frame(index=i, timestamp=0.0, image=np.zeros((4, 4, 3), np.uint8)) for i in range(5)]
    assert len(DenseUniformSampler(target_fps=1.0).select(frames)) == 5


def test_suspicion_sampler_picks_highest_scores_in_index_order():
    frames = _frames([0, 0, 0, 0, 0])
    scores = {0: 0.1, 1: 0.9, 2: 0.2, 3: 0.8, 4: 0.05}
    chosen = SuspicionSampler().select(frames, budget=2, scores=scores)
    assert [f.index for f in chosen] == [1, 3]


def test_group_windows_merges_within_gap_and_pads():
    windows = group_flagged_into_windows([5, 6, 20], n_frames=30, gap=8, pad=2)
    assert windows == [(3, 9), (18, 23)]


def test_group_windows_merges_overlap_after_padding():
    windows = group_flagged_into_windows([5, 15], n_frames=30, gap=3, pad=6)
    assert windows == [(0, 22)]


def test_group_windows_empty():
    assert group_flagged_into_windows([], n_frames=10, gap=2, pad=1) == []


def _metas(motions, fps=10.0):
    return [
        FrameMeta(index=i, timestamp=i / fps, motion_score=m) for i, m in enumerate(motions)
    ]


def test_samplers_pick_the_same_frames_from_metadata_as_from_frames():
    # Sampling runs on metadata now and pixels are fetched afterwards, so the two must
    # agree exactly or the scan would moderate a different frame than the one chosen.
    motions = [(i * 37) % 100 for i in range(40)]
    frames, metas = _frames(motions), _metas(motions)
    scores = {i: (i % 7) / 10.0 for i in range(40)}

    assert (
        [f.index for f in DenseUniformSampler(2.0).select(frames)]
        == [m.index for m in DenseUniformSampler(2.0).select(metas)]
    )
    assert (
        [f.index for f in MotionBucketSampler().select(frames, 6)]
        == [m.index for m in MotionBucketSampler().select(metas, 6)]
    )
    assert (
        [f.index for f in SuspicionSampler().select(frames, 5, scores)]
        == [m.index for m in SuspicionSampler().select(metas, 5, scores)]
    )
