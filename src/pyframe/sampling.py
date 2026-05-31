from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .media import Frame


class MotionBucketSampler:
    # Highest-motion frame per equal-width bucket. Lossy/content-blind: cost lever
    # only, never the cascade gate (motion is uncorrelated with NSFW content).
    def select(self, frames: Sequence[Frame], budget: int) -> list[Frame]:
        n = len(frames)
        if n == 0:
            return []
        if budget <= 0 or n <= budget:
            return list(frames)

        chunk = n / budget
        chosen: list[Frame] = []
        for i in range(budget):
            start = int(i * chunk)
            end = n if i == budget - 1 else int((i + 1) * chunk)
            bucket = frames[start:end]
            if bucket:
                chosen.append(max(bucket, key=lambda f: f.motion_score))
        return chosen


class DenseUniformSampler:
    # Uniform sub-sampling to a target effective fps. Sampling by time cadence
    # (not a fixed count) bounds the NSFW event duration that can slip between
    # samples; unknown source rate falls back to every frame (recall-safe).
    def __init__(self, target_fps: float = 2.0):
        self.target_fps = max(target_fps, 0.01)

    def select(self, frames: Sequence[Frame]) -> list[Frame]:
        n = len(frames)
        if n <= 1:
            return list(frames)

        duration = frames[-1].timestamp - frames[0].timestamp
        if duration <= 0:
            return list(frames)

        source_fps = (n - 1) / duration
        stride = max(1, round(source_fps / self.target_fps))
        return list(frames[::stride])


class SuspicionSampler:
    # Keep the most-suspicious frames in a window (screen score, then motion).
    def select(
        self,
        frames: Sequence[Frame],
        budget: int,
        scores: Mapping[int, float] | None = None,
    ) -> list[Frame]:
        n = len(frames)
        if n == 0:
            return []
        if budget <= 0 or n <= budget:
            return list(frames)

        scores = scores or {}
        ranked = sorted(
            frames,
            key=lambda f: (scores.get(f.index, -1.0), f.motion_score),
            reverse=True,
        )
        chosen = ranked[:budget]
        return sorted(chosen, key=lambda f: f.index)


def group_flagged_into_windows(
    indices: Iterable[int],
    n_frames: int,
    gap: int,
    pad: int,
) -> list[tuple[int, int]]:
    # Flags within `gap` join one window, padded by `pad`, clamped to [0, n).
    ordered = sorted({i for i in indices if 0 <= i < n_frames})
    if not ordered:
        return []

    raw: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for i in ordered[1:]:
        if i - prev <= gap:
            prev = i
        else:
            raw.append((max(0, start - pad), min(n_frames, prev + pad + 1)))
            start = prev = i
    raw.append((max(0, start - pad), min(n_frames, prev + pad + 1)))

    # Merge overlaps created by padding so frames aren't scored twice.
    merged: list[tuple[int, int]] = []
    for window in raw:
        if merged and window[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], window[1]))
        else:
            merged.append(window)
    return merged
