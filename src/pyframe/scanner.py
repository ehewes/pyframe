from __future__ import annotations

import os
import time

import cv2

from .backends import Backend, load_backend
from .config import Config
from .image_utils import merge_to_grid
from .media import MediaKind, iter_frames, media_kind
from .results import ScanResult, Severity, Verdict
from .sampling import (
    DenseUniformSampler,
    MotionBucketSampler,
    SuspicionSampler,
    group_flagged_into_windows,
)


class Scanner:
    def __init__(self, precise: Backend, *, screen: Backend | None = None, config: Config | None = None):
        self.precise = precise
        self.screen = screen
        self.config = config or Config()
        self.min_confidence = (
            self.config.min_confidence
            if self.config.min_confidence is not None
            else precise.default_min_confidence
        )

    @classmethod
    def from_config(cls, config: Config) -> "Scanner":
        precise = load_backend(config.backend, model=config.model, region=config.region)
        screen = None
        if config.prescreen.enabled:
            screen = load_backend(config.screen_backend, model=config.screen_model)
        return cls(precise, screen=screen, config=config)

    def scan(self, source) -> ScanResult:
        start = time.perf_counter()
        kind = media_kind(source)
        frames = list(iter_frames(source))

        if kind is MediaKind.IMAGE:
            verdicts = self.precise.classify_batch(frames, min_confidence=self.min_confidence)
            self._save(verdicts, frames, source)
            return self._aggregate(source, kind, verdicts, [], len(frames), start)

        if not frames:
            return self._aggregate(source, kind, [], [], 0, start)

        if self.config.prescreen.enabled and self.screen is not None:
            return self._cascade(source, kind, frames, start)
        return self._single_pass(source, kind, frames, start)

    def _single_pass(self, source, kind, frames, start) -> ScanResult:
        cfg = self.config
        if cfg.sampler == "dense":
            selected = DenseUniformSampler(cfg.prescreen.screen_fps).select(frames)
            if len(selected) > cfg.max_frames:
                selected = MotionBucketSampler().select(selected, cfg.max_frames)
        else:
            selected = MotionBucketSampler().select(frames, cfg.max_frames)

        if cfg.use_merged:
            verdicts = self._classify_merged(selected)
        else:
            verdicts = self.precise.classify_batch(selected, min_confidence=self.min_confidence)
        self._save(verdicts, selected, source)
        return self._aggregate(source, kind, verdicts, [], len(frames), start)

    def _cascade(self, source, kind, frames, start) -> ScanResult:
        cfg = self.config
        pc = cfg.prescreen

        screen_frames = DenseUniformSampler(pc.screen_fps).select(frames)
        screen_verdicts = self.screen.classify_batch(screen_frames, min_confidence=pc.escalate_threshold)
        scores = {v.frame_index: v.score for v in screen_verdicts}

        flagged = [
            v.frame_index
            for v in screen_verdicts
            if v.score >= pc.escalate_threshold or (v.error and pc.fail_open)
        ]
        if not flagged:
            return self._aggregate(
                source, kind, [], screen_verdicts, len(frames), start, escalated=False, windows=0
            )

        # Keep the most-suspicious flagged frames, capped so we make at most
        # max_escalations merged calls (each grid holds frames_per_batch frames).
        per_batch = max(1, cfg.frames_per_batch)
        frame_budget = pc.max_escalations * per_batch
        flagged_set = set(flagged)
        flagged_frames = [f for f in frames if f.index in flagged_set]
        selected = SuspicionSampler().select(flagged_frames, frame_budget, scores)
        # Always fill at least one full grid (send both even if only one frame flagged).
        selected = self._ensure_min_frames(selected, frames, scores, per_batch)

        # Send the top suspicious frames to the precise backend as merged grids.
        precise = self._classify_merged(selected)
        self._save(precise, selected, source)

        windows = group_flagged_into_windows(flagged, len(frames), pc.group_gap, pc.window_pad)
        return self._aggregate(
            source, kind, precise, screen_verdicts, len(frames), start,
            escalated=True, windows=len(windows),
        )

    def _ensure_min_frames(self, selected, frames, scores, minimum):
        if len(selected) >= minimum:
            return selected
        have = {f.index for f in selected}
        extra = sorted(
            (f for f in frames if f.index not in have),
            key=lambda f: scores.get(f.index, f.motion_score),
            reverse=True,
        )
        if not extra:
            return selected
        selected = selected + extra[: minimum - len(selected)]
        return sorted(selected, key=lambda f: f.index)

    def _classify_merged(self, frames) -> list[Verdict]:
        cfg = self.config
        per_batch = max(1, cfg.frames_per_batch)
        verdicts: list[Verdict] = []
        for i in range(0, len(frames), per_batch):
            batch = frames[i : i + per_batch]
            grid = merge_to_grid([f.to_pil() for f in batch])
            verdicts.append(
                self.precise.classify_image(
                    grid,
                    min_confidence=self.min_confidence,
                    index=i // per_batch,
                    timestamp=batch[0].timestamp,
                )
            )
        return verdicts

    def _save(self, verdicts, frames, source) -> None:
        out_dir = self.config.save_frames
        if not out_dir:
            return
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(str(source)))[0]
        for rank, frame in enumerate(frames):
            name = f"{stem}_{rank:02d}_frame{frame.index:04d}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), frame.image)

    def _aggregate(
        self, source, kind, classified, screen_verdicts, frames_total, start,
        *, escalated=None, windows=0,
    ) -> ScanResult:
        cfg = self.config
        primary = classified if classified else list(screen_verdicts)
        worst = max(primary, key=lambda v: v.score) if primary else None
        max_score = worst.score if worst else 0.0

        is_nsfw = any(v.is_nsfw for v in classified)
        errored = bool(primary) and all(v.error for v in primary)
        severity = Severity.from_score(max_score, self.min_confidence, cfg.uncertain_threshold, errored=errored)

        cost = len(classified) * self.precise.cost_per_image
        if self.screen is not None:
            cost += len(screen_verdicts) * self.screen.cost_per_image

        all_verdicts = list(screen_verdicts) + list(classified)
        backends_used = tuple(dict.fromkeys(v.backend for v in all_verdicts if v.backend))

        return ScanResult(
            source=str(source),
            media_kind=kind.value,
            verdict=severity,
            is_nsfw=is_nsfw,
            max_score=max_score,
            frames=tuple(primary),
            worst_frame=worst,
            backends_used=backends_used,
            frames_total=frames_total,
            frames_screened=len(screen_verdicts),
            frames_classified=len(classified),
            cost_usd=cost,
            prescreen_used=cfg.prescreen.enabled,
            escalated=escalated,
            windows=windows,
            elapsed_s=time.perf_counter() - start,
        )
