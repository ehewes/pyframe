from __future__ import annotations

import time

from .backends import Backend, load_backend
from .config import Config
from .errors import MediaDecodeError
from .image_utils import merge_to_grid
from .media import (
    MediaKind,
    iter_frame_meta,
    iter_frame_meta_from_bytes,
    iter_frames_at,
    iter_frames_from_bytes_at,
    media_kind,
)
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
        # Validate before load_backend: constructing a backend pulls ~0.5 GB of weights,
        # which is a lot of work to do before rejecting the config.
        #
        # A budget below 1 doesn't disable escalation, it removes the cap: the suspicion
        # sampler treats a non-positive budget as "keep everything", so every flagged
        # frame would be escalated. Reject it rather than guess which of "never escalate"
        # or "escalate once" was meant.
        if config.prescreen.enabled and config.prescreen.max_escalations < 1:
            raise ValueError(
                f"max_escalations must be >= 1, got {config.prescreen.max_escalations}"
            )
        # Same trap on the single-pass side: the samplers read a non-positive budget as
        # "keep everything", which would materialise the whole clip.
        if config.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {config.max_frames}")

        precise = load_backend(config.backend, model=config.model, region=config.region)
        screen = None
        if config.prescreen.enabled:
            screen = load_backend(config.screen_backend, model=config.screen_model)
        return cls(precise, screen=screen, config=config)

    def scan(self, source) -> ScanResult:
        start = time.perf_counter()
        kind = media_kind(source)
        # Pass one holds the whole timeline as metadata; pixels are fetched later, and
        # only for the frames sampling actually selects.
        metas = list(iter_frame_meta(source))

        def fetch(selected):
            return iter_frames_at(source, selected)

        return self._scan_frames(str(source), kind, metas, fetch, start)

    def scan_bytes(self, data, *, label: str = "<bytes>") -> ScanResult:
        """Scan a GIF/image decoded from memory, no disk touched."""
        start = time.perf_counter()
        metas = list(iter_frame_meta_from_bytes(data))
        kind = MediaKind.ANIMATION if len(metas) > 1 else MediaKind.IMAGE

        def fetch(selected):
            return iter_frames_from_bytes_at(data, selected)

        return self._scan_frames(label, kind, metas, fetch, start)

    def _scan_frames(self, source, kind, metas, fetch, start) -> ScanResult:
        # Nothing to look at is not the same as nothing to find: aggregating zero frames
        # would report a confident "clean" for media no backend ever saw.
        if not metas:
            raise MediaDecodeError(f"decoded 0 frames from {source}")

        if kind is MediaKind.IMAGE:
            verdicts = self.precise.classify_batch(fetch(metas), min_confidence=self.min_confidence)
            return self._aggregate(source, kind, verdicts, [], len(metas), start)

        if self.config.prescreen.enabled and self.screen is not None:
            return self._cascade(source, kind, metas, fetch, start)
        return self._single_pass(source, kind, metas, fetch, start)

    def _single_pass(self, source, kind, metas, fetch, start) -> ScanResult:
        cfg = self.config
        if cfg.sampler == "dense":
            selected = DenseUniformSampler(cfg.prescreen.screen_fps).select(metas)
            if len(selected) > cfg.max_frames:
                selected = MotionBucketSampler().select(selected, cfg.max_frames)
        else:
            selected = self._motion_select_with_floor(metas)

        # The only materialisation on this path, and every branch above caps `selected`
        # at max_frames.
        frames = list(fetch(selected))
        if cfg.use_merged:
            verdicts = self._classify_merged(frames)
        else:
            verdicts = self.precise.classify_batch(frames, min_confidence=self.min_confidence)
        return self._aggregate(source, kind, verdicts, [], len(metas), start)

    def _motion_select_with_floor(self, metas):
        # Recall floor for the default (motion) sampler. The uniform-by-time sample at
        # screen_fps bounds the sampling stride, so no NSFW event longer than that stride
        # can fall entirely between selected frames. Motion is content-blind (it can keep
        # a moving SFW frame over a static NSFW one in the same region), so it only ever
        # spends the *spare* budget, never replaces the time-coverage floor.
        # cf. Ding, Sener, and Yao, arXiv:2210.10352 (temporal coverage as a prior, and
        # the decoupling of motion from static semantic content).
        cfg = self.config
        floor = DenseUniformSampler(cfg.prescreen.screen_fps).select(metas)
        if len(floor) >= cfg.max_frames:
            # The floor already fills the budget; motion only decides what to drop,
            # exactly as the `dense` path trims its own uniform sample.
            return MotionBucketSampler().select(floor, cfg.max_frames)
        # Spare budget: keep the whole time-coverage floor, then fill the remainder with
        # the highest-motion frames the floor did not already include.
        have = {f.index for f in floor}
        extra = sorted(
            (m for m in metas if m.index not in have),
            key=lambda f: f.motion_score,
            reverse=True,
        )
        selected = floor + extra[: cfg.max_frames - len(floor)]
        return sorted(selected, key=lambda f: f.index)

    def _cascade(self, source, kind, metas, fetch, start) -> ScanResult:
        cfg = self.config
        pc = cfg.prescreen

        screen_metas = DenseUniformSampler(pc.screen_fps).select(metas)
        # The screen set is screen_fps x duration, not max_frames, so on a long clip it
        # is most of the timeline. classify_batch only iterates, so handing it the lazy
        # fetch keeps one decoded frame alive at a time instead of all of them.
        screen_verdicts = self.screen.classify_batch(
            fetch(screen_metas), min_confidence=pc.escalate_threshold
        )
        scores = {v.frame_index: v.score for v in screen_verdicts}

        flagged = [
            v.frame_index
            for v in screen_verdicts
            if v.score >= pc.escalate_threshold or (v.error and pc.fail_open)
        ]
        if not flagged:
            return self._aggregate(
                source, kind, [], screen_verdicts, len(metas), start, escalated=False, windows=0
            )

        # Keep the most-suspicious flagged frames, capped so we make at most
        # max_escalations merged calls (each grid holds frames_per_batch frames).
        per_batch = max(1, cfg.frames_per_batch)
        frame_budget = pc.max_escalations * per_batch
        flagged_set = set(flagged)
        flagged_metas = [m for m in metas if m.index in flagged_set]
        selected = SuspicionSampler().select(flagged_metas, frame_budget, scores)
        # Always fill at least one full grid (send both even if only one frame flagged).
        selected = self._ensure_min_frames(selected, metas, scores, per_batch)

        # Send the top suspicious frames to the precise backend as merged grids. A third
        # decode, and only when something flagged: clean media stops after two.
        precise = self._classify_merged(list(fetch(selected)))

        windows = group_flagged_into_windows(flagged, len(metas), pc.group_gap, pc.window_pad)
        return self._aggregate(
            source, kind, precise, screen_verdicts, len(metas), start,
            escalated=True, windows=len(windows),
        )

    def _ensure_min_frames(self, selected, metas, scores, minimum):
        if len(selected) >= minimum:
            return selected
        have = {f.index for f in selected}
        # Same key as SuspicionSampler: screen score first, motion only as a tiebreak.
        # These are different units -- scores are 0..1, motion is a pixel-diff sum up to
        # ~1e6 -- so one flat key would rank any moving frame above a screened frame that
        # scored 0.99.
        extra = sorted(
            (m for m in metas if m.index not in have),
            key=lambda f: (scores.get(f.index, -1.0), f.motion_score),
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

    def _aggregate(
        self, source, kind, classified, screen_verdicts, frames_total, start,
        *, escalated=None, windows=0,
    ) -> ScanResult:
        cfg = self.config
        primary = classified if classified else list(screen_verdicts)
        worst = max(primary, key=lambda v: v.score) if primary else None
        max_score = worst.score if worst else 0.0

        errored = bool(primary) and all(v.error for v in primary)
        severity = Severity.from_score(max_score, self.min_confidence, cfg.uncertain_threshold, errored=errored)
        # Derive is_nsfw from the severity rather than computing it separately, so the
        # two can't disagree. They used to: a short-circuited cascade scores max_score
        # off the screen verdicts, which no classified frame ever backed, and the result
        # could read verdict=nsfw with is_nsfw=False.
        is_nsfw = severity is Severity.NSFW

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
