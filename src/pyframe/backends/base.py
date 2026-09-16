from __future__ import annotations

import abc
from collections.abc import Iterable

from ..media import Frame
from ..results import Label, Verdict


class Backend(abc.ABC):
    name: str = "backend"
    cost_per_image: float = 0.0
    default_min_confidence: float = 0.8

    @abc.abstractmethod
    def _score(self, image) -> tuple[float, list[Label], object]:
        """Score a PIL RGB image -> (nsfw_score in [0,1], labels, raw payload)."""

    def classify_image(
        self,
        image,
        *,
        min_confidence: float = 0.8,
        index: int = -1,
        timestamp: float = 0.0,
    ) -> Verdict:
        try:
            score, labels, raw = self._score(image)
        except Exception as exc:
            return Verdict(
                score=0.0,
                is_nsfw=False,
                backend=self.name,
                frame_index=index,
                timestamp=timestamp,
                error=str(exc),
            )
        return Verdict(
            score=score,
            is_nsfw=score >= min_confidence,
            labels=tuple(labels),
            backend=self.name,
            frame_index=index,
            timestamp=timestamp,
            raw=raw,
        )

    def classify(self, frame: Frame, *, min_confidence: float = 0.8) -> Verdict:
        return self.classify_image(
            frame.to_pil(),
            min_confidence=min_confidence,
            index=frame.index,
            timestamp=frame.timestamp,
        )

    def classify_batch(self, frames: Iterable[Frame], *, min_confidence: float = 0.8) -> list[Verdict]:
        """Score frames in order.

        `frames` may be a lazy iterable that decodes as it is consumed, so an override
        should iterate it once and not index, len() or retain it. Materialising it puts
        the whole sample in memory, which on a long video is the thing this avoids.
        """
        return [self.classify(f, min_confidence=min_confidence) for f in frames]
