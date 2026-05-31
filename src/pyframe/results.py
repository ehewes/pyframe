from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    CLEAN = "clean"
    UNCERTAIN = "uncertain"
    NSFW = "nsfw"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_score(
        cls,
        score: float,
        min_confidence: float,
        uncertain_threshold: float,
        *,
        errored: bool = False,
    ) -> "Severity":
        if errored:
            return cls.ERROR
        if score >= min_confidence:
            return cls.NSFW
        if score >= uncertain_threshold:
            return cls.UNCERTAIN
        return cls.CLEAN


@dataclass(frozen=True)
class Label:
    name: str
    confidence: float
    taxonomy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "confidence": self.confidence}
        if self.taxonomy:
            d["taxonomy"] = self.taxonomy
        return d


@dataclass(frozen=True)
class Verdict:
    score: float
    is_nsfw: bool
    labels: tuple[Label, ...] = ()
    backend: str = ""
    frame_index: int = -1
    timestamp: float = 0.0
    raw: Any = field(default=None, repr=False)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "score": round(self.score, 4),
            "is_nsfw": self.is_nsfw,
            "backend": self.backend,
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "labels": [label.to_dict() for label in self.labels],
        }
        if self.error:
            d["error"] = self.error
        return d


@dataclass(frozen=True)
class ScanResult:
    source: str
    media_kind: str
    verdict: Severity
    is_nsfw: bool
    max_score: float
    frames: tuple[Verdict, ...]
    worst_frame: Verdict | None
    backends_used: tuple[str, ...]
    frames_total: int
    frames_screened: int
    frames_classified: int
    cost_usd: float
    prescreen_used: bool = False
    escalated: bool | None = None
    windows: int = 0
    elapsed_s: float | None = None

    @property
    def flagged_frames(self) -> list[Verdict]:
        return [v for v in self.frames if v.is_nsfw]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "media_kind": self.media_kind,
            "verdict": self.verdict.value,
            "is_nsfw": self.is_nsfw,
            "max_score": round(self.max_score, 4),
            "worst_frame": self.worst_frame.to_dict() if self.worst_frame else None,
            "frames": [v.to_dict() for v in self.frames],
            "backends_used": list(self.backends_used),
            "frames_total": self.frames_total,
            "frames_screened": self.frames_screened,
            "frames_classified": self.frames_classified,
            "cost_usd": round(self.cost_usd, 6),
            "prescreen_used": self.prescreen_used,
            "escalated": self.escalated,
            "windows": self.windows,
            "elapsed_s": round(self.elapsed_s, 3) if self.elapsed_s is not None else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
