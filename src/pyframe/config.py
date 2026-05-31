from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PrescreenConfig:
    enabled: bool = False
    screen_fps: float = 2.0
    escalate_threshold: float = 0.15
    group_gap: int = 8
    window_pad: int = 4
    max_escalations: int = 2  # hard cap on precise (e.g. AWS) calls per file
    fail_open: bool = True


@dataclass
class Config:
    backend: object = "auto"
    model: str | None = None
    region: str = "us-east-1"
    max_frames: int = 10
    min_confidence: float | None = None  # None -> backend's recall-safe default
    uncertain_threshold: float = 0.3
    sampler: str = "motion"
    use_merged: bool = False
    frames_per_batch: int = 2
    screen_backend: object = "local"
    screen_model: str | None = None
    save_frames: str | None = None
    prescreen: PrescreenConfig = field(default_factory=PrescreenConfig)
