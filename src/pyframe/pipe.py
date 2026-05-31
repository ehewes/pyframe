from __future__ import annotations

from .config import Config, PrescreenConfig
from .results import ScanResult
from .scanner import Scanner


class Pipe:
    def __init__(
        self,
        input_path,
        *,
        backend="auto",
        model=None,
        region="us-east-1",
        max_frames=10,
        min_confidence=None,
        uncertain_threshold=0.3,
        sampler="motion",
        use_merged=False,
        frames_per_batch=2,
        prescreen=False,
        screen_backend="local",
        screen_model=None,
        escalate_threshold=0.15,
        screen_fps=2.0,
        group_gap=8,
        window_pad=4,
        max_escalations=2,
        fail_open=True,
        save_frames=None,
    ):
        self.input_path = input_path
        self.config = Config(
            backend=backend,
            model=model,
            region=region,
            max_frames=max_frames,
            min_confidence=min_confidence,
            uncertain_threshold=uncertain_threshold,
            sampler=sampler,
            use_merged=use_merged,
            frames_per_batch=frames_per_batch,
            screen_backend=screen_backend,
            screen_model=screen_model,
            save_frames=save_frames,
            prescreen=PrescreenConfig(
                enabled=prescreen,
                screen_fps=screen_fps,
                escalate_threshold=escalate_threshold,
                group_gap=group_gap,
                window_pad=window_pad,
                max_escalations=max_escalations,
                fail_open=fail_open,
            ),
        )

    def run(self) -> ScanResult:
        return Scanner.from_config(self.config).scan(self.input_path)


def scan(source, **kwargs) -> ScanResult:
    return Pipe(source, **kwargs).run()
