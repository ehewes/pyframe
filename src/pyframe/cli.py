from __future__ import annotations

import argparse
import sys

from .errors import (
    BackendUnavailableError,
    MediaDecodeError,
    PyFrameError,
    UnsupportedMediaError,
)
from .results import ScanResult, Severity

_ICON = {
    Severity.CLEAN: "[clean]",
    Severity.UNCERTAIN: "[?]",
    Severity.NSFW: "[NSFW]",
    Severity.ERROR: "[error]",
}


def _print_human(result: ScanResult) -> None:
    head = f"{result.source}  {_ICON[result.verdict]}  score {result.max_score:.2f}"
    if result.prescreen_used:
        head += f"  (screened {result.frames_screened}, classified {result.frames_classified}"
        head += ", escalated)" if result.escalated else ", short-circuit clean)"
    print(head)

    for frame in result.flagged_frames:
        names = ", ".join(label.name for label in frame.labels) or "flagged"
        print(f"    t={frame.timestamp:.2f}s  {frame.score:.2f}  {names}")

    if result.cost_usd:
        print(f"    cost: ${result.cost_usd:.4f}  ({', '.join(result.backends_used)})")


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="pyframe",
        description="Moderate GIFs, videos, and images for NSFW content.",
    )
    parser.add_argument("paths", nargs="+", help="file(s) to moderate")
    parser.add_argument("--version", action="version", version=f"pyframe {__version__}")
    parser.add_argument("--backend", default="auto", help="auto, local, aws, or local:<model-id>")
    parser.add_argument("--model", default=None, help="HuggingFace model id (local backend)")
    parser.add_argument("--region", default="us-east-1", help="AWS region (aws backend)")
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=None, help="NSFW threshold (0-1); default is the backend's")
    parser.add_argument("--sampler", choices=("motion", "dense"), default="motion")
    parser.add_argument("--use-merged", action="store_true", help="merge frames into a grid before classifying")
    parser.add_argument("--frames-per-batch", type=int, default=2)
    parser.add_argument("--prescreen", action="store_true", help="enable the two-stage cascade")
    parser.add_argument("--escalate-threshold", type=float, default=0.15, help="cascade gate (low = recall-safe)")
    parser.add_argument("--max-escalations", type=int, default=2, help="max precise (AWS) calls per file")
    parser.add_argument("--screen-fps", type=float, default=2.0, help="soft-screen sample rate")
    parser.add_argument("--save-frames", default=None, metavar="DIR", help="write the classified frames to DIR")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--fail-on", choices=("nsfw", "uncertain", "never"), default="nsfw")
    return parser


def main() -> int:
    from .pipe import scan

    args = build_parser().parse_args()
    rc = 0
    for path in args.paths:
        try:
            result = scan(
                path,
                backend=args.backend,
                model=args.model,
                region=args.region,
                max_frames=args.max_frames,
                min_confidence=args.min_confidence,
                sampler=args.sampler,
                use_merged=args.use_merged,
                frames_per_batch=args.frames_per_batch,
                prescreen=args.prescreen,
                escalate_threshold=args.escalate_threshold,
                max_escalations=args.max_escalations,
                screen_fps=args.screen_fps,
                save_frames=args.save_frames,
            )
        except BackendUnavailableError as exc:
            print(exc, file=sys.stderr)
            return 3
        except (UnsupportedMediaError, MediaDecodeError, FileNotFoundError, PyFrameError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            rc = max(rc, 2)
            continue

        if args.json:
            print(result.to_json())
        else:
            _print_human(result)

        if args.fail_on == "nsfw" and result.is_nsfw:
            rc = max(rc, 1)
        elif args.fail_on == "uncertain" and result.verdict in (Severity.NSFW, Severity.UNCERTAIN):
            rc = max(rc, 1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
