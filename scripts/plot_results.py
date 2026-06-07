#!/usr/bin/env python3
"""Generate PyFrame performance charts from a bench_results.jsonl into media/.

    pip install matplotlib
    python scripts/plot_results.py [bench_results.jsonl]

Produces media/perf_stages.png (per-stage median timing) and
media/perf_latency.png (per-GIF latency percentiles).
"""

import json
import statistics
import sys
from pathlib import Path

try:
    import matplotlib
except ImportError:  # plot-only dep, not a runtime requirement of pyframe
    raise SystemExit("plot_results.py needs matplotlib.\n    pip install matplotlib")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "bench_results.jsonl"
    recs = load(src)
    n = len(recs)
    out = Path("media")
    out.mkdir(exist_ok=True)

    # Chart 1: per-stage median time per GIF (log x, since inference dwarfs the rest)
    stages = ["t_decode_sample", "t_preprocess", "t_inference", "t_gate"]
    labels = ["decode + sample", "preprocess", "inference (ViT)", "gate"]
    vals = [statistics.median(r[s] for r in recs) * 1000 for s in stages]
    colors = ["#6c8ebf", "#bdbdbd", "#d6604d", "#bdbdbd"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bars = ax.barh(labels, vals, color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("median time per GIF (ms, log scale)")
    ax.set_title(f"PyFrame per-stage timing  (n={n} GIFs, single worker, CPU)")
    ax.invert_yaxis()
    for b, v in zip(bars, vals):
        ax.text(v * 1.05, b.get_y() + b.get_height() / 2, f"{v:.1f} ms", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "perf_stages.png", dpi=130)
    plt.close(fig)

    # Chart 2: per-GIF latency percentiles
    lat = np.array([r["latency_ms"] for r in recs])
    pcts = [50, 90, 95, 99]
    pv = [float(np.percentile(lat, p)) for p in pcts]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar([f"p{p}" for p in pcts], pv, color="#6c8ebf")
    ax.set_ylabel("per-GIF latency (ms)")
    ax.set_title(f"PyFrame latency percentiles  (n={n} GIFs, single worker, CPU)")
    for i, v in enumerate(pv):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "perf_latency.png", dpi=130)
    plt.close(fig)

    print(f"wrote media/perf_stages.png and media/perf_latency.png from {n} records")
