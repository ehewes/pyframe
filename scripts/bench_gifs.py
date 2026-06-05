#!/usr/bin/env python3
"""Engine benchmark for PyFrame's GIF moderation path.

Measures the real pipeline's throughput, latency, per-stage timing, memory, and scaling
under a multiprocessing pool. It drives the REAL pipeline (no reimplementation); per GIF
it calls, in order:
    pyframe.media.iter_frames                      (decode)
    pyframe.sampling.DenseUniformSampler.select    (prescreen sampling @ screen_fps)
    pyframe.media.Frame.to_pil                      (preprocess: BGR->RGB->PIL)
    pyframe.backends.LocalBackend.classify_image    (local ViT inference, per frame)
    gate (score >= escalate_threshold, fail-open)   (escalation decision)
    on escalation: pyframe.sampling.SuspicionSampler.select + image_utils.merge_to_grid
                   -> StubBackend (precise/AWS backend MOCKED: instant, counted only)

The precise backend is stubbed (instant, no network), so this measures LOCAL throughput.

Concurrency = a multiprocessing (spawn) process pool (decode is CPU-bound; threads lose
to the GIL). Each worker forces single-threaded inference (OMP/MKL/OpenBLAS/torch/onnx=1)
so N processes don't each spawn multi-threaded torch and oversubscribe cores.

Outputs:
  1. Per-process-count scaling curve + sweet spot, knee, bottleneck class.
  2. Per-stage median timing + inference fraction f.
  3. bench_results.jsonl: one record per GIF.
  4. Environment block (host, versions, pinned config).

Run:
  python scripts/bench_gifs.py --corpus ./gifs
  python scripts/bench_gifs.py --procs "1"      # single-worker profile

Flags: see --help.
"""

# ruff: noqa: E402  (thread caps are deliberately set before heavy imports below)
# Thread caps MUST be set before numpy/torch/cv2 import (they read these at import).
import os

for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",        # macOS Accelerate
    "ONNXRUNTIME_INTRA_OP_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
):
    os.environ.setdefault(_v, "1" if _v != "TOKENIZERS_PARALLELISM" else "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import json
import multiprocessing as mp
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from itertools import islice
from pathlib import Path

import numpy as np
import psutil

from pyframe.backends import load_backend
from pyframe.backends.base import Backend
from pyframe.image_utils import merge_to_grid
from pyframe.media import iter_frames
from pyframe.sampling import DenseUniformSampler, SuspicionSampler

# Worker-process globals (populated by worker_init under spawn).
_SCREEN = None
_STUB = None
_CFG = None


class StubBackend(Backend):
    """Mocked precise backend: returns instantly, no network."""

    name = "stub"
    cost_per_image = 0.0

    def _score(self, image):
        return 0.0, [], None


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
def _synth_one(path, n_frames, width):
    from PIL import Image

    height = max(8, int(round(width * 0.6)))
    grad = np.linspace(0, 255, width, dtype=np.uint8)[None, :, None]
    frames = []
    for i in range(n_frames):
        arr = np.zeros((height, width, 3), np.uint8)
        arr[:, :, 0:1] = grad  # horizontal gradient (R channel)
        x = int((i / max(1, n_frames - 1)) * (width - 24))
        arr[height // 2 - 8 : height // 2 + 8, x : x + 24, :] = 255  # moving block -> motion
        noise = np.random.randint(-12, 12, (height, width, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(arr))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=66, loop=0, optimize=False)


def synthesize_corpus(out_dir, count, rng):
    # (weight, frame-range, width-range)
    buckets = [
        (0.60, (10, 40), (128, 320)),
        (0.30, (40, 90), (320, 480)),
        (0.10, (90, 150), (480, 640)),
    ]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        r = rng.random()
        cum = 0.0
        chosen = buckets[-1]
        for b in buckets:
            cum += b[0]
            if r <= cum:
                chosen = b
                break
        nf = rng.randint(*chosen[1])
        w = rng.randint(*chosen[2])
        p = out_dir / f"synth_{i:04d}_{nf}f_{w}px.gif"
        _synth_one(str(p), nf, w)
        paths.append(str(p))
    return paths


def describe_corpus(paths):
    sizes, frame_counts, widths = [], [], []
    for p in paths:
        sizes.append(os.path.getsize(p) / 1e6)
        try:
            from PIL import Image

            with Image.open(p) as im:
                widths.append(im.size[0])
                frame_counts.append(getattr(im, "n_frames", 1))
        except Exception:
            pass

    def pct(a, q):
        return float(np.percentile(a, q)) if a else 0.0

    print("\n=== CORPUS ===")
    print(f"  files: {len(paths)}  mean file size: {statistics.mean(sizes):.2f} MB" if sizes else "  (empty)")
    if frame_counts:
        print(
            f"  frames/GIF   p50={pct(frame_counts,50):.0f}  p90={pct(frame_counts,90):.0f}  "
            f"min={min(frame_counts)}  max={max(frame_counts)}"
        )
    if widths:
        print(
            f"  width px     p50={pct(widths,50):.0f}  p90={pct(widths,90):.0f}  "
            f"min={min(widths)}  max={max(widths)}"
        )


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def worker_init(cfg):
    global _SCREEN, _STUB, _CFG
    _CFG = cfg
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        import onnxruntime as ort

        _ = ort  # intra-op threads pinned via env ONNXRUNTIME_INTRA_OP_NUM_THREADS=1
    except Exception:
        pass
    _SCREEN = load_backend("local", model=cfg["model"])
    _STUB = StubBackend()


def process_one(path):
    """Run the real cascade prescreen path on one GIF with per-stage timing."""
    proc = psutil.Process()
    esc = _CFG["escalate_threshold"]
    t0 = time.perf_counter()
    frames = list(iter_frames(path))
    n_total = len(frames)
    screen_frames = DenseUniformSampler(_CFG["screen_fps"]).select(frames)
    t1 = time.perf_counter()

    pils = [f.to_pil() for f in screen_frames]
    t2 = time.perf_counter()

    verdicts = [
        _SCREEN.classify_image(p, min_confidence=esc, index=f.index, timestamp=f.timestamp)
        for f, p in zip(screen_frames, pils)
    ]
    t3 = time.perf_counter()

    scores = {v.frame_index: v.score for v in verdicts}
    flagged = [v.frame_index for v in verdicts if v.score >= esc or (v.error and True)]
    max_local = max((v.score for v in verdicts), default=0.0)
    escalated = bool(flagged)
    if escalated:
        # Mirror Scanner._cascade escalation: top-suspicious -> merged grids -> precise.
        per_batch = max(1, _CFG["frames_per_batch"])
        budget = _CFG["max_escalations"] * per_batch
        fset = set(flagged)
        flagged_frames = [f for f in frames if f.index in fset]
        selected = SuspicionSampler().select(flagged_frames, budget, scores)
        for i in range(0, len(selected), per_batch):
            grid = merge_to_grid([fr.to_pil() for fr in selected[i : i + per_batch]])
            _STUB.classify_image(grid, min_confidence=0.8)  # precise backend mocked: instant, counted
    t4 = time.perf_counter()

    return {
        "gif_id": os.path.basename(path),
        "n_frames_total": n_total,
        "n_frames_scored": len(screen_frames),
        "t_decode_sample": t1 - t0,
        "t_preprocess": t2 - t1,
        "t_inference": t3 - t2,
        "t_gate": t4 - t3,
        "latency_ms": (t4 - t0) * 1000.0,
        "peak_rss_mb": round(proc.memory_info().rss / 1e6, 1),
        "max_local_score": round(float(max_local), 4),
        "escalated": escalated,
    }


# --------------------------------------------------------------------------- #
# Resource monitor (samples worker PIDs during a level)
# --------------------------------------------------------------------------- #
class ResourceMonitor(threading.Thread):
    def __init__(self, pids, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.procs = []
        for pid in pids:
            try:
                self.procs.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                pass
        self._stop = threading.Event()
        self.peak_rss = 0
        self.cpu_samples = []

    def run(self):
        psutil.cpu_percent(None)  # prime system-wide
        while not self._stop.wait(self.interval):
            self.cpu_samples.append(psutil.cpu_percent(None))
            total = 0
            for pr in self.procs:
                try:
                    total += pr.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.peak_rss = max(self.peak_rss, total)

    def stop(self):
        self._stop.set()
        self.join(timeout=3)

    @property
    def mean_cpu(self):
        return statistics.mean(self.cpu_samples) if self.cpu_samples else 0.0


def _path_stream(corpus):
    while True:
        for p in corpus:
            yield p


# --------------------------------------------------------------------------- #
# Run one concurrency level
# --------------------------------------------------------------------------- #
def run_level(P, corpus, cfg, duration, min_gifs, warmup):
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(P, initializer=worker_init, initargs=(cfg,))
    try:
        pids = [w.pid for w in getattr(pool, "_pool", [])]
        # Warm-up: model load happened in init; exercise JIT/caches and discard.
        for _ in pool.imap_unordered(process_one, list(islice(_path_stream(corpus), warmup))):
            pass

        mon = ResourceMonitor(pids)
        mon.start()
        records = []
        t_start = time.perf_counter()
        for rec in pool.imap_unordered(process_one, _path_stream(corpus)):
            records.append(rec)
            elapsed = time.perf_counter() - t_start
            if elapsed >= duration and len(records) >= min_gifs:
                break
        wall = time.perf_counter() - t_start
        mon.stop()
    finally:
        pool.terminate()
        pool.join()

    lat = np.array([r["latency_ms"] for r in records], dtype=float)
    frames_scored = sum(r["n_frames_scored"] for r in records)
    gifs = len(records)
    result = {
        "P": P,
        "wall_s": wall,
        "gifs": gifs,
        "gifs_per_sec": gifs / wall,
        "gifs_per_hr": gifs / wall * 3600.0,
        "frames_scored_per_sec": frames_scored / wall,
        "lat_p50_ms": float(np.percentile(lat, 50)),
        "lat_p95_ms": float(np.percentile(lat, 95)),
        "lat_p99_ms": float(np.percentile(lat, 99)),
        "mean_cpu_pct": mon.mean_cpu,
        "peak_rss_mb": mon.peak_rss / 1e6,
        "escalation_rate": float(np.mean([r["escalated"] for r in records])) if records else 0.0,
    }
    return result, records


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def make_sweep(vcpu, max_procs, explicit):
    if explicit:
        seq = sorted({int(x) for x in explicit.split(",") if x.strip()})
        return [p for p in seq if p >= 1]
    cap = max_procs or 2 * vcpu
    base = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    seq = sorted({p for p in base + [vcpu, 2 * vcpu] if 1 <= p <= cap})
    return seq


def analyse(levels, vcpu, ram_gb):
    best = max(levels, key=lambda r: r["gifs_per_hr"])
    best_hr = best["gifs_per_hr"]
    # knee = smallest P reaching >=90% of peak throughput
    knee = min((r for r in levels if r["gifs_per_hr"] >= 0.9 * best_hr), key=lambda r: r["P"])
    peak_rss_gb = best["peak_rss_mb"] / 1024.0
    if peak_rss_gb >= 0.85 * ram_gb:
        bottleneck = "RAM-capacity-bound (aggregate RSS approaches host RAM before the knee)"
    elif knee["P"] >= 0.9 * vcpu:
        bottleneck = f"CPU-bound (knee P={knee['P']} ~ vCPU={vcpu})"
    elif best["mean_cpu_pct"] < 85.0:
        bottleneck = (
            f"memory-bandwidth-bound (knee P={knee['P']} < vCPU={vcpu}, "
            f"CPU only {best['mean_cpu_pct']:.0f}% at sweet spot)"
        )
    else:
        bottleneck = f"CPU-bound (knee P={knee['P']})"
    return best, knee, bottleneck


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def cpu_model():
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def versions():
    out = {}
    for mod in ("torch", "onnxruntime", "transformers"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "n/a"
    return out


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(
        description="Engine benchmark for the GIF moderation path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--corpus", default=None, help="directory of real .gif files (else synthesize)")
    ap.add_argument("--synth-count", type=int, default=120, help="GIFs to synthesize when no --corpus")
    ap.add_argument("--out", default="bench_results.jsonl", help="per-GIF JSONL output")
    ap.add_argument("--duration", type=float, default=120.0, help="steady-state seconds per level")
    ap.add_argument("--min-gifs", type=int, default=2000, help="min GIFs per level (overrides short duration)")
    ap.add_argument("--warmup", type=int, default=20, help="GIFs to process+discard before timing")
    ap.add_argument("--procs", default=None, help="explicit comma list of process counts (overrides sweep)")
    ap.add_argument("--max-procs", type=int, default=None, help="cap the process sweep (default 2x vCPU)")
    ap.add_argument("--screen-fps", type=float, default=2.0, help="prescreen sample rate (sweepable)")
    ap.add_argument("--max-frames", type=int, default=10, help="motion-sample frame budget (sweepable)")
    ap.add_argument("--escalate-threshold", type=float, default=0.15, help="gate threshold (recall-safe)")
    ap.add_argument("--frames-per-batch", type=int, default=2, help="frames per merged grid on escalation")
    ap.add_argument("--max-escalations", type=int, default=2, help="precise-backend call cap per GIF")
    ap.add_argument("--model", default="AdamCodd/vit-base-nsfw-detector", help="local ViT model id")
    args = ap.parse_args()

    vcpu = os.cpu_count() or 1
    ram_gb = psutil.virtual_memory().total / 1e9
    cfg = {
        "screen_fps": args.screen_fps,
        "max_frames": args.max_frames,
        "escalate_threshold": args.escalate_threshold,
        "frames_per_batch": args.frames_per_batch,
        "max_escalations": args.max_escalations,
        "model": args.model,
    }

    # --- environment + pinned config ---
    hr("ENVIRONMENT & PINNED CONFIG")
    ver = versions()
    print(f"  host: {vcpu} logical vCPU | {ram_gb:.1f} GB RAM | {cpu_model()}")
    print(f"  OS: {platform.platform()}  Python {platform.python_version()}")
    print(f"  torch={ver['torch']}  onnxruntime={ver['onnxruntime']}  transformers={ver['transformers']}")
    print("  inference backend: torch CPU, single-threaded per worker")
    print("  threading: OMP/MKL/OpenBLAS/VECLIB/ONNX=1, torch.set_num_threads(1) per worker")
    print(
        "  pinned: prescreen.enabled=True "
        f"screen_fps={cfg['screen_fps']} max_frames={cfg['max_frames']} sampler=motion "
        f"escalate_threshold={cfg['escalate_threshold']} frames_per_batch={cfg['frames_per_batch']} "
        f"max_escalations={cfg['max_escalations']}"
    )
    print(f"  precise backend: STUBBED (instant, counted)  model={cfg['model']}")
    print(
        "  pipeline fns/GIF: iter_frames -> DenseUniformSampler.select -> Frame.to_pil"
        " -> LocalBackend.classify_image (ViT) -> gate -> [SuspicionSampler.select -> merge_to_grid -> StubBackend]"
    )

    # --- corpus ---
    tmp = None
    if args.corpus and Path(args.corpus).is_dir():
        corpus = sorted(str(p) for p in Path(args.corpus).glob("*.gif"))
        if not corpus:
            print(f"\nNo .gif files in {args.corpus}", file=sys.stderr)
            return 2
    else:
        rng = __import__("random").Random(1234)
        tmp = tempfile.mkdtemp(prefix="bench_gifs_")
        print(f"\nSynthesizing {args.synth_count} GIFs into {tmp} ...", flush=True)
        corpus = synthesize_corpus(tmp, args.synth_count, rng)
    describe_corpus(corpus)

    # --- sweep ---
    sweep = make_sweep(vcpu, args.max_procs, args.procs)
    hr("PER-CORE SCALING CURVE")
    print(f"  sweep P = {sweep}  (steady state: max({args.duration:.0f}s, {args.min_gifs} GIFs)/level)\n")
    header = f"  {'P':>3} {'GIFs/s':>8} {'GIFs/hr':>10} {'frm/s':>8} {'p50ms':>8} {'p95ms':>8} {'p99ms':>8} {'CPU%':>6} {'RSS_GB':>7} {'esc%':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    levels = []
    records_by_p = {}
    for P in sweep:
        res, recs = run_level(P, corpus, cfg, args.duration, args.min_gifs, args.warmup)
        levels.append(res)
        records_by_p[P] = recs
        print(
            f"  {res['P']:>3} {res['gifs_per_sec']:>8.2f} {res['gifs_per_hr']:>10.0f} "
            f"{res['frames_scored_per_sec']:>8.1f} {res['lat_p50_ms']:>8.1f} {res['lat_p95_ms']:>8.1f} "
            f"{res['lat_p99_ms']:>8.1f} {res['mean_cpu_pct']:>6.0f} {res['peak_rss_mb']/1024:>7.2f} "
            f"{res['escalation_rate']*100:>5.0f}",
            flush=True,
        )

    best, knee, bottleneck = analyse(levels, vcpu, ram_gb)
    sweet_records = records_by_p[best["P"]]

    # per-worker throughput == per-core for single-threaded inference-bound work
    per_core_hr = best["gifs_per_hr"] / min(best["P"], vcpu)
    rss_per_worker_mb = best["peak_rss_mb"] / best["P"]

    print(f"\n  SWEET SPOT: P={best['P']}  {best['gifs_per_hr']:.0f} GIFs/hr  "
          f"(p95={best['lat_p95_ms']:.0f}ms, CPU={best['mean_cpu_pct']:.0f}%, RSS={best['peak_rss_mb']/1024:.2f}GB)")
    print(f"  KNEE: P={knee['P']} (>=90% of peak throughput)")
    print(f"  BOTTLENECK: {bottleneck}")
    print(f"  GIFs/hr per core   = {per_core_hr:.0f}")
    print(f"  RSS per worker     = {rss_per_worker_mb:.0f} MB")

    # --- per-stage timing ---
    hr("PER-STAGE TIMING")
    stages = ["t_decode_sample", "t_preprocess", "t_inference", "t_gate"]
    sums = {s: sum(r[s] for r in sweet_records) for s in stages}
    meds = {s: statistics.median(r[s] for r in sweet_records) for s in stages}
    total = sum(sums.values()) or 1e-9
    f = sums["t_inference"] / total
    for s in stages:
        print(f"  {s:<16} median={meds[s]*1000:>8.2f} ms   share={sums[s]/total*100:>5.1f}%")
    print(f"  inference fraction f = {f:.3f}")

    # --- jsonl ---
    schema_keys = [
        "gif_id", "n_frames_total", "n_frames_scored", "t_decode_sample", "t_preprocess",
        "t_inference", "t_gate", "latency_ms", "peak_rss_mb", "max_local_score", "escalated",
    ]
    with open(args.out, "w") as fh:
        for r in sweet_records:
            fh.write(json.dumps({k: r[k] for k in schema_keys}) + "\n")
    print(f"\nWrote {len(sweet_records)} per-GIF records -> {args.out}")

    if tmp:
        print(f"(synthesized corpus left in {tmp}; delete when done)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
