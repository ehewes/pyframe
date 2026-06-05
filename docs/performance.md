# Performance

How PyFrame's GIF moderation path (decode -> motion-sample -> local ViT prescreen ->
gate) performs, and how to reproduce it.

Absolute numbers are **illustrative** (single CPU core, single-threaded) and scale with
hardware and model. The library-relevant takeaway is the *shape*, which is
hardware-independent: the pipeline is **inference-bound**.

Pinned config for every number below: `prescreen` on, `screen_fps=2.0`,
`escalate_threshold=0.15`, `frames_per_batch=2`, model `AdamCodd/vit-base-nsfw-detector`,
torch CPU single-threaded, AWS stubbed (so this is local throughput only).

## Per-stage timing (the defining characteristic)

![PyFrame per-stage timing](https://raw.githubusercontent.com/ehewes/pyframe/main/media/perf_stages.png)

| Stage | Share | Illustrative |
|-------|-------|--------------|
| decode + sample | ~9% | 14 ms |
| preprocess (to PIL) | ~0.1% | 0.2 ms |
| **inference (ViT)** | **~91%** | 239 ms |
| gate | ~0% | <0.01 ms |

The ViT forward pass is essentially the entire cost (`f ~ 0.91`). Decode, sampling, and
the gate are negligible, so the only things that move throughput are the **model**
(smaller / quantized) and the **backend** (CPU vs GPU). The proportions hold across
hardware; only the absolute milliseconds change.

## Throughput & latency (single worker, one core)

![PyFrame latency percentiles](https://raw.githubusercontent.com/ehewes/pyframe/main/media/perf_latency.png)

- **~3 GIFs/s per core** (~11k GIFs/hr), single-threaded.
- per-GIF latency: p50 ~250 ms, p95 ~1.1 s. The spread tracks GIF size (more frames =
  more ViT calls), not load.

Per-core throughput is the unit that transfers between machines, not a box total.

## Memory

~0.5 GB resident per worker (model weights + buffers). Memory is not the bottleneck for
this path.

## Decode: file path vs in-memory (`scan_bytes`)

60-frame 320px GIF, 25 runs:

| Decoder | Median |
|---------|--------|
| cv2 file path (`iter_frames`) | 39 ms |
| in-memory bytes (`scan_bytes`) | 31 ms |

In-memory is ~21% faster (skips cv2/ffmpeg per-open overhead) and never touches disk.
Against ~239 ms of inference, the difference is noise.

## Notes

- **Inference-bound (`f ~ 0.91`):** the model and the backend are the only real levers;
  decode and sampling are already negligible.
- **Concurrency scaling is hardware-dependent.** Many single-threaded workers contend on
  memory bandwidth, so per-worker throughput can drop under load. Measure on your own
  target rather than multiplying the single-core number blindly.

## Reproduce

```bash
pip install psutil matplotlib                          # bench/plot tools, not runtime deps
python scripts/bench_gifs.py --procs "1" --duration 30  # single-worker profile -> bench_results.jsonl
python scripts/bench_decode.py                          # decode comparison
python scripts/plot_results.py bench_results.jsonl      # regenerate the charts in media/
```

## Results log

Append a row when you measure on new hardware.

| Run | Backend / threads | GIFs/s per core | f (inference share) | RSS/worker |
|-----|-------------------|-----------------|---------------------|------------|
| reference | torch CPU, 1 thread | ~3.1 | ~0.91 | ~0.5 GB |
| | | | | |
