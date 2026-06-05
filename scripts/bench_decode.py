#!/usr/bin/env python3
"""Micro-benchmark: cv2 file-path decode vs in-memory Pillow decode (the scan_bytes path).

Shows the per-GIF decode cost of each, so you can see the millisecond delta against
the ~789 ms ViT inference that dominates total time.

    python scripts/bench_decode.py [path/to.gif]   # synthesizes one if omitted
"""

import os
import statistics
import sys
import tempfile
import time

import numpy as np
from PIL import Image

from pyframe.media import iter_frames, iter_frames_from_bytes


def synth(path, n=60, w=320):
    h = int(w * 0.6)
    frames = []
    for i in range(n):
        a = np.random.randint(0, 255, (h, w, 3), np.uint8)
        x = int(i / n * (w - 24))
        a[h // 2 - 8 : h // 2 + 8, x : x + 24] = 255
        frames.append(Image.fromarray(a))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=66, loop=0)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        path = os.path.join(tempfile.mkdtemp(), "decode_bench.gif")
        synth(path)
    data = open(path, "rb").read()

    N = 25
    cv2_ms, mem_ms = [], []
    for _ in range(N):
        t = time.perf_counter()
        n1 = len(list(iter_frames(path)))
        cv2_ms.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        n2 = len(list(iter_frames_from_bytes(data)))
        mem_ms.append((time.perf_counter() - t) * 1000)

    c, m = statistics.median(cv2_ms), statistics.median(mem_ms)
    print(f"GIF: {n1} frames (cv2) / {n2} frames (mem), {len(data) / 1e6:.2f} MB, {N} runs")
    print(f"  cv2 file-path decode:   median {c:6.1f} ms")
    print(f"  in-memory bytes decode: median {m:6.1f} ms")
    delta = m - c
    print(f"  delta: {delta:+.1f} ms ({delta / c * 100:+.0f}%)  -- vs ~789 ms ViT inference, this is noise")
