#!/usr/bin/env python3
"""Real-time detection pipeline benchmark.

Examples:
    python run_benchmark.py
    python run_benchmark.py --frames 120 --min-fps 30
    python run_benchmark.py --sweep-stride
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from src.benchmark import (benchmark, compare, save_report, speedup_table,
                           sweep_stride)
from src.detector import CascadeDetector, NaiveDetector, VectorizedDetector
from src.stream import generate_stream, make_template


def main() -> int:
    ap = argparse.ArgumentParser(description="Real-time detection benchmark")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--targets", type=int, default=1)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--min-fps", type=float, default=None,
                    help="fail if the best backend cannot hit this frame rate")
    ap.add_argument("--sweep-stride", action="store_true")
    ap.add_argument("--json", type=Path, default=Path("reports/benchmark.json"))
    args = ap.parse_args()

    print("=" * 74)
    print("REAL-TIME OBJECT DETECTION - LATENCY & THROUGHPUT BENCHMARK")
    print("=" * 74)
    print(f"\nPlatform: {platform.python_implementation()} {platform.python_version()} "
          f"on {platform.machine()}")

    frames = generate_stream(n_frames=args.frames, seed=args.seed,
                             n_targets=args.targets, noise=args.noise)
    template = make_template()
    print(f"Stream: {len(frames)} frames @ {frames[0].image.shape[1]}x"
          f"{frames[0].image.shape[0]}, {args.targets} target(s), stride={args.stride}")

    backends = [
        NaiveDetector(template, stride=args.stride),
        VectorizedDetector(template, stride=args.stride),
        CascadeDetector(template, stride=args.stride),
    ]

    print("\nRunning benchmarks (with warm-up)...\n")
    results = [benchmark(d, frames) for d in backends]

    print(compare(results))

    print("\nSPEEDUP vs slowest backend")
    for name, s in sorted(speedup_table(results).items(), key=lambda kv: kv[1]):
        print(f"  {name:<14}{s:>6.2f}x")

    for r in results:
        if "stage1_rejection_rate" in r.extra:
            print(f"\nCascade stage-1 filter rejected "
                  f"{r.extra['stage1_rejection_rate']:.1%} of candidate windows "
                  f"before scoring")

    print("\nACCURACY (identical across backends confirms optimisation is lossless)")
    for r in results:
        a = r.accuracy
        print(f"  {r.detector:<14}P={a.precision:.3f} R={a.recall:.3f} "
              f"F1={a.f1:.3f} IoU={a.mean_iou:.3f} (tp={a.tp} fp={a.fp} fn={a.fn})")

    f1s = {round(r.accuracy.f1, 4) for r in results}
    print(f"  -> {'LOSSLESS: all backends agree' if len(f1s) == 1 else 'WARNING: F1 differs across backends'}")

    if args.sweep_stride:
        print("\nSTRIDE SWEEP (accuracy vs latency trade-off)")
        print(f"  {'stride':>7}{'mean_ms':>10}{'FPS':>8}{'F1':>8}{'IoU':>8}")
        for s, r in sweep_stride(VectorizedDetector, template, frames):
            print(f"  {s:>7}{r.latency.mean_ms:>10.2f}{r.latency.fps:>8.1f}"
                  f"{r.accuracy.f1:>8.3f}{r.accuracy.mean_iou:>8.3f}")

    save_report(results, args.json, metadata={
        "frames": len(frames),
        "resolution": f"{frames[0].image.shape[1]}x{frames[0].image.shape[0]}",
        "stride": args.stride,
        "targets": args.targets,
        "python": platform.python_version(),
    })
    print(f"\nJSON report written to {args.json}")

    best = max(results, key=lambda r: r.latency.fps)
    print(f"\nBest backend: {best.detector} at {best.latency.fps:.1f} FPS "
          f"(p95 latency {best.latency.p95_ms:.2f} ms)")

    if args.min_fps is not None:
        if best.latency.fps < args.min_fps:
            print(f"\nPERFORMANCE GATE FAILED: {best.latency.fps:.1f} < {args.min_fps} FPS")
            return 1
        print(f"\nPERFORMANCE GATE PASSED: {best.latency.fps:.1f} >= {args.min_fps} FPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
