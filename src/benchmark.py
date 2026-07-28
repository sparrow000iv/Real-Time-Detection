"""Latency, throughput and accuracy benchmarking for detection backends.

Reports percentile latency (p50/p95/p99) rather than only the mean, because
real-time systems are judged by tail behaviour: a pipeline averaging 20 ms but
spiking to 80 ms drops frames.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .detector import Box, Detector
from .stream import Frame


@dataclass
class DetectionMetrics:
    """Accuracy of a detector over a stream."""

    precision: float
    recall: float
    f1: float
    mean_iou: float
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mean_iou": round(self.mean_iou, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class LatencyStats:
    """Per-frame timing distribution."""

    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    fps: float

    def to_dict(self) -> dict:
        return {
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "fps": round(self.fps, 2),
        }


@dataclass
class BenchmarkResult:
    """Combined accuracy and performance for one backend."""

    detector: str
    latency: LatencyStats
    accuracy: DetectionMetrics
    frames: int
    realtime_capable: bool
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "frames": self.frames,
            "realtime_capable_30fps": self.realtime_capable,
            "latency": self.latency.to_dict(),
            "accuracy": self.accuracy.to_dict(),
            "extra": {k: round(v, 4) for k, v in self.extra.items()},
        }


def latency_stats(timings_ms: Sequence[float]) -> LatencyStats:
    """Summarise a list of per-frame durations."""
    a = np.asarray(timings_ms, dtype=float)
    if a.size == 0:
        return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0)
    mean = float(a.mean())
    return LatencyStats(
        mean_ms=mean,
        p50_ms=float(np.percentile(a, 50)),
        p95_ms=float(np.percentile(a, 95)),
        p99_ms=float(np.percentile(a, 99)),
        min_ms=float(a.min()),
        max_ms=float(a.max()),
        std_ms=float(a.std()),
        fps=1000.0 / mean if mean > 0 else 0.0,
    )


def score_detections(
    predictions: Sequence[Sequence[Box]],
    frames: Sequence[Frame],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Match predictions to ground truth by greedy IoU."""
    tp = fp = fn = 0
    ious: List[float] = []

    for preds, frame in zip(predictions, frames):
        gts = [Box(x, y, w, h, 1.0) for (x, y, w, h) in frame.boxes]
        unmatched = list(range(len(gts)))

        for p in sorted(preds, key=lambda b: b.score, reverse=True):
            best_i, best_iou = -1, 0.0
            for gi in unmatched:
                v = p.iou(gts[gi])
                if v > best_iou:
                    best_i, best_iou = gi, v
            if best_i >= 0 and best_iou >= iou_threshold:
                tp += 1
                ious.append(best_iou)
                unmatched.remove(best_i)
            else:
                fp += 1
        fn += len(unmatched)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=float(np.mean(ious)) if ious else 0.0,
        tp=tp, fp=fp, fn=fn,
    )


def benchmark(
    detector: Detector,
    frames: Sequence[Frame],
    warmup: int = 3,
    target_fps: float = 30.0,
    iou_threshold: float = 0.5,
) -> BenchmarkResult:
    """Time a detector across a stream and score its accuracy.

    A warm-up pass is run first and discarded, so allocator and cache effects
    do not contaminate the measured distribution.
    """
    for f in frames[:warmup]:
        detector.detect(f.image)

    timings: List[float] = []
    predictions: List[List[Box]] = []

    for f in frames:
        t0 = time.perf_counter()
        boxes = detector.detect(f.image)
        timings.append((time.perf_counter() - t0) * 1000.0)
        predictions.append(boxes)

    lat = latency_stats(timings)
    acc = score_detections(predictions, frames, iou_threshold)

    extra: Dict[str, float] = {}
    if hasattr(detector, "rejection_rate"):
        extra["stage1_rejection_rate"] = float(detector.rejection_rate)

    return BenchmarkResult(
        detector=detector.name,
        latency=lat,
        accuracy=acc,
        frames=len(frames),
        realtime_capable=lat.fps >= target_fps,
        extra=extra,
    )


def compare(results: Sequence[BenchmarkResult]) -> str:
    """Render a comparison table across backends."""
    lines = [
        f"{'detector':<14}{'mean':>9}{'p50':>8}{'p95':>8}{'p99':>8}"
        f"{'FPS':>8}{'F1':>7}{'IoU':>7}{'RT':>5}",
        "-" * 74,
    ]
    for r in results:
        lines.append(
            f"{r.detector:<14}{r.latency.mean_ms:>8.2f}m{r.latency.p50_ms:>8.2f}"
            f"{r.latency.p95_ms:>8.2f}{r.latency.p99_ms:>8.2f}"
            f"{r.latency.fps:>8.1f}{r.accuracy.f1:>7.3f}"
            f"{r.accuracy.mean_iou:>7.3f}{'yes' if r.realtime_capable else 'no':>5}"
        )
    return "\n".join(lines)


def speedup_table(results: Sequence[BenchmarkResult]) -> Dict[str, float]:
    """Speedup of each backend relative to the slowest."""
    if not results:
        return {}
    slowest = max(r.latency.mean_ms for r in results)
    return {r.detector: slowest / r.latency.mean_ms for r in results}


def save_report(
    results: Sequence[BenchmarkResult],
    path: Path,
    metadata: Optional[dict] = None,
) -> None:
    """Write the full benchmark report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "speedup_vs_slowest": {k: round(v, 2) for k, v in speedup_table(results).items()},
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sweep_stride(
    detector_cls,
    template: np.ndarray,
    frames: Sequence[Frame],
    strides: Sequence[int] = (2, 4, 8, 16),
) -> List[Tuple[int, BenchmarkResult]]:
    """Measure the accuracy/latency trade-off across search strides."""
    out: List[Tuple[int, BenchmarkResult]] = []
    for s in strides:
        det = detector_cls(template, stride=s)
        out.append((s, benchmark(det, frames, warmup=2)))
    return out
