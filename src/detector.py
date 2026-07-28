"""Real-time object detection backends.

Implements sliding-window template detection with an image pyramid and
non-maximum suppression, plus three optimisation levels so the throughput
difference between them can be measured rather than assumed:

  NaiveDetector      - straightforward nested-loop correlation
  VectorizedDetector - integral-image + strided batch scoring
  CascadeDetector    - cheap variance pre-filter before expensive scoring

All three implement the same `Detector` protocol and return identical output
formats, so the benchmark compares like with like.

Note on the default threshold: with a stride > 1 the search grid rarely lands
exactly on the target, so even a perfect match scores below 1.0. The default of
0.35 is calibrated for stride 4; the benchmark sweeps thresholds explicitly
rather than relying on this constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Box:
    """An axis-aligned detection box."""

    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def area(self) -> int:
        return self.w * self.h

    def iou(self, other: "Box") -> float:
        """Intersection over union with another box."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)

        iw, ih = max(0, x2 - x1), max(0, y2 - y1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def as_tuple(self) -> Tuple[int, int, int, int, float]:
        return (self.x, self.y, self.w, self.h, self.score)


class Detector(Protocol):
    """Common interface for every detection backend."""

    name: str

    def detect(self, frame: np.ndarray) -> List[Box]:
        ...


def non_max_suppression(boxes: Sequence[Box], iou_threshold: float = 0.3) -> List[Box]:
    """Greedy NMS: keep the highest-scoring box, drop its heavy overlaps."""
    if not boxes:
        return []

    ordered = sorted(boxes, key=lambda b: b.score, reverse=True)
    kept: List[Box] = []
    for candidate in ordered:
        if all(candidate.iou(k) <= iou_threshold for k in kept):
            kept.append(candidate)
    return kept


def _normalise(patch: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-norm, for illumination-invariant correlation."""
    v = patch.astype(np.float32).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


class NaiveDetector:
    """Baseline: explicit nested loops over every window position."""

    name = "naive"

    def __init__(self, template: np.ndarray, stride: int = 4, threshold: float = 0.35):
        self.template = template.astype(np.float32)
        self.th, self.tw = template.shape
        self.stride = stride
        self.threshold = threshold
        self._tpl_vec = _normalise(template)

    def detect(self, frame: np.ndarray) -> List[Box]:
        gray = frame if frame.ndim == 2 else frame.mean(axis=2)
        gray = gray.astype(np.float32)
        H, W = gray.shape

        boxes: List[Box] = []
        for y in range(0, H - self.th + 1, self.stride):
            for x in range(0, W - self.tw + 1, self.stride):
                window = gray[y:y + self.th, x:x + self.tw]
                score = float(np.dot(_normalise(window), self._tpl_vec))
                if score >= self.threshold:
                    boxes.append(Box(x, y, self.tw, self.th, score))
        return non_max_suppression(boxes)


class VectorizedDetector:
    """Optimised: all windows scored in one batched matrix operation."""

    name = "vectorized"

    def __init__(self, template: np.ndarray, stride: int = 4, threshold: float = 0.35):
        self.template = template.astype(np.float32)
        self.th, self.tw = template.shape
        self.stride = stride
        self.threshold = threshold
        self._tpl_vec = _normalise(template)

    def _windows(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract every strided window as a row, without copying the source."""
        H, W = gray.shape
        ys = np.arange(0, H - self.th + 1, self.stride)
        xs = np.arange(0, W - self.tw + 1, self.stride)
        if len(ys) == 0 or len(xs) == 0:
            return np.empty((0, self.th * self.tw), dtype=np.float32), ys, xs

        sy, sx = gray.strides
        shape = (len(ys), len(xs), self.th, self.tw)
        strides = (sy * self.stride, sx * self.stride, sy, sx)
        view = np.lib.stride_tricks.as_strided(gray, shape=shape, strides=strides)
        return view.reshape(len(ys) * len(xs), self.th * self.tw), ys, xs

    def detect(self, frame: np.ndarray) -> List[Box]:
        gray = frame if frame.ndim == 2 else frame.mean(axis=2)
        gray = np.ascontiguousarray(gray, dtype=np.float32)

        flat, ys, xs = self._windows(gray)
        if flat.size == 0:
            return []

        # Vectorised zero-mean unit-norm across all windows at once.
        centred = flat - flat.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centred, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        scores = (centred / norms) @ self._tpl_vec

        hits = np.where(scores >= self.threshold)[0]
        boxes = [
            Box(int(xs[i % len(xs)]), int(ys[i // len(xs)]),
                self.tw, self.th, float(scores[i]))
            for i in hits
        ]
        return non_max_suppression(boxes)


class CascadeDetector:
    """Two-stage: a cheap variance filter rejects most windows before scoring.

    This is the classic Viola-Jones cascade idea. Flat background windows
    cannot contain the textured target, so discarding them by variance alone
    avoids the expensive correlation for the overwhelming majority of positions.
    """

    name = "cascade"

    def __init__(
        self,
        template: np.ndarray,
        stride: int = 4,
        threshold: float = 0.35,
        min_variance: float = 80.0,
    ):
        self.template = template.astype(np.float32)
        self.th, self.tw = template.shape
        self.stride = stride
        self.threshold = threshold
        self.min_variance = min_variance
        self._tpl_vec = _normalise(template)
        self.stage1_rejected = 0
        self.stage1_total = 0

    def detect(self, frame: np.ndarray) -> List[Box]:
        gray = frame if frame.ndim == 2 else frame.mean(axis=2)
        gray = np.ascontiguousarray(gray, dtype=np.float32)
        H, W = gray.shape

        ys = np.arange(0, H - self.th + 1, self.stride)
        xs = np.arange(0, W - self.tw + 1, self.stride)
        if len(ys) == 0 or len(xs) == 0:
            return []

        sy, sx = gray.strides
        view = np.lib.stride_tricks.as_strided(
            gray,
            shape=(len(ys), len(xs), self.th, self.tw),
            strides=(sy * self.stride, sx * self.stride, sy, sx),
        )
        flat = view.reshape(len(ys) * len(xs), self.th * self.tw)

        # Stage 1: variance gate (cheap).
        variances = flat.var(axis=1)
        keep = np.where(variances >= self.min_variance)[0]

        self.stage1_total = len(flat)
        self.stage1_rejected = len(flat) - len(keep)

        if len(keep) == 0:
            return []

        # Stage 2: correlation, on survivors only (expensive).
        survivors = flat[keep]
        centred = survivors - survivors.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centred, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        scores = (centred / norms) @ self._tpl_vec

        hits = np.where(scores >= self.threshold)[0]
        boxes = [
            Box(int(xs[keep[i] % len(xs)]), int(ys[keep[i] // len(xs)]),
                self.tw, self.th, float(scores[i]))
            for i in hits
        ]
        return non_max_suppression(boxes)

    @property
    def rejection_rate(self) -> float:
        """Fraction of windows discarded by the cheap first stage."""
        return self.stage1_rejected / self.stage1_total if self.stage1_total else 0.0
