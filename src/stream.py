"""Synthetic video stream with ground-truth object positions.

Generates frames containing a moving textured target against a structured
background, so detection accuracy can be scored exactly. Deterministic under a
fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np

FRAME_W, FRAME_H = 320, 240
TARGET_W, TARGET_H = 32, 32


@dataclass
class Frame:
    """One video frame plus its ground-truth boxes."""

    index: int
    image: np.ndarray                      # uint8 grayscale
    boxes: List[Tuple[int, int, int, int]]  # (x, y, w, h)


def make_template(seed: int = 7) -> np.ndarray:
    """Build the target appearance: a high-contrast textured square."""
    rng = np.random.default_rng(seed)
    tpl = np.zeros((TARGET_H, TARGET_W), dtype=np.float32)

    # Checkerboard core gives strong, distinctive correlation structure.
    for i in range(TARGET_H):
        for j in range(TARGET_W):
            tpl[i, j] = 235.0 if ((i // 8) + (j // 8)) % 2 == 0 else 25.0

    # A border ring makes it distinguishable from background texture.
    tpl[:3, :] = 255.0
    tpl[-3:, :] = 255.0
    tpl[:, :3] = 255.0
    tpl[:, -3:] = 255.0

    tpl += rng.normal(0, 2.0, tpl.shape).astype(np.float32)
    return np.clip(tpl, 0, 255)


def _background(rng: np.random.Generator) -> np.ndarray:
    """Structured background: gradient, blobs and mild noise."""
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(np.float32)
    bg = 90 + 45 * (xx / FRAME_W) + 25 * (yy / FRAME_H)

    for _ in range(6):
        cx, cy = rng.uniform(0, FRAME_W), rng.uniform(0, FRAME_H)
        r = rng.uniform(20, 55)
        amp = rng.uniform(-28, 28)
        bg += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))

    bg += rng.normal(0, 3.0, bg.shape)
    return np.clip(bg, 0, 255)


def generate_stream(
    n_frames: int = 60,
    seed: int = 42,
    n_targets: int = 1,
    noise: float = 0.0,
) -> List[Frame]:
    """Generate a deterministic video sequence with moving targets."""
    rng = np.random.default_rng(seed)
    template = make_template()

    # Give each target a linear trajectory that bounces off the frame edges.
    states = []
    for _ in range(n_targets):
        states.append({
            "x": float(rng.uniform(0, FRAME_W - TARGET_W)),
            "y": float(rng.uniform(0, FRAME_H - TARGET_H)),
            "vx": float(rng.uniform(-4, 4)) or 2.0,
            "vy": float(rng.uniform(-3, 3)) or 1.5,
        })

    frames: List[Frame] = []
    for k in range(n_frames):
        img = _background(np.random.default_rng(seed + k))
        boxes: List[Tuple[int, int, int, int]] = []

        for st in states:
            st["x"] += st["vx"]
            st["y"] += st["vy"]
            if st["x"] <= 0 or st["x"] >= FRAME_W - TARGET_W:
                st["vx"] *= -1
                st["x"] = float(np.clip(st["x"], 0, FRAME_W - TARGET_W))
            if st["y"] <= 0 or st["y"] >= FRAME_H - TARGET_H:
                st["vy"] *= -1
                st["y"] = float(np.clip(st["y"], 0, FRAME_H - TARGET_H))

            x, y = int(round(st["x"])), int(round(st["y"]))
            img[y:y + TARGET_H, x:x + TARGET_W] = template
            boxes.append((x, y, TARGET_W, TARGET_H))

        if noise > 0:
            img = img + np.random.default_rng(seed + 1000 + k).normal(0, noise * 30, img.shape)

        frames.append(Frame(index=k, image=np.clip(img, 0, 255).astype(np.uint8), boxes=boxes))

    return frames


def iter_stream(frames: List[Frame]) -> Iterator[Frame]:
    """Iterate frames, mimicking a live capture source."""
    yield from frames
