"""Tests for detection backends, IoU, and NMS."""
from __future__ import annotations

import numpy as np
import pytest

from src.detector import (Box, CascadeDetector, NaiveDetector,
                          VectorizedDetector, non_max_suppression)
from src.stream import generate_stream, make_template


class TestBox:
    def test_area(self):
        assert Box(0, 0, 10, 20, 1.0).area == 200

    def test_identical_iou_is_one(self):
        b = Box(5, 5, 10, 10, 1.0)
        assert b.iou(b) == 1.0

    def test_disjoint_iou_is_zero(self):
        assert Box(0, 0, 10, 10, 1.0).iou(Box(100, 100, 10, 10, 1.0)) == 0.0

    def test_half_overlap(self):
        a, b = Box(0, 0, 10, 10, 1.0), Box(5, 0, 10, 10, 1.0)
        # intersection 50, union 150
        assert a.iou(b) == pytest.approx(50 / 150)

    def test_symmetric(self):
        a, b = Box(0, 0, 10, 10, 1.0), Box(3, 3, 10, 10, 1.0)
        assert a.iou(b) == pytest.approx(b.iou(a))

    def test_touching_edges_is_zero(self):
        assert Box(0, 0, 10, 10, 1.0).iou(Box(10, 0, 10, 10, 1.0)) == 0.0

    def test_contained_box(self):
        outer, inner = Box(0, 0, 20, 20, 1.0), Box(5, 5, 10, 10, 1.0)
        assert outer.iou(inner) == pytest.approx(100 / 400)

    def test_as_tuple(self):
        assert Box(1, 2, 3, 4, 0.5).as_tuple() == (1, 2, 3, 4, 0.5)


class TestNMS:
    def test_empty(self):
        assert non_max_suppression([]) == []

    def test_keeps_single(self):
        assert len(non_max_suppression([Box(0, 0, 10, 10, .9)])) == 1

    def test_suppresses_overlap(self):
        boxes = [Box(0, 0, 10, 10, .9), Box(1, 1, 10, 10, .8)]
        assert len(non_max_suppression(boxes, iou_threshold=.3)) == 1

    def test_keeps_highest_score(self):
        boxes = [Box(0, 0, 10, 10, .5), Box(1, 1, 10, 10, .95)]
        assert non_max_suppression(boxes, .3)[0].score == .95

    def test_keeps_disjoint(self):
        boxes = [Box(0, 0, 10, 10, .9), Box(50, 50, 10, 10, .8)]
        assert len(non_max_suppression(boxes, .3)) == 2

    def test_threshold_controls_strictness(self):
        boxes = [Box(0, 0, 10, 10, .9), Box(6, 0, 10, 10, .8)]
        assert len(non_max_suppression(boxes, .01)) <= len(non_max_suppression(boxes, .99))

    def test_output_sorted_by_score(self):
        boxes = [Box(0, 0, 5, 5, .3), Box(50, 50, 5, 5, .9), Box(100, 100, 5, 5, .6)]
        out = non_max_suppression(boxes)
        assert [b.score for b in out] == sorted([b.score for b in out], reverse=True)


@pytest.fixture(scope="module")
def stream():
    return generate_stream(n_frames=6, seed=42)


@pytest.fixture(scope="module")
def template():
    return make_template()


BACKENDS = [NaiveDetector, VectorizedDetector, CascadeDetector]


class TestDetectorContract:
    @pytest.mark.parametrize("cls", BACKENDS)
    def test_returns_boxes(self, cls, template, stream):
        out = cls(template, stride=8).detect(stream[0].image)
        assert isinstance(out, list)
        assert all(isinstance(b, Box) for b in out)

    @pytest.mark.parametrize("cls", BACKENDS)
    def test_finds_the_target(self, cls, template, stream):
        f = stream[0]
        boxes = cls(template, stride=4).detect(f.image)
        gt = Box(*f.boxes[0], 1.0)
        assert boxes, "no detections at all"
        assert max(b.iou(gt) for b in boxes) > 0.5

    @pytest.mark.parametrize("cls", BACKENDS)
    def test_deterministic(self, cls, template, stream):
        d = cls(template, stride=8)
        a = [b.as_tuple() for b in d.detect(stream[0].image)]
        b = [b.as_tuple() for b in d.detect(stream[0].image)]
        assert a == b

    @pytest.mark.parametrize("cls", BACKENDS)
    def test_handles_blank_frame(self, cls, template):
        blank = np.full((240, 320), 128, dtype=np.uint8)
        assert cls(template, stride=8).detect(blank) == []

    @pytest.mark.parametrize("cls", BACKENDS)
    def test_handles_frame_smaller_than_template(self, cls, template):
        tiny = np.zeros((8, 8), dtype=np.uint8)
        assert cls(template, stride=4).detect(tiny) == []

    @pytest.mark.parametrize("cls", BACKENDS)
    def test_accepts_colour_frame(self, cls, template, stream):
        rgb = np.stack([stream[0].image] * 3, axis=2)
        assert isinstance(cls(template, stride=8).detect(rgb), list)

    @pytest.mark.parametrize("cls", BACKENDS)
    def test_scores_within_range(self, cls, template, stream):
        for b in cls(template, stride=4).detect(stream[0].image):
            assert -1.01 <= b.score <= 1.01


class TestBackendEquivalence:
    """The optimised backends must be lossless relative to the naive one."""

    def test_all_backends_agree(self, template, stream):
        results = {}
        for cls in BACKENDS:
            d = cls(template, stride=4)
            boxes = d.detect(stream[0].image)
            results[d.name] = sorted((b.x, b.y) for b in boxes)
        assert results["naive"] == results["vectorized"] == results["cascade"]

    def test_scores_match_closely(self, template, stream):
        n = {(b.x, b.y): b.score for b in NaiveDetector(template, stride=4).detect(stream[0].image)}
        v = {(b.x, b.y): b.score for b in VectorizedDetector(template, stride=4).detect(stream[0].image)}
        assert set(n) == set(v)
        for k in n:
            assert n[k] == pytest.approx(v[k], abs=1e-4)


class TestCascade:
    def test_rejects_most_windows(self, template, stream):
        d = CascadeDetector(template, stride=4)
        d.detect(stream[0].image)
        assert d.rejection_rate > 0.5, "cascade filter is not rejecting enough"

    def test_rejection_rate_before_run_is_zero(self, template):
        assert CascadeDetector(template).rejection_rate == 0.0

    def test_high_variance_gate_rejects_everything(self, template, stream):
        d = CascadeDetector(template, stride=8, min_variance=1e9)
        assert d.detect(stream[0].image) == []
