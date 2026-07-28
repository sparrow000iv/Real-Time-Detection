"""Tests for the stream generator and benchmarking harness."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.benchmark import (benchmark, compare, latency_stats, save_report,
                           score_detections, speedup_table, sweep_stride)
from src.detector import Box, CascadeDetector, NaiveDetector, VectorizedDetector
from src.stream import (FRAME_H, FRAME_W, TARGET_H, TARGET_W, generate_stream,
                        iter_stream, make_template)


class TestStream:
    def test_frame_count(self):
        assert len(generate_stream(n_frames=7, seed=1)) == 7

    def test_frame_shape(self):
        f = generate_stream(n_frames=1, seed=1)[0]
        assert f.image.shape == (FRAME_H, FRAME_W)
        assert f.image.dtype == np.uint8

    def test_ground_truth_present(self):
        for f in generate_stream(n_frames=5, seed=1):
            assert len(f.boxes) == 1
            x, y, w, h = f.boxes[0]
            assert w == TARGET_W and h == TARGET_H

    def test_boxes_inside_frame(self):
        for f in generate_stream(n_frames=20, seed=3):
            for x, y, w, h in f.boxes:
                assert 0 <= x <= FRAME_W - w
                assert 0 <= y <= FRAME_H - h

    def test_target_moves(self):
        frames = generate_stream(n_frames=10, seed=1)
        positions = {f.boxes[0][:2] for f in frames}
        assert len(positions) > 1, "target never moved"

    def test_multiple_targets(self):
        f = generate_stream(n_frames=3, seed=1, n_targets=3)[0]
        assert len(f.boxes) == 3

    def test_reproducible(self):
        a = generate_stream(n_frames=4, seed=11)
        b = generate_stream(n_frames=4, seed=11)
        assert all(np.array_equal(x.image, y.image) for x, y in zip(a, b))

    def test_different_seeds_differ(self):
        a = generate_stream(n_frames=2, seed=1)
        b = generate_stream(n_frames=2, seed=2)
        assert not np.array_equal(a[0].image, b[0].image)

    def test_noise_changes_frames(self):
        a = generate_stream(n_frames=2, seed=1, noise=0.0)
        b = generate_stream(n_frames=2, seed=1, noise=1.0)
        assert not np.array_equal(a[0].image, b[0].image)

    def test_iter_stream(self):
        frames = generate_stream(n_frames=3, seed=1)
        assert len(list(iter_stream(frames))) == 3

    def test_template_shape(self):
        assert make_template().shape == (TARGET_H, TARGET_W)

    def test_template_has_contrast(self):
        assert make_template().std() > 30


class TestLatencyStats:
    def test_empty(self):
        s = latency_stats([])
        assert s.mean_ms == 0 and s.fps == 0

    def test_known_values(self):
        s = latency_stats([10.0] * 10)
        assert s.mean_ms == 10.0
        assert s.p50_ms == 10.0
        assert s.fps == pytest.approx(100.0)

    def test_percentile_ordering(self):
        s = latency_stats(list(range(1, 101)))
        assert s.min_ms <= s.p50_ms <= s.p95_ms <= s.p99_ms <= s.max_ms

    def test_fps_is_inverse_of_mean(self):
        s = latency_stats([25.0] * 5)
        assert s.fps == pytest.approx(40.0)


class TestScoreDetections:
    def _frame(self, boxes):
        from src.stream import Frame
        return Frame(index=0, image=np.zeros((10, 10), np.uint8), boxes=boxes)

    def test_perfect_detection(self):
        f = self._frame([(0, 0, 10, 10)])
        m = score_detections([[Box(0, 0, 10, 10, .9)]], [f])
        assert m.precision == 1.0 and m.recall == 1.0 and m.tp == 1

    def test_missed_detection(self):
        f = self._frame([(0, 0, 10, 10)])
        m = score_detections([[]], [f])
        assert m.recall == 0.0 and m.fn == 1

    def test_false_positive(self):
        f = self._frame([])
        m = score_detections([[Box(0, 0, 10, 10, .9)]], [f])
        assert m.fp == 1 and m.precision == 0.0

    def test_low_iou_counts_as_fp(self):
        f = self._frame([(0, 0, 10, 10)])
        m = score_detections([[Box(9, 9, 10, 10, .9)]], [f], iou_threshold=0.5)
        assert m.tp == 0 and m.fp == 1

    def test_duplicate_predictions_penalised(self):
        f = self._frame([(0, 0, 10, 10)])
        preds = [[Box(0, 0, 10, 10, .9), Box(0, 0, 10, 10, .8)]]
        m = score_detections(preds, [f])
        assert m.tp == 1 and m.fp == 1

    def test_f1_is_harmonic_mean(self):
        f = self._frame([(0, 0, 10, 10)])
        m = score_detections([[Box(0, 0, 10, 10, .9)]], [f])
        assert m.f1 == pytest.approx(1.0)


class TestBenchmark:
    @pytest.fixture(scope="class")
    def frames(self):
        return generate_stream(n_frames=6, seed=42)

    @pytest.fixture(scope="class")
    def template(self):
        return make_template()

    def test_returns_result(self, frames, template):
        r = benchmark(VectorizedDetector(template, stride=8), frames, warmup=1)
        assert r.frames == len(frames)
        assert r.latency.mean_ms > 0

    def test_detects_accurately(self, frames, template):
        r = benchmark(VectorizedDetector(template, stride=4), frames, warmup=1)
        assert r.accuracy.recall > 0.8

    def test_realtime_flag(self, frames, template):
        r = benchmark(VectorizedDetector(template, stride=8), frames,
                      warmup=1, target_fps=0.001)
        assert r.realtime_capable is True

    def test_cascade_reports_rejection(self, frames, template):
        r = benchmark(CascadeDetector(template, stride=8), frames, warmup=1)
        assert "stage1_rejection_rate" in r.extra

    def test_optimised_is_faster_than_naive(self, frames, template):
        slow = benchmark(NaiveDetector(template, stride=4), frames, warmup=1)
        fast = benchmark(VectorizedDetector(template, stride=4), frames, warmup=1)
        assert fast.latency.mean_ms < slow.latency.mean_ms

    def test_optimisation_is_lossless(self, frames, template):
        slow = benchmark(NaiveDetector(template, stride=4), frames, warmup=1)
        fast = benchmark(VectorizedDetector(template, stride=4), frames, warmup=1)
        assert fast.accuracy.f1 == pytest.approx(slow.accuracy.f1)

    def test_speedup_table(self, frames, template):
        rs = [benchmark(c(template, stride=8), frames, warmup=1)
              for c in (NaiveDetector, VectorizedDetector)]
        sp = speedup_table(rs)
        assert min(sp.values()) == pytest.approx(1.0)

    def test_compare_renders(self, frames, template):
        rs = [benchmark(VectorizedDetector(template, stride=8), frames, warmup=1)]
        assert "detector" in compare(rs)

    def test_save_report(self, frames, template, tmp_path):
        rs = [benchmark(VectorizedDetector(template, stride=8), frames, warmup=1)]
        p = tmp_path / "b.json"
        save_report(rs, p, metadata={"x": 1})
        data = json.loads(p.read_text())
        assert "results" in data and "speedup_vs_slowest" in data

    def test_stride_sweep_trades_accuracy_for_speed(self, frames, template):
        out = sweep_stride(VectorizedDetector, template, frames, strides=(4, 16))
        (_, fine), (_, coarse) = out
        assert coarse.latency.mean_ms < fine.latency.mean_ms
        assert coarse.accuracy.f1 <= fine.accuracy.f1
