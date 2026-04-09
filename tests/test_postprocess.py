"""
Tests for post-processing NMS utilities.

Tests the unified NMS interface across all backends
(C++ extension, torchvision, pure Python).
"""

import pytest
import torch

from app.postprocess import (
    batched_nms,
    get_nms_backend,
    nms,
    xywh_to_xyxy,
    _nms_python,
)


class TestNMSBackend:
    """Tests for NMS backend detection."""

    def test_backend_is_detected(self):
        """Backend detection returns a valid string."""
        backend = get_nms_backend()
        assert backend in ("edge_nms", "torchvision", "python")


class TestNMS:
    """Tests for the unified NMS interface."""

    def test_nms_basic_suppression(self):
        """Overlapping boxes: lower-score box should be suppressed."""
        boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.1, 1.1, 1.1],  # overlaps with box 0
            [5.0, 5.0, 6.0, 6.0],  # no overlap
        ], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8, 0.7])

        keep = nms(boxes, scores, iou_threshold=0.5)
        keep_list = keep.tolist()

        assert 0 in keep_list  # highest score kept
        assert 1 not in keep_list  # suppressed by box 0
        assert 2 in keep_list  # no overlap, kept

    def test_nms_no_suppression(self):
        """Non-overlapping boxes should all be kept."""
        boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
            [4.0, 4.0, 5.0, 5.0],
        ], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8, 0.7])

        keep = nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 3

    def test_nms_empty_input(self):
        """Empty input should return empty tensor."""
        boxes = torch.empty(0, 4)
        scores = torch.empty(0)

        keep = nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 0

    def test_nms_single_box(self):
        """Single box should always be kept."""
        boxes = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
        scores = torch.tensor([0.9])

        keep = nms(boxes, scores, iou_threshold=0.5)
        assert keep.tolist() == [0]

    def test_nms_returns_sorted_by_score(self):
        """Kept indices should be in score-descending order."""
        boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [10.0, 10.0, 11.0, 11.0],
            [20.0, 20.0, 21.0, 21.0],
        ], dtype=torch.float32)
        scores = torch.tensor([0.3, 0.9, 0.6])

        keep = nms(boxes, scores, iou_threshold=0.5)
        # Score order: 1 (0.9), 2 (0.6), 0 (0.3)
        assert keep.tolist() == [1, 2, 0]


class TestBatchedNMS:
    """Tests for per-class NMS."""

    def test_batched_nms_different_classes(self):
        """Overlapping boxes of different classes should NOT suppress each other."""
        boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.1, 1.1, 1.1],  # overlaps with box 0 but different class
        ], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8])
        classes = torch.tensor([0, 1], dtype=torch.int64)

        keep = batched_nms(boxes, scores, classes, iou_threshold=0.5)
        assert len(keep) == 2  # Both kept (different classes)

    def test_batched_nms_same_class(self):
        """Overlapping boxes of same class should suppress."""
        boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.1, 1.1, 1.1],
        ], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8])
        classes = torch.tensor([0, 0], dtype=torch.int64)

        keep = batched_nms(boxes, scores, classes, iou_threshold=0.5)
        assert len(keep) == 1


class TestXYWHToXYXY:
    """Tests for coordinate conversion."""

    def test_basic_conversion(self):
        """Center-form to corner-form conversion."""
        xywh = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        xyxy = xywh_to_xyxy(xywh)
        expected = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        assert torch.allclose(xyxy, expected, atol=1e-6)

    def test_multiple_boxes(self):
        """Multiple boxes converted correctly."""
        xywh = torch.tensor([
            [5.0, 5.0, 2.0, 4.0],
            [10.0, 10.0, 6.0, 2.0],
        ])
        xyxy = xywh_to_xyxy(xywh)
        expected = torch.tensor([
            [4.0, 3.0, 6.0, 7.0],
            [7.0, 9.0, 13.0, 11.0],
        ])
        assert torch.allclose(xyxy, expected, atol=1e-6)

    def test_empty_input(self):
        """Empty input should return empty tensor."""
        xywh = torch.empty(0, 4)
        xyxy = xywh_to_xyxy(xywh)
        assert xyxy.shape == (0, 4)


class TestPythonFallback:
    """Tests that the pure-Python NMS fallback works correctly."""

    def test_python_nms_matches_expected(self):
        """Pure Python NMS should give same results as C++ version."""
        boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.1, 1.1, 1.1],
            [5.0, 5.0, 6.0, 6.0],
        ], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8, 0.7])

        keep = _nms_python(boxes, scores, 0.5)
        keep_list = keep.tolist()

        assert 0 in keep_list
        assert 1 not in keep_list
        assert 2 in keep_list
