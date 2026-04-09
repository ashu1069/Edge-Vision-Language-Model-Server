"""
Post-processing utilities for YOLO detection output.

Tries to use the compiled C++ edge_nms extension for fast NMS.
Falls back to torchvision.ops.nms, then to a pure-Python implementation.

The C++ extension is optional — install with:
    cd csrc && pip install .
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection (runs once at import time)
# ---------------------------------------------------------------------------

_NMS_BACKEND: Optional[str] = None
_EDGE_NMS_MODULE = None  # Cached module reference (for JIT-compiled case)


def _get_edge_nms_module():
    """Return the edge_nms module (pre-installed or JIT-compiled)."""
    global _EDGE_NMS_MODULE
    if _EDGE_NMS_MODULE is not None:
        return _EDGE_NMS_MODULE
    import edge_nms
    return edge_nms


def _detect_nms_backend() -> str:
    """Detect the best available NMS implementation."""
    global _NMS_BACKEND, _EDGE_NMS_MODULE
    if _NMS_BACKEND is not None:
        return _NMS_BACKEND

    # 1. Try pre-installed C++ extension
    try:
        import edge_nms
        _EDGE_NMS_MODULE = edge_nms
        _NMS_BACKEND = "edge_nms"
        logger.info("NMS backend: edge_nms (C++ extension)")
        return _NMS_BACKEND
    except ImportError:
        pass

    # 2. Try JIT-compiling the C++ extension from source
    try:
        from pathlib import Path
        src = Path(__file__).parent.parent / "csrc" / "nms.cpp"
        if src.exists():
            import torch.utils.cpp_extension
            _EDGE_NMS_MODULE = torch.utils.cpp_extension.load(
                name="edge_nms",
                sources=[str(src)],
                extra_cflags=["-O3", "-std=c++17"],
                verbose=False,
            )
            _NMS_BACKEND = "edge_nms"
            logger.info("NMS backend: edge_nms (JIT-compiled from csrc/)")
            return _NMS_BACKEND
    except Exception as e:
        logger.debug(f"JIT compile of edge_nms failed: {e}")
        pass

    # 3. Try torchvision
    try:
        import torchvision.ops  # noqa: F401
        _NMS_BACKEND = "torchvision"
        logger.info("NMS backend: torchvision")
        return _NMS_BACKEND
    except ImportError:
        pass

    # 4. Fallback to pure Python
    _NMS_BACKEND = "python"
    logger.info("NMS backend: pure Python (install edge_nms or torchvision for speed)")
    return _NMS_BACKEND


def get_nms_backend() -> str:
    """Return the name of the active NMS backend."""
    return _detect_nms_backend()


# ---------------------------------------------------------------------------
# Unified NMS interface
# ---------------------------------------------------------------------------

def nms(boxes, scores, iou_threshold: float = 0.45):
    """
    Non-Maximum Suppression with automatic backend selection.

    Args:
        boxes: [N, 4] tensor in xyxy format (x1, y1, x2, y2)
        scores: [N] tensor of confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        1-D tensor of kept indices, sorted by score descending
    """
    backend = _detect_nms_backend()

    if backend == "edge_nms":
        _ext = _get_edge_nms_module()
        return _ext.nms(boxes, scores, iou_threshold)

    elif backend == "torchvision":
        from torchvision.ops import nms as tv_nms
        return tv_nms(boxes, scores, iou_threshold)

    else:
        return _nms_python(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, classes, iou_threshold: float = 0.45):
    """
    Per-class NMS with automatic backend selection.

    Args:
        boxes: [N, 4] tensor in xyxy format
        scores: [N] tensor of confidence scores
        classes: [N] tensor of class IDs (int64)
        iou_threshold: IoU threshold

    Returns:
        1-D tensor of kept indices
    """
    backend = _detect_nms_backend()

    if backend == "edge_nms":
        _ext = _get_edge_nms_module()
        return _ext.batched_nms(boxes, scores, classes, iou_threshold)

    elif backend == "torchvision":
        from torchvision.ops import batched_nms as tv_batched
        return tv_batched(boxes, scores, classes, iou_threshold)

    else:
        # Fallback: offset-based batched NMS
        import torch
        max_coord = boxes.max().item() + 1.0
        offsets = classes.float().unsqueeze(1) * max_coord
        return _nms_python(boxes + offsets, scores, iou_threshold)


def xywh_to_xyxy(boxes):
    """
    Convert [center_x, center_y, width, height] → [x1, y1, x2, y2].

    Args:
        boxes: [N, 4] tensor in xywh format

    Returns:
        [N, 4] tensor in xyxy format
    """
    backend = _detect_nms_backend()

    if backend == "edge_nms":
        _ext = _get_edge_nms_module()
        return _ext.xywh_to_xyxy(boxes)

    import torch  # noqa: F811
    x, y, w, h = boxes.unbind(1)
    half_w = w * 0.5
    half_h = h * 0.5
    return torch.stack([x - half_w, y - half_h, x + half_w, y + half_h], dim=1)


# ---------------------------------------------------------------------------
# Pure-Python NMS fallback
# ---------------------------------------------------------------------------

def _nms_python(boxes, scores, iou_threshold: float):
    """Pure-Python greedy NMS. Slow but dependency-free."""
    import torch

    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.int64)

    boxes_np = boxes.cpu().float().numpy()
    scores_np = scores.cpu().float().numpy()

    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores_np.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        xx1 = x1[order[1:]].clip(min=x1[i])
        yy1 = y1[order[1:]].clip(min=y1[i])
        xx2 = x2[order[1:]].clip(max=x2[i])
        yy2 = y2[order[1:]].clip(max=y2[i])

        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
        union = areas[i] + areas[order[1:]] - inter
        ious = inter / union.clip(min=1e-6)

        mask = ious <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.int64)
