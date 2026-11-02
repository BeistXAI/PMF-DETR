"""Bounding-box geometry, with no torchvision dependency.

IoU functions take ``xyxy`` boxes. Detector outputs and regression loss helpers
take normalized ``cxcywh`` boxes, as stated in their docstrings.
"""

from __future__ import annotations

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    lower, upper = boxes[..., :2], boxes[..., 2:]
    return torch.cat(((lower + upper) / 2, upper - lower), dim=-1)


# Explicit aliases are convenient for dataset and inference code.
cxcywh_to_xyxy = box_cxcywh_to_xyxy
xyxy_to_cxcywh = box_xyxy_to_cxcywh


def box_area(boxes: Tensor) -> Tensor:
    size = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
    return size[..., 0] * size[..., 1]


def _pairwise_iou_union(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    lower = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    upper = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    size = (upper - lower).clamp(min=0)
    intersection = size[..., 0] * size[..., 1]
    union = box_area(boxes1)[:, None] + box_area(boxes2)[None] - intersection

    epsilon = torch.finfo(union.dtype).tiny
    return intersection / union.clamp(min=epsilon), union


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise IoU matrix for ``[N,4]`` and ``[M,4]`` xyxy boxes."""
    return _pairwise_iou_union(boxes1, boxes2)[0]


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise generalized IoU, including gradients for disjoint boxes."""
    iou, union = _pairwise_iou_union(boxes1, boxes2)
    lower = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    upper = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    size = (upper - lower).clamp(min=0)
    enclosing = size[..., 0] * size[..., 1]
    return iou - (enclosing - union) / enclosing.clamp(min=torch.finfo(enclosing.dtype).tiny)


def aligned_iou_and_giou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """IoU and GIoU for corresponding xyxy pairs, without an N-by-N matrix."""
    if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError("Aligned boxes must have the same [N, 4] shape.")
    lower = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    upper = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    size = (upper - lower).clamp(min=0)
    intersection = size[:, 0] * size[:, 1]
    union = box_area(boxes1) + box_area(boxes2) - intersection
    eps = torch.finfo(union.dtype).tiny
    iou = intersection / union.clamp(min=eps)
    enclosing_size = (torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
                      - torch.minimum(boxes1[:, :2], boxes2[:, :2])).clamp(min=0)
    enclosing = enclosing_size[:, 0] * enclosing_size[:, 1]
    giou = iou - (enclosing - union) / enclosing.clamp(min=eps)
    return iou, giou


def focaler_giou_loss(pred_boxes: Tensor, target_boxes: Tensor,
                     lower: float = 0.0, upper: float = 0.95) -> Tensor:
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("Focaler interval must satisfy 0 <= lower < upper <= 1.")
    iou, giou = aligned_iou_and_giou(box_cxcywh_to_xyxy(pred_boxes),
                                   box_cxcywh_to_xyxy(target_boxes))
    mapped_iou = ((iou - lower) / (upper - lower)).clamp(0, 1)
    return 1 - giou + iou - mapped_iou


def normalized_wasserstein_loss(pred_boxes: Tensor, target_boxes: Tensor,
                               image_size: tuple[int, int] | int = 640,
                               constant: float = 12.8) -> Tensor:
    if constant <= 0:
        raise ValueError("NWD constant must be positive (pixel units).")
    height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
    if height <= 0 or width <= 0:
        raise ValueError("NWD image dimensions must be positive.")
    scale = pred_boxes.new_tensor([width, height, width / 2, height / 2])
    distance = torch.linalg.vector_norm((pred_boxes - target_boxes) * scale, dim=-1)
    return -torch.expm1(-distance / constant)


def validate_targets(targets: list[dict], batch_size: int, num_classes: int) -> None:
    if not isinstance(targets, (list, tuple)) or len(targets) != batch_size:
        raise ValueError(f"Expected a target list of length {batch_size}.")
    for index, target in enumerate(targets):
        if not isinstance(target, dict) or not {"labels", "boxes"} <= target.keys():
            raise ValueError(f"Target {index} must contain labels and boxes.")
        labels, boxes = target["labels"], target["boxes"]
        if not isinstance(labels, Tensor) or not isinstance(boxes, Tensor):
            raise ValueError(f"Target {index}: labels and boxes must be tensors.")
        if labels.ndim != 1 or labels.dtype != torch.long:
            raise ValueError(f"Target {index}: labels must be an int64 tensor of shape [N].")
        if boxes.shape != (len(labels), 4) or not boxes.is_floating_point():
            raise ValueError(f"Target {index}: boxes must be floating-point [N, 4] cxcywh.")
        if not torch.isfinite(boxes).all():
            raise ValueError(f"Target {index}: boxes contain nonfinite coordinates.")
        if labels.numel() and ((labels < 0).any() or (labels >= num_classes).any()):
            raise ValueError(f"Target {index}: class IDs must be in [0, {num_classes - 1}].")
        if boxes.numel() and ((boxes < 0).any() or (boxes > 1).any() or (boxes[:, 2:] <= 0).any()):
            raise ValueError(f"Target {index}: boxes must be normalized cxcywh with positive width/height.")
