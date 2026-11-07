from __future__ import annotations

import math

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F

from .box_ops import (aligned_iou_and_giou, box_cxcywh_to_xyxy,
                      focaler_giou_loss, generalized_box_iou,
                      normalized_wasserstein_loss, validate_targets)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0,
                 cost_giou: float = 2.0, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        costs = (cost_class, cost_bbox, cost_giou)
        if not all(math.isfinite(value) and value >= 0 for value in costs) or not any(costs):
            raise ValueError("Matching costs must be finite, nonnegative and not all zero.")
        if not 0 <= alpha <= 1 or gamma < 0:
            raise ValueError("Invalid focal matching alpha/gamma.")
        self.cost_class, self.cost_bbox, self.cost_giou = costs
        self.alpha, self.gamma = alpha, gamma

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]) -> list[tuple[Tensor, Tensor]]:
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        _validate_outputs(outputs)
        validate_targets(targets, len(boxes), logits.shape[-1])
        assignments = []
        for image_index, target in enumerate(targets):
            device = logits.device
            if len(target["labels"]) == 0:
                empty = torch.empty(0, dtype=torch.long, device=device)
                assignments.append((empty, empty))
                continue
            target_labels = target["labels"].to(device=device)
            target_boxes = target["boxes"].to(device=device, dtype=torch.float32)
            probabilities = logits[image_index].float().sigmoid()
            positive = (self.alpha * (1 - probabilities).pow(self.gamma)
                        * -(probabilities + 1e-8).log())
            negative = ((1 - self.alpha) * probabilities.pow(self.gamma)
                        * -(1 - probabilities + 1e-8).log())
            cost_class = (positive - negative)[:, target_labels]
            pred_boxes = boxes[image_index].float()
            cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)
            cost_giou = 1 - generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes),
                                               box_cxcywh_to_xyxy(target_boxes))
            cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            if not torch.isfinite(cost).all():
                raise FloatingPointError("Hungarian costs contain NaN/Inf; inspect model outputs and learning rate.")
            rows, columns = linear_sum_assignment(cost.cpu().numpy())
            assignments.append((torch.as_tensor(rows, dtype=torch.long, device=device),
                                torch.as_tensor(columns, dtype=torch.long, device=device)))
        return assignments


def _validate_outputs(outputs: dict) -> None:
    if not isinstance(outputs, dict) or not {"pred_logits", "pred_boxes"} <= outputs.keys():
        raise ValueError("Outputs must contain pred_logits and pred_boxes.")
    logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
    if (logits.ndim != 3 or boxes.shape != (*logits.shape[:2], 4)
            or logits.shape[0] < 1 or logits.shape[1] < 1 or logits.shape[2] < 1):
        raise ValueError("Expected logits [B,Q,C] and boxes [B,Q,4] with positive B,Q,C.")
    if logits.device != boxes.device:
        raise ValueError("Prediction logits and boxes must be on the same device.")
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
        raise FloatingPointError("Model predictions contain NaN or Inf.")


# giou_weight is the localization weight; loss_giou retains the selected loss.
# With box_loss="focaler_giou", both refer to Focaler-GIoU, not an extra GIoU term.
class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: HungarianMatcher | None = None,
                 box_loss: str = "focaler_giou", focaler_lower: float = 0.0,
                 focaler_upper: float = 0.95, class_weight: float = 1.0,
                 bbox_weight: float = 5.0, giou_weight: float = 2.0,
                 class_loss: str = "varifocal", focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0, varifocal_alpha: float = 0.75,
                 nwd_weight: float = 0.0, nwd_constant: float = 12.8,
                 image_size: int | tuple[int, int] = 640,
                 aux_weight: float = 1.0, encoder_weight: float = 1.0,
                 dn_weight: float = 1.0):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if box_loss not in {"giou", "focaler_giou"}:
            raise ValueError("box_loss must be 'giou' or 'focaler_giou'.")
        if class_loss not in {"varifocal", "focal"}:
            raise ValueError("class_loss must be 'varifocal' or 'focal'.")
        if not 0 <= focaler_lower < focaler_upper <= 1:
            raise ValueError("Focaler interval must satisfy 0 <= lower < upper <= 1.")
        if not 0 <= nwd_weight <= 1 or not math.isfinite(nwd_constant) or nwd_constant <= 0:
            raise ValueError("NWD weight must lie in [0,1] and its pixel constant must be positive.")
        if not 0 <= focal_alpha <= 1 or not 0 <= varifocal_alpha <= 1 or focal_gamma < 0:
            raise ValueError("Invalid classification focal alpha/gamma.")
        for name, value in (("class_weight", class_weight), ("bbox_weight", bbox_weight),
                            ("giou_weight", giou_weight), ("aux_weight", aux_weight),
                            ("encoder_weight", encoder_weight), ("dn_weight", dn_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        self.num_classes = num_classes
        self.matcher = matcher if matcher is not None else HungarianMatcher()
        self.box_loss = box_loss
        self.focaler_lower, self.focaler_upper = focaler_lower, focaler_upper
        self.class_weight, self.bbox_weight, self.giou_weight = class_weight, bbox_weight, giou_weight
        self.class_loss = class_loss
        self.focal_alpha, self.focal_gamma, self.varifocal_alpha = focal_alpha, focal_gamma, varifocal_alpha
        self.nwd_weight, self.nwd_constant, self.image_size = nwd_weight, nwd_constant, image_size
        self.aux_weight, self.encoder_weight, self.dn_weight = aux_weight, encoder_weight, dn_weight

    def _losses(self, outputs: dict, targets: list[dict], assignments: list[tuple[Tensor, Tensor]],
                normalizer: float, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        _validate_outputs(outputs)
        logits, boxes = outputs["pred_logits"].float(), outputs["pred_boxes"].float()
        one_hot = torch.zeros_like(logits)
        quality = torch.zeros_like(logits)
        matched_predictions, matched_targets = [], []
        for batch_index, (prediction_indices, target_indices) in enumerate(assignments):
            if prediction_indices.numel() == 0:
                continue
            prediction_indices = prediction_indices.to(logits.device)
            target_indices = target_indices.to(logits.device)
            labels = targets[batch_index]["labels"].to(logits.device)[target_indices]
            target_boxes = targets[batch_index]["boxes"].to(device=logits.device, dtype=boxes.dtype)[target_indices]
            prediction_boxes = boxes[batch_index, prediction_indices]
            pair_iou, _ = aligned_iou_and_giou(box_cxcywh_to_xyxy(prediction_boxes),
                                              box_cxcywh_to_xyxy(target_boxes))
            one_hot[batch_index, prediction_indices, labels] = 1
            quality[batch_index, prediction_indices, labels] = pair_iou.detach().clamp(0, 1)
            matched_predictions.append(prediction_boxes)
            matched_targets.append(target_boxes)

        probabilities = logits.sigmoid()
        if self.class_loss == "varifocal" and matched_predictions:
            weights = (self.varifocal_alpha * probabilities.pow(self.focal_gamma) * (1 - one_hot)
                       + quality)
            classification = F.binary_cross_entropy_with_logits(logits, quality, reduction="none") * weights
        else:
            probability_of_target = probabilities * one_hot + (1 - probabilities) * (1 - one_hot)
            alpha_factor = self.focal_alpha * one_hot + (1 - self.focal_alpha) * (1 - one_hot)
            classification = (F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
                              * (1 - probability_of_target).pow(self.focal_gamma) * alpha_factor)
        if valid_mask is not None:
            if valid_mask.shape != logits.shape[:2]:
                raise ValueError("Denoising valid_mask must match [B, DN_queries].")
            classification = classification * valid_mask.to(device=logits.device)[..., None]
        result = {"loss_class": classification.sum() / normalizer * self.class_weight}

        if matched_predictions:
            pred_boxes, target_boxes = torch.cat(matched_predictions), torch.cat(matched_targets)
            result["loss_bbox"] = F.l1_loss(pred_boxes, target_boxes, reduction="sum") / normalizer * self.bbox_weight
            if self.box_loss == "focaler_giou":
                regression = focaler_giou_loss(pred_boxes, target_boxes, self.focaler_lower, self.focaler_upper)
            else:
                _, giou = aligned_iou_and_giou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes))
                regression = 1 - giou
            result["loss_giou"] = regression.sum() / normalizer * self.giou_weight * (1 - self.nwd_weight)
            if self.nwd_weight:
                nwd = normalized_wasserstein_loss(pred_boxes, target_boxes, self.image_size, self.nwd_constant)
                result["loss_nwd"] = nwd.sum() / normalizer * self.giou_weight * self.nwd_weight
        else:
            zero = boxes.sum() * 0
            result.update(loss_bbox=zero, loss_giou=zero)
            if self.nwd_weight:
                result["loss_nwd"] = zero
        return result

    def forward(self, outputs: dict, targets: list[dict]) -> dict[str, Tensor]:
        _validate_outputs(outputs)
        if outputs["pred_logits"].shape[-1] != self.num_classes:
            raise ValueError("Criterion and detector have different class counts.")
        validate_targets(targets, outputs["pred_logits"].shape[0], self.num_classes)
        normalizer = float(max(1, sum(len(target["labels"]) for target in targets)))
        losses = self._losses(outputs, targets, self.matcher(outputs, targets), normalizer)
        if self.aux_weight:
            for layer, auxiliary in enumerate(outputs.get("aux_outputs", [])):
                auxiliary_losses = self._losses(auxiliary, targets, self.matcher(auxiliary, targets), normalizer)
                losses.update({f"{key}_aux_{layer}": value * self.aux_weight for key, value in auxiliary_losses.items()})
        if self.encoder_weight and outputs.get("enc_outputs") is not None:
            encoder = outputs["enc_outputs"]
            encoder_losses = self._losses(encoder, targets, self.matcher(encoder, targets), normalizer)
            losses.update({f"{key}_enc": value * self.encoder_weight for key, value in encoder_losses.items()})
        dn_outputs, dn_meta = outputs.get("dn_outputs", []), outputs.get("dn_meta")
        if bool(dn_outputs) != (dn_meta is not None):
            raise ValueError("dn_outputs and dn_meta must be provided together.")
        if self.dn_weight and dn_meta is not None:
            if dn_meta["num_groups"] < 1:
                raise ValueError("Denoising metadata num_groups must be positive.")
            if len(dn_meta["positive_indices"]) != len(targets) or len(dn_meta["target_indices"]) != len(targets):
                raise ValueError("Denoising assignment metadata must have one entry per image.")
            assignments = list(zip(dn_meta["positive_indices"], dn_meta["target_indices"]))
            for layer, denoising in enumerate(dn_outputs):
                known_losses = self._losses(denoising, targets, assignments,
                                            normalizer * dn_meta["num_groups"], dn_meta["valid_mask"])
                losses.update({f"{key}_dn_{layer}": value * self.dn_weight for key, value in known_losses.items()})
        losses["loss_total"] = sum(losses.values())
        return losses


def build_criterion(config: dict, num_classes: int) -> SetCriterion:
    loss = dict(config.get("loss", {}))
    matcher = HungarianMatcher(cost_class=float(loss.pop("cost_class", 2)),
                                cost_bbox=float(loss.pop("cost_bbox", 5)),
                                cost_giou=float(loss.pop("cost_giou", 2)),
                                alpha=float(loss.get("focal_alpha", 0.25)),
                                gamma=float(loss.get("focal_gamma", 2)))
    allowed = {"box_loss", "focaler_lower", "focaler_upper", "class_weight", "bbox_weight",
               "giou_weight", "class_loss", "focal_alpha", "focal_gamma", "varifocal_alpha",
               "nwd_weight", "nwd_constant", "aux_weight", "encoder_weight", "dn_weight"}
    unknown = set(loss) - allowed
    if unknown:
        raise ValueError(f"Unknown loss configuration options: {', '.join(sorted(unknown))}")
    return SetCriterion(num_classes, matcher=matcher,
                        image_size=config.get("train", {}).get("imgsz", 640), **loss)
