import numpy as np
import torch


def _cxcywh_to_xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), dim=-1)


@torch.no_grad()
def postprocess(outputs, max_dets=300, score_threshold=0.001):
    """Select top query/class pairs, without NMS, and return normalized xyxy boxes."""
    if not 0 <= score_threshold <= 1 or max_dets < 1:
        raise ValueError("score_threshold must be in [0,1] and max_dets must be positive.")
    logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
    if logits.ndim != 3 or boxes.shape != (*logits.shape[:2], 4):
        raise ValueError("Invalid detector output shapes.")
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
        raise FloatingPointError("Detector produced non-finite scores or boxes.")
    scores = logits.sigmoid().flatten(1)
    scores, indices = scores.topk(min(max_dets, scores.shape[1]), dim=1)
    num_classes = logits.shape[-1]
    labels, query_ids = indices % num_classes, indices // num_classes
    xyxy = _cxcywh_to_xyxy(boxes).clamp(0, 1)
    selected = xyxy.gather(1, query_ids.unsqueeze(-1).expand(-1, -1, 4))
    predictions = []
    for image_boxes, image_scores, image_labels in zip(selected, scores, labels):
        keep = ((image_scores >= score_threshold) & (image_boxes[:, 2] > image_boxes[:, 0])
                & (image_boxes[:, 3] > image_boxes[:, 1]))
        predictions.append({"boxes": image_boxes[keep].cpu(), "scores": image_scores[keep].cpu(),
                            "labels": image_labels[keep].cpu()})
    return predictions


def _pairwise_iou(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    low = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    high = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = np.prod(np.clip(high - low, 0, None), axis=-1)
    areas1 = np.prod(np.clip(boxes1[:, 2:] - boxes1[:, :2], 0, None), axis=-1)
    areas2 = np.prod(np.clip(boxes2[:, 2:] - boxes2[:, :2], 0, None), axis=-1)
    return intersection / np.maximum(areas1[:, None] + areas2[None, :] - intersection, 1e-12)


def _interpolated_ap(true_positives, num_gt):
    """COCO-style mean over the 101 recall points; all detections must be score sorted."""
    if num_gt == 0 or len(true_positives) == 0:
        return 0.0
    tp = np.cumsum(true_positives, dtype=np.float64)
    fp = np.cumsum(~true_positives, dtype=np.float64)
    recall = tp / num_gt
    precision = tp / np.maximum(tp + fp, 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    positions = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    values = np.zeros(101, dtype=np.float64)
    valid = positions < len(precision)
    values[valid] = precision[positions[valid]]
    return float(values.mean())


class DetectionEvaluator:
    """All-size, max-detections-limited AP. No crowd/ignore or area-specific COCO modes."""

    def __init__(self, num_classes, conf=0.25, names=None):
        if num_classes < 1 or not 0 <= conf <= 1:
            raise ValueError("num_classes must be positive and conf must lie in [0,1].")
        self.num_classes, self.conf = num_classes, conf
        self.names = names or [str(i) for i in range(num_classes)]
        if len(self.names) != num_classes or len(set(self.names)) != num_classes:
            raise ValueError("names must contain one unique name for each class.")
        self.iou_thresholds = np.linspace(0.5, 0.95, 10)
        self.num_gt = np.zeros(num_classes, dtype=np.int64)
        self.records = []
        self.num_images = 0

    def update(self, predictions, targets):
        if len(predictions) != len(targets):
            raise ValueError("Prediction and target batches have different sizes.")
        for prediction, target in zip(predictions, targets):
            self.num_images += 1
            pred_boxes = prediction["boxes"].detach().cpu().numpy().astype(np.float64)
            pred_scores = prediction["scores"].detach().cpu().numpy().astype(np.float64)
            pred_labels = prediction["labels"].detach().cpu().numpy().astype(np.int64)
            gt_boxes = _cxcywh_to_xyxy(target["boxes"].detach().cpu()).numpy().astype(np.float64)
            gt_labels = target["labels"].detach().cpu().numpy().astype(np.int64)
            if ((gt_labels < 0) | (gt_labels >= self.num_classes)).any() or ((pred_labels < 0) | (pred_labels >= self.num_classes)).any():
                raise ValueError("Class ID outside the configured label set.")
            self.num_gt += np.bincount(gt_labels, minlength=self.num_classes)
            order = np.argsort(-pred_scores, kind="stable")
            pred_boxes, pred_scores, pred_labels = pred_boxes[order], pred_scores[order], pred_labels[order]
            matches = np.zeros((len(pred_scores), len(self.iou_thresholds)), dtype=bool)
            ious = _pairwise_iou(pred_boxes, gt_boxes)
            for threshold_index, threshold in enumerate(self.iou_thresholds):
                used = np.zeros(len(gt_labels), dtype=bool)
                for index, label in enumerate(pred_labels):
                    candidates = np.flatnonzero((gt_labels == label) & ~used & (ious[index] >= threshold - 1e-9))
                    if len(candidates):
                        best = candidates[np.argmax(ious[index, candidates])]
                        used[best] = True
                        matches[index, threshold_index] = True
            self.records.append((pred_scores, pred_labels, matches))

    def compute(self):
        if self.records:
            scores = np.concatenate([r[0] for r in self.records])
            labels = np.concatenate([r[1] for r in self.records])
            correct = np.concatenate([r[2] for r in self.records])
        else:
            scores = np.zeros(0)
            labels = np.zeros(0, dtype=np.int64)
            correct = np.zeros((0, 10), dtype=bool)
        order = np.argsort(-scores, kind="stable")
        scores, labels, correct = scores[order], labels[order], correct[order]
        aps, per_class = [], {}
        for cls in range(self.num_classes):
            mask = labels == cls
            row = [_interpolated_ap(correct[mask, t], int(self.num_gt[cls])) for t in range(10)]
            if self.num_gt[cls]:
                aps.append(row)
            per_class[self.names[cls]] = {"ground_truth": int(self.num_gt[cls]),
                "ap50": row[0] if self.num_gt[cls] else None,
                "ap50_95": float(np.mean(row)) if self.num_gt[cls] else None}
        operating = scores >= self.conf
        tp = int(correct[operating, 0].sum())
        fp = int(operating.sum()) - tp
        fn = int(self.num_gt.sum()) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return {"images": self.num_images, "ground_truth": int(self.num_gt.sum()),
                "map50": float(np.mean(np.asarray(aps)[:, 0])) if aps else 0.0,
                "map50_95": float(np.mean(aps)) if aps else 0.0,
                "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                "tp": tp, "fp": fp, "fn": fn, "conf": self.conf,
                "iou_for_pr": 0.5, "ap_recall_points": 101, "per_class": per_class}
