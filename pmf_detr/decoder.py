from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, validate_targets


def inverse_sigmoid(values: Tensor, eps: float = 1e-5) -> Tensor:
    values = values.clamp(0, 1)
    return torch.log(values.clamp(min=eps) / (1 - values).clamp(min=eps))


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dimensions[:-1], dimensions[1:]))

    def forward(self, inputs: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            inputs = layer(inputs)
            if index < len(self.layers) - 1:
                inputs = F.relu(inputs)
        return inputs


def multi_scale_deformable_attention(value: Tensor, shapes: list[tuple[int, int]],
                                     locations: Tensor, weights: Tensor) -> Tensor:
    """Grid-sample implementation of multi-scale deformable attention.

    value: B,T,heads,head_dim; locations: B,Q,heads,levels,points,2.
    Spatial shapes are H,W; grid coordinates are x,y in [0,1].
    """
    batch_size, _, num_heads, head_dim = value.shape
    num_queries, num_points = locations.shape[1], locations.shape[-2]
    chunks = value.split([height * width for height, width in shapes], dim=1)
    sampled = []
    grids = 2 * locations - 1
    for level, (height, width) in enumerate(shapes):
        level_value = chunks[level].flatten(2).transpose(1, 2)
        level_value = level_value.reshape(batch_size * num_heads, head_dim, height, width)
        grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(F.grid_sample(level_value, grid, mode="bilinear",
                                     padding_mode="zeros", align_corners=False))
    attention = weights.transpose(1, 2).reshape(batch_size * num_heads, 1, num_queries,
                                               len(shapes) * num_points)
    output = (torch.stack(sampled, dim=-2).flatten(-2) * attention).sum(-1)
    return output.view(batch_size, num_heads * head_dim, num_queries).transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    """Multi-scale deformable cross attention with learned offsets and weights."""

    def __init__(self, hidden_dim: int, num_levels: int = 3,
                 num_heads: int = 8, num_points: int = 4):
        super().__init__()
        if num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by positive num_heads.")
        if num_points <= 0:
            raise ValueError("num_points must be positive.")
        self.hidden_dim, self.num_levels = hidden_dim, num_levels
        self.num_heads, self.num_points = num_heads, num_points
        self.sampling_offsets = nn.Linear(hidden_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(hidden_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.sampling_offsets.weight)
        angles = torch.arange(self.num_heads, dtype=torch.float32) * (2 * math.pi / self.num_heads)
        grid = torch.stack((angles.cos(), angles.sin()), -1)
        grid /= grid.abs().amax(-1, keepdim=True)
        grid = grid[:, None, None, :].repeat(1, self.num_levels, self.num_points, 1)
        grid *= torch.arange(1, self.num_points + 1)[None, None, :, None]
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.flatten())
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        for layer in (self.value_proj, self.output_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, query: Tensor, reference_boxes: Tensor, value: Tensor,
                shapes: list[tuple[int, int]]) -> Tensor:
        batch_size, num_queries, _ = query.shape
        if len(shapes) != self.num_levels or sum(h * w for h, w in shapes) != value.shape[1]:
            raise ValueError("Value sequence and feature-pyramid shapes disagree.")
        value = self.value_proj(value).reshape(batch_size, -1, self.num_heads,
                                                self.hidden_dim // self.num_heads)
        offsets = self.sampling_offsets(query).reshape(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2)
        weights = self.attention_weights(query).reshape(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points)
        weights = weights.softmax(-1).reshape(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points)
        locations = reference_boxes[:, :, None, None, None, :2] + (
            offsets / self.num_points * reference_boxes[:, :, None, None, None, 2:] * 0.5)
        return self.output_proj(multi_scale_deformable_attention(value, shapes, locations, weights))


class DeformableDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_points: int,
                 dim_feedforward: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = MSDeformAttn(hidden_dim, 3, num_heads, num_points)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, embeddings: Tensor, references: Tensor, features: Tensor,
                shapes: list[tuple[int, int]], query_position: Tensor,
                attention_mask: Tensor | None) -> Tensor:
        query = embeddings + query_position
        attended = self.self_attn(query, query, embeddings, attn_mask=attention_mask,
                                  need_weights=False)[0]
        embeddings = self.norm1(embeddings + attended)
        attended = self.cross_attn(embeddings + query_position, references, features, shapes)
        embeddings = self.norm2(embeddings + attended)
        return self.norm3(embeddings + self.linear2(F.relu(self.linear1(embeddings))))


def build_denoising_group(targets: list[dict], class_embed: nn.Embedding,
                          num_queries: int, num_denoising: int,
                          label_noise_ratio: float = 0.5,
                          box_noise_scale: float = 1.0):
    counts = [len(target["labels"]) for target in targets]
    max_count = max(counts, default=0)
    if num_denoising <= 0 or max_count == 0:
        return None, None, None, None
    device = class_embed.weight.device
    dtype = class_embed.weight.dtype
    groups = max(1, num_denoising // max_count)
    num_dn = 2 * max_count * groups
    batch_size = len(targets)
    classes = class_embed.num_embeddings - 1
    padded_labels = torch.full((batch_size, num_dn), classes, dtype=torch.long, device=device)
    padded_boxes = torch.zeros((batch_size, num_dn, 4), dtype=dtype, device=device)
    valid_mask = torch.zeros((batch_size, num_dn), dtype=torch.bool, device=device)
    negative_mask = torch.zeros_like(valid_mask)
    positive_indices, target_indices = [], []
    for batch_index, (count, target) in enumerate(zip(counts, targets)):
        original = torch.arange(count, device=device)
        positives = []
        for group in range(groups):
            positive = group * 2 * max_count + original
            negative = positive + max_count
            for indices in (positive, negative):
                padded_labels[batch_index, indices] = target["labels"].to(device)
                padded_boxes[batch_index, indices] = target["boxes"].to(device=device, dtype=dtype)
                valid_mask[batch_index, indices] = True
            negative_mask[batch_index, negative] = True
            positives.append(positive)
        positive_indices.append(torch.cat(positives))
        target_indices.append(original.repeat(groups))

    if label_noise_ratio > 0:
        corrupted = (torch.rand((batch_size, num_dn), device=device) < label_noise_ratio * 0.5) & valid_mask
        random_labels = torch.randint(classes, (batch_size, num_dn), device=device)
        padded_labels = torch.where(corrupted, random_labels, padded_labels)
    if box_noise_scale > 0:
        xyxy = box_cxcywh_to_xyxy(padded_boxes)
        difference = padded_boxes[..., 2:].repeat(1, 1, 2) * 0.5 * box_noise_scale
        sign = torch.randint(2, xyxy.shape, device=device, dtype=torch.long) * 2 - 1
        magnitude = torch.rand_like(xyxy) + negative_mask[..., None].to(dtype)
        xyxy = (xyxy + sign * magnitude * difference).clamp(0, 1)
        # Perturbed corners can cross. Keep widths/heights valid for reference boxes.
        lower = torch.minimum(xyxy[..., :2], xyxy[..., 2:])
        upper = torch.maximum(xyxy[..., :2], xyxy[..., 2:])
        padded_boxes = box_xyxy_to_cxcywh(torch.cat((lower, upper), dim=-1))

    embeddings = class_embed(padded_labels)
    embeddings = embeddings * valid_mask[..., None]
    box_logits = inverse_sigmoid(padded_boxes)
    box_logits = torch.where(valid_mask[..., None], box_logits, torch.zeros_like(box_logits))
    attention_mask = torch.zeros((num_dn + num_queries, num_dn + num_queries),
                                 dtype=torch.bool, device=device)
    attention_mask[num_dn:, :num_dn] = True
    for group in range(groups):
        start, end = group * 2 * max_count, (group + 1) * 2 * max_count
        attention_mask[start:end, :start] = True
        attention_mask[start:end, end:num_dn] = True
    metadata = {"positive_indices": positive_indices, "target_indices": target_indices,
                "num_groups": groups, "num_denoising": num_dn, "valid_mask": valid_mask}
    return embeddings, box_logits, attention_mask, metadata


class RTDETRDecoder(nn.Module):
    """Three-level RT-DETR decoder returning a stable dictionary interface.

    The feature levels must all have ``hidden_dim`` channels and correspond
    to strides 8, 16 and 32. Outputs contain normalized cxcywh boxes and raw
    class logits. Auxiliary and denoising outputs are training-only.
    """

    def __init__(self, num_classes: int, hidden_dim: int = 256, num_queries: int = 300,
                 num_decoder_layers: int = 3, num_heads: int = 8, num_points: int = 4,
                 dim_feedforward: int = 1024, num_denoising: int = 100,
                 label_noise_ratio: float = 0.5, box_noise_scale: float = 1.0,
                 eval_idx: int = -1):
        super().__init__()
        if min(num_classes, hidden_dim, num_queries, num_decoder_layers, dim_feedforward) <= 0:
            raise ValueError("Class count, widths, query count and decoder layers must be positive.")
        if num_denoising < 0 or not 0 <= label_noise_ratio <= 1 or box_noise_scale < 0:
            raise ValueError("Invalid denoising count, label noise ratio or box noise scale.")
        self.num_classes, self.hidden_dim, self.num_queries = num_classes, hidden_dim, num_queries
        self.num_decoder_layers = num_decoder_layers
        self.num_denoising, self.label_noise_ratio = num_denoising, label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.eval_idx = eval_idx if eval_idx >= 0 else num_decoder_layers + eval_idx
        if not 0 <= self.eval_idx < num_decoder_layers:
            raise ValueError("eval_idx must select an existing decoder layer.")
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                                                      nn.BatchNorm2d(hidden_dim)) for _ in range(3))
        layer = DeformableDecoderLayer(hidden_dim, num_heads, num_points, dim_feedforward)
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_decoder_layers))
        self.query_position = MLP(4, 2 * hidden_dim, hidden_dim, 2)
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.dec_score_head = nn.ModuleList(nn.Linear(hidden_dim, num_classes) for _ in self.layers)
        self.dec_bbox_head = nn.ModuleList(MLP(hidden_dim, hidden_dim, 4, 3) for _ in self.layers)
        self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Preserve the classification-prior initialization from the supplied base.
        class_bias = -math.log((1 - 0.01) / 0.01) / 80 * self.num_classes
        for head in (self.enc_score_head, *self.dec_score_head):
            nn.init.constant_(head.bias, class_bias)
        for head in (self.enc_bbox_head, *self.dec_bbox_head):
            nn.init.zeros_(head.layers[-1].weight)
            nn.init.zeros_(head.layers[-1].bias)
        nn.init.xavier_uniform_(self.enc_output[0].weight)
        for layer in self.query_position.layers:
            nn.init.xavier_uniform_(layer.weight)
        for projection in self.input_proj:
            nn.init.xavier_uniform_(projection[0].weight)

    @staticmethod
    def generate_anchors(shapes: list[tuple[int, int]], device: torch.device,
                         dtype: torch.dtype, grid_size: float = 0.05,
                         eps: float = 1e-2) -> tuple[Tensor, Tensor]:
        anchors = []
        for level, (height, width) in enumerate(shapes):
            yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                                    torch.arange(width, device=device, dtype=dtype), indexing="ij")
            xy = torch.stack((xx, yy), dim=-1)
            xy = (xy + 0.5) / xy.new_tensor([width, height])
            wh = torch.full_like(xy, grid_size * 2 ** level)
            anchors.append(torch.cat((xy, wh), -1).reshape(1, height * width, 4))
        anchors = torch.cat(anchors, dim=1)
        valid = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
        logits = inverse_sigmoid(anchors).masked_fill(~valid, float("inf"))
        return logits, valid

    def forward(self, features: list[Tensor], targets: list[dict] | None = None) -> dict:
        if len(features) != 3:
            raise ValueError("RTDETRDecoder requires exactly three feature levels (P3, P4, P5).")
        batch_size = features[0].shape[0]
        if any(feature.ndim != 4 or feature.shape[:2] != (batch_size, self.hidden_dim)
               for feature in features):
            raise ValueError(f"All feature levels must be [B, {self.hidden_dim}, H, W].")
        if targets is not None:
            validate_targets(targets, batch_size, self.num_classes)
        shapes = [tuple(feature.shape[-2:]) for feature in features]
        token_count = sum(height * width for height, width in shapes)
        if token_count < self.num_queries:
            raise ValueError(f"Feature pyramid has {token_count} tokens, fewer than num_queries={self.num_queries}. "
                             "Increase image size or use fewer queries for a smoke test.")
        projected = [projection(feature) for projection, feature in zip(self.input_proj, features)]
        memory = torch.cat([feature.flatten(2).transpose(1, 2) for feature in projected], dim=1)
        anchors, valid = self.generate_anchors(shapes, memory.device, memory.dtype)
        if int(valid.sum()) < self.num_queries:
            raise ValueError("The feature pyramid has fewer valid anchors than num_queries.")
        encoded = self.enc_output(memory * valid)
        all_logits = self.enc_score_head(encoded)
        rankings = all_logits.amax(-1).masked_fill(~valid.squeeze(-1), float("-inf"))
        selected = rankings.topk(self.num_queries, dim=1).indices
        selected_features = encoded.gather(1, selected[..., None].expand(-1, -1, self.hidden_dim))
        selected_anchors = anchors.expand(batch_size, -1, -1).gather(1, selected[..., None].expand(-1, -1, 4))
        reference_logits = self.enc_bbox_head(selected_features) + selected_anchors
        encoder_outputs = {"pred_logits": all_logits.gather(
            1, selected[..., None].expand(-1, -1, self.num_classes)),
            "pred_boxes": reference_logits.sigmoid()}
        embeddings = selected_features.detach() if self.training else selected_features
        reference_logits = reference_logits.detach() if self.training else reference_logits
        dn_embeddings = dn_references = attention_mask = dn_meta = None
        if self.training and targets is not None:
            dn_embeddings, dn_references, attention_mask, dn_meta = build_denoising_group(
                targets, self.denoising_class_embed, self.num_queries, self.num_denoising,
                self.label_noise_ratio, self.box_noise_scale)
        if dn_meta is not None:
            embeddings = torch.cat((dn_embeddings, embeddings), dim=1)
            reference_logits = torch.cat((dn_references, reference_logits), dim=1)
        references = reference_logits.sigmoid()
        outputs, last_refined = [], None
        for index, layer in enumerate(self.layers):
            embeddings = layer(embeddings, references, memory, shapes,
                               self.query_position(references), attention_mask)
            delta = self.dec_bbox_head[index](embeddings)
            refined = (delta + inverse_sigmoid(references)).sigmoid()
            # Match the base RT-DETR's refinement-gradient path in training.
            boxes = ((delta + inverse_sigmoid(last_refined)).sigmoid()
                     if self.training and last_refined is not None else refined)
            outputs.append({"pred_logits": self.dec_score_head[index](embeddings), "pred_boxes": boxes})
            if not self.training and index == self.eval_idx:
                break
            last_refined = refined
            references = refined.detach() if self.training else refined
        dn_outputs = []
        if dn_meta is not None:
            num_dn = dn_meta["num_denoising"]
            dn_outputs = [{key: value[:, :num_dn] for key, value in output.items()} for output in outputs]
            outputs = [{key: value[:, num_dn:] for key, value in output.items()} for output in outputs]
        return {**outputs[-1], "aux_outputs": outputs[:-1] if self.training else [],
                "enc_outputs": encoder_outputs, "dn_outputs": dn_outputs, "dn_meta": dn_meta}
