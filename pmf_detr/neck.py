from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .modules import ConvBNAct, MBConv, SADCA, SemanticDecoupling


class AIFI(nn.Module):
    """Attention-based intra-scale feature interaction on the coarsest map."""

    def __init__(self, channels: int, num_heads: int = 8, dim_feedforward: int = 1024):
        super().__init__()
        if channels % 4 or channels % num_heads:
            raise ValueError("AIFI channels must be divisible by 4 and num_heads")
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, dim_feedforward), nn.GELU(),
                                 nn.Linear(dim_feedforward, channels))

    @staticmethod
    def position_encoding(height: int, width: int, channels: int,
                          device: torch.device, dtype: torch.dtype) -> Tensor:
        quarter = channels // 4
        omega = torch.arange(quarter, device=device, dtype=torch.float32) / quarter
        omega = 10000.0 ** (-omega)
        y, x = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                              torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
        x, y = x.reshape(-1, 1) * omega, y.reshape(-1, 1) * omega
        return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1).unsqueeze(0).to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        pos = self.position_encoding(height, width, channels, x.device, x.dtype)
        qk = tokens + pos
        tokens = self.norm1(tokens + self.attention(qk, qk, tokens, need_weights=False)[0])
        tokens = self.norm2(tokens + self.ffn(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class RepBlock(nn.Module):
    """Sum of 3x3 and 1x1 Conv-BN branches, used by the RT-DETR CSP neck."""

    def __init__(self, channels: int):
        super().__init__()
        self.spatial = ConvBNAct(channels, channels, 3, activation=None)
        self.pointwise = ConvBNAct(channels, channels, 1, activation=None)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.spatial(x) + self.pointwise(x))


class RepCSP(nn.Module):
    """Compact CSP fusion following the supplied RT-DETR RepC3 structure."""

    def __init__(self, in_channels: int, out_channels: int, depth: int = 3):
        super().__init__()
        hidden = max(1, out_channels // 2)
        self.path = nn.Sequential(ConvBNAct(in_channels, hidden),
                                  *[RepBlock(hidden) for _ in range(depth)])
        self.shortcut = ConvBNAct(in_channels, hidden)
        self.output = ConvBNAct(hidden, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.output(self.path(x) + self.shortcut(x))


class HybridEncoder(nn.Module):
    """Align P2-P5 for SADCA, restore native strides, then run FPN/PAN.

    P2 participates in both the common semantic descriptor and the bottom-up
    path. Decoder outputs use P3/P4/P5. Keeping this topology for all ablations
    isolates module toggles; the all-disabled variant is a controlled baseline,
    not a claim to be byte-identical to the uploaded three-scale R18 YAML.
    """

    def __init__(self, in_channels=(64, 128, 256, 512), hidden_dim: int = 256,
                 num_heads: int = 8, dim_feedforward: int = 1024,
                 sadca: bool = True, mbconv: bool = True,
                 sadca_kernel_size: int = 3, sadca_reduction: int = 16,
                 sadca_spatial_kernel: int = 7, mbconv_kernel_size: int = 5,
                 mbconv_branch_ratio: float = 0.25, mbconv_diagonal_scale: float = 2.0,
                 fpn_depth: int = 3):
        super().__init__()
        if len(in_channels) != 4 or fpn_depth < 1:
            raise ValueError("HybridEncoder requires four levels and fpn_depth >= 1")
        self.projections = nn.ModuleList(
            ConvBNAct(channels, hidden_dim, 1, activation=None) for channels in in_channels)
        self.use_sadca = sadca
        if sadca:
            self.semantic = SemanticDecoupling(hidden_dim)
            self.aggregation = nn.ModuleList(
                SADCA(hidden_dim, sadca_kernel_size, sadca_reduction, sadca_spatial_kernel)
                for _ in in_channels)
        self.aifi = AIFI(hidden_dim, num_heads, dim_feedforward)
        self.lateral = nn.ModuleList(ConvBNAct(hidden_dim, hidden_dim) for _ in range(3))
        self.upsample_branches = nn.ModuleList(
            MBConv(hidden_dim, kernel_size=mbconv_kernel_size,
                   branch_ratio=mbconv_branch_ratio, diagonal_scale=mbconv_diagonal_scale)
            if mbconv else nn.Identity() for _ in range(3))
        self.top_down = nn.ModuleList(RepCSP(2 * hidden_dim, hidden_dim, fpn_depth) for _ in range(3))
        self.downsample = nn.ModuleList(ConvBNAct(hidden_dim, hidden_dim, 3, 2) for _ in range(3))
        self.bottom_up = nn.ModuleList(RepCSP(2 * hidden_dim, hidden_dim, fpn_depth) for _ in range(3))

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        if len(features) != 4:
            raise ValueError("Expected backbone P2, P3, P4, P5 features")
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        if self.use_sadca:
            size = projected[0].shape[-2:]
            aligned = [x if x.shape[-2:] == size else
                       F.interpolate(x, size=size, mode="bilinear", align_corners=False)
                       for x in projected]
            # Sum avoids materializing an extra 4*C-channel concatenation.
            context = (aligned[0] + aligned[1] + aligned[2] + aligned[3]) / 4
            foreground, background = self.semantic(context)
            restored = []
            for original, aligned_map, aggregation in zip(projected, aligned, self.aggregation):
                refined = aggregation(aligned_map, foreground, background)
                restored.append(refined if refined.shape[-2:] == original.shape[-2:] else
                                F.interpolate(refined, size=original.shape[-2:], mode="bilinear",
                                              align_corners=False))
            projected = restored
        projected[-1] = self.aifi(projected[-1])
        fused = [projected[-1]]
        for step, index in enumerate((2, 1, 0)):
            lateral = self.lateral[step](fused[0])
            fused[0] = lateral
            up = F.interpolate(lateral, size=projected[index].shape[-2:], mode="nearest")
            up = self.upsample_branches[step](up)
            fused.insert(0, self.top_down[step](torch.cat((up, projected[index]), dim=1)))
        outputs = [fused[0]]
        for index in range(3):
            down = self.downsample[index](outputs[-1])
            # Supports odd-sized inputs; the normal data path uses multiples of 32.
            if down.shape[-2:] != fused[index + 1].shape[-2:]:
                down = F.interpolate(down, size=fused[index + 1].shape[-2:], mode="nearest")
            outputs.append(self.bottom_up[index](torch.cat((down, fused[index + 1]), dim=1)))
        return outputs[1:]
