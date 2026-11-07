from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvBNAct(nn.Sequential):
    """Convolution with shape-preserving padding, BatchNorm and activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size=1,
                 stride=1, activation: str | None = "silu"):
        kernel = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        act = {"silu": nn.SiLU, "relu": nn.ReLU, None: nn.Identity}[activation]()
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel, stride,
                      padding=(kernel[0] // 2, kernel[1] // 2), bias=False),
            nn.BatchNorm2d(out_channels), act,
        )


def _gabor_bank(sigma: float, wavelength: float, gamma: float) -> Tensor:
    """Four zero-mean, L1-normalized analytic 3x3 Gabor filters."""
    if sigma <= 0 or wavelength <= 0 or gamma <= 0:
        raise ValueError("Gabor sigma, wavelength and gamma must be positive")
    axis = torch.arange(-1, 2, dtype=torch.float32)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    kernels = []
    for angle in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        xr = x * math.cos(angle) + y * math.sin(angle)
        yr = -x * math.sin(angle) + y * math.cos(angle)
        kernel = torch.exp(-(xr.square() + gamma**2 * yr.square()) / (2 * sigma**2))
        kernel = kernel * torch.cos(2 * math.pi * xr / wavelength)
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.abs().sum().clamp_min(1e-12)
        kernels.append(kernel)
    return torch.stack(kernels).unsqueeze(1)


# MEEA: Multi-scale Edge-Enhanced Attention.
class MEEA(nn.Module):
    def __init__(self, channels: int, dilations: Sequence[int] = (1, 2, 5),
                 sigma: float = 1.0, wavelength: float = 2.0, gamma: float = 0.5):
        super().__init__()
        if channels < 1 or not dilations or any(int(d) != d or d < 1 for d in dilations):
            raise ValueError("MEEA requires positive channels and integer dilations")
        self.channels = channels
        self.dilations = tuple(int(d) for d in dilations)
        gabor = _gabor_bank(sigma, wavelength, gamma)
        sobel_x = torch.tensor([[-1.0, 0, 1], [-2.0, 0, 2], [-1.0, 0, 1]]) / 4
        sobel = torch.stack((sobel_x, sobel_x.T)).unsqueeze(1)
        self.register_buffer("gabor_kernels", gabor.repeat(channels, 1, 1, 1))
        self.register_buffer("sobel_kernels", sobel.repeat(channels, 1, 1, 1))
        self.gabor_fuse = nn.Conv2d(channels * 4 * len(dilations), channels, 1)
        self.sobel_fuse = nn.Conv2d(channels * 2 * len(dilations), channels, 1)
        self.branch_fuse = nn.Conv2d(channels * 2, channels, 1)
        self.attention = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        gabor = [F.conv2d(x, self.gabor_kernels, padding=d, dilation=d,
                          groups=self.channels) for d in self.dilations]
        sobel = [F.conv2d(x, self.sobel_kernels, padding=d, dilation=d,
                          groups=self.channels) for d in self.dilations]
        features = torch.cat((self.gabor_fuse(torch.cat(gabor, dim=1)),
                              self.sobel_fuse(torch.cat(sobel, dim=1))), dim=1)
        attention = self.attention(self.branch_fuse(features)).sigmoid()
        return x * attention


class SemanticDecoupling(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.context = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(channels, 1, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        context = self.context(x)
        gate = self.gate(context).sigmoid()
        return context * gate, context * (1 - gate)


# SADCA: Self-Adaptive Dual Contrast Aggregation; kernel_size is a fixed window.
class SADCA(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3,
                 reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        if channels < 1 or reduction < 1 or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("SADCA requires positive channels/reduction and an odd kernel")
        if spatial_kernel < 1 or spatial_kernel % 2 == 0:
            raise ValueError("SADCA spatial kernel must be a positive odd integer")
        self.channels = channels
        self.kernel_size = kernel_size
        self.fg_kernel = nn.Linear(channels, channels * kernel_size**2)
        self.bg_kernel = nn.Linear(channels, channels * kernel_size**2)
        self.local_gate = nn.Conv2d(channels * 2, 1, 1)
        hidden = max(1, channels // reduction)
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels), nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2), nn.Sigmoid(),
        )
        for layer in (self.fg_kernel, self.bg_kernel):
            nn.init.normal_(layer.weight, std=0.01)
            with torch.no_grad():
                layer.bias.zero_()
                layer.bias.view(channels, kernel_size**2)[:, kernel_size**2 // 2] = 1

    def dynamic_conv(self, x: Tensor, kernels: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        weights = kernels.reshape(batch * channels, 1, self.kernel_size, self.kernel_size)
        response = F.conv2d(x.reshape(1, batch * channels, height, width), weights,
                            padding=self.kernel_size // 2, groups=batch * channels)
        return response.reshape(batch, channels, height, width)

    def forward(self, x: Tensor, foreground: Tensor, background: Tensor) -> Tensor:
        fg = self.fg_kernel(foreground.mean(dim=(2, 3)))
        bg = self.bg_kernel(background.mean(dim=(2, 3)))
        r_fg, r_bg = self.dynamic_conv(x, fg), self.dynamic_conv(x, bg)
        alpha = self.local_gate(torch.cat((r_fg, r_bg), dim=1)).sigmoid()
        response = alpha * r_fg + (1 - alpha) * r_bg
        channel = self.channel_attention(response.mean(dim=(2, 3)))[:, :, None, None]
        spatial = self.spatial_attention(torch.cat(
            (response.mean(dim=1, keepdim=True), response.amax(dim=1, keepdim=True)), dim=1))
        return response * channel * spatial


# MBConv: Orientation-Sensitive Multi-Branch Convolution.
class MBConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None,
                 kernel_size: int = 5, branch_ratio: float = 0.25,
                 diagonal_scale: float = 2.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        if kernel_size < 1 or kernel_size % 2 == 0 or branch_ratio <= 0 or diagonal_scale < 1:
            raise ValueError("MBConv needs an odd kernel, positive width and diagonal_scale >= 1")
        branch_channels = max(1, round(out_channels * branch_ratio))
        self.horizontal = ConvBNAct(in_channels, branch_channels, (1, kernel_size))
        self.vertical = ConvBNAct(in_channels, branch_channels, (kernel_size, 1))
        self.diagonal = ConvBNAct(in_channels, branch_channels, 3)
        self.antidiagonal = ConvBNAct(in_channels, branch_channels, 3)
        self.fuse = ConvBNAct(branch_channels * 4, out_channels, 1)
        for branch, mask in ((self.diagonal, torch.eye(3)),
                             (self.antidiagonal, torch.eye(3).flip(1))):
            with torch.no_grad():
                branch[0].weight.mul_(1 + (diagonal_scale - 1) * mask)

    def forward(self, x: Tensor) -> Tensor:
        return self.fuse(torch.cat((self.horizontal(x), self.vertical(x),
                                   self.diagonal(x), self.antidiagonal(x)), dim=1))
