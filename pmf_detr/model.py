"""PMF-DETR construction and forward contract."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn

from .backbone import MEEAResNet18
from .decoder import RTDETRDecoder
from .neck import HybridEncoder


DEFAULT_MODEL_CONFIG = {
    "num_classes": 1, "input_channels": 3, "hidden_dim": 256,
    "num_queries": 300, "num_decoder_layers": 3, "num_heads": 8,
    "num_points": 4, "dim_feedforward": 1024, "num_denoising": 100,
    "label_noise_ratio": 0.5, "box_noise_scale": 1.0, "eval_idx": -1,
    "meea": True, "sadca": True, "mbconv": True,
    "meea_dilations": [1, 2, 5], "meea_sigma": 1.0,
    "meea_wavelength": 2.0, "meea_gamma": 0.5,
    "sadca_kernel_size": 3, "sadca_reduction": 16, "sadca_spatial_kernel": 7,
    "mbconv_kernel_size": 5, "mbconv_branch_ratio": 0.25,
    "mbconv_diagonal_scale": 2.0, "fpn_depth": 3,
}


class PMFDETR(nn.Module):
    def __init__(self, **kwargs: Any):
        super().__init__()
        unknown = set(kwargs) - set(DEFAULT_MODEL_CONFIG)
        if unknown:
            raise ValueError(f"Unknown model configuration keys: {sorted(unknown)}")
        config = deepcopy(DEFAULT_MODEL_CONFIG)
        config.update(deepcopy(kwargs))
        for name in ("meea", "sadca", "mbconv"):
            if not isinstance(config[name], bool):
                raise ValueError(f"{name} must be a YAML boolean, not {config[name]!r}")
        if config["num_classes"] < 1 or config["input_channels"] < 1:
            raise ValueError("num_classes and input_channels must be positive")
        self.model_config = config
        self.backbone = MEEAResNet18(**{k: config[k] for k in (
            "input_channels", "meea", "meea_dilations", "meea_sigma", "meea_wavelength", "meea_gamma")})
        self.neck = HybridEncoder(in_channels=self.backbone.out_channels, **{k: config[k] for k in (
            "hidden_dim", "num_heads", "dim_feedforward", "sadca", "mbconv", "sadca_kernel_size",
            "sadca_reduction", "sadca_spatial_kernel", "mbconv_kernel_size", "mbconv_branch_ratio",
            "mbconv_diagonal_scale", "fpn_depth")})
        self.decoder = RTDETRDecoder(**{k: config[k] for k in (
            "num_classes", "hidden_dim", "num_queries", "num_decoder_layers", "num_heads",
            "num_points", "dim_feedforward", "num_denoising", "label_noise_ratio", "box_noise_scale", "eval_idx")})

    def forward(self, images: Tensor, targets: list[dict[str, Tensor]] | None = None) -> dict:
        if images.ndim != 4 or images.shape[1] != self.model_config["input_channels"]:
            raise ValueError("images must have shape [batch, input_channels, height, width]")
        if min(images.shape[-2:]) < 32:
            raise ValueError("Input spatial dimensions must each be at least 32 pixels")
        return self.decoder(self.neck(self.backbone(images)), targets)


def build_model(config: dict[str, Any] | None = None) -> PMFDETR:
    """Accept a complete experiment YAML or just its ``model`` dictionary."""
    if config is None:
        return PMFDETR()
    model_config = config.get("model", config)
    if not isinstance(model_config, dict):
        raise ValueError("The model configuration must be a mapping")
    return PMFDETR(**model_config)
