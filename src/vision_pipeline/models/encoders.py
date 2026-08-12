from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vision_pipeline.exceptions import ConfigurationError


class StatisticalEncoder(nn.Module):
    output_dim = 18

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")

        height_midpoint = images.shape[2] // 2
        width_midpoint = images.shape[3] // 2
        channel_mean = images.mean(dim=(2, 3))
        channel_std = images.std(dim=(2, 3), unbiased=False)
        quadrants = [
            images[:, :, :height_midpoint, :width_midpoint],
            images[:, :, :height_midpoint, width_midpoint:],
            images[:, :, height_midpoint:, :width_midpoint],
            images[:, :, height_midpoint:, width_midpoint:],
        ]
        quadrant_means = [quadrant.mean(dim=(2, 3)) for quadrant in quadrants]
        return torch.cat([channel_mean, channel_std, *quadrant_means], dim=1)


class TorchvisionEncoder(nn.Module):
    def __init__(self, name: str = "mobilenet_v3_small", pretrained: bool = True):
        super().__init__()
        try:
            from torchvision import models
        except ImportError as error:
            raise ConfigurationError(
                "torchvision encoder requires installation with the 'vision' extra"
            ) from error

        model_builders: dict[str, tuple[Any, str]] = {
            "mobilenet_v3_small": (models.mobilenet_v3_small, "MobileNet_V3_Small_Weights"),
            "resnet18": (models.resnet18, "ResNet18_Weights"),
        }
        if name not in model_builders:
            supported = ", ".join(sorted(model_builders))
            raise ConfigurationError(f"Unsupported torchvision encoder '{name}'. Supported: {supported}")

        builder, weights_name = model_builders[name]
        weights = getattr(models, weights_name).DEFAULT if pretrained else None
        model = builder(weights=weights)
        if name == "resnet18":
            self.output_dim = model.fc.in_features
            model.fc = nn.Identity()
        else:
            self.output_dim = model.classifier[0].in_features
            model.classifier = nn.Identity()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)
