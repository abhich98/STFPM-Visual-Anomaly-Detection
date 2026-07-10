from __future__ import annotations

from collections.abc import Callable

import torch
import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights, Wide_ResNet50_2_Weights


def build_resnet18(pretrained: bool) -> torch.nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    return tv_models.resnet18(weights=weights)


def build_wide_resnet50_2(pretrained: bool) -> torch.nn.Module:
    weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
    return tv_models.wide_resnet50_2(weights=weights)


BACKBONES: dict[str, Callable[[bool], torch.nn.Module]] = {
    "resnet18": build_resnet18,
    "wide_resnet50_2": build_wide_resnet50_2,
}
