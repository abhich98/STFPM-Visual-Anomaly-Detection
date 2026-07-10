from __future__ import annotations

from typing import Any

from .backbones import BACKBONES
from .stfpm import FeatureExtractor, STFPM


def build_stfpm_model(config: dict[str, Any]) -> STFPM:
    model_cfg = config["model"]
    backbone_name = model_cfg["name"]
    if backbone_name not in BACKBONES:
        available = ", ".join(sorted(BACKBONES.keys()))
        raise ValueError(f"Unknown model '{backbone_name}'. Available: {available}")

    feature_layers = model_cfg["feature_layers"]
    factory = BACKBONES[backbone_name]
    teacher_backbone = factory(bool(model_cfg["teacher_pretrained"]))
    student_backbone = factory(bool(model_cfg["student_pretrained"]))

    teacher = FeatureExtractor(teacher_backbone, feature_layers)
    student = FeatureExtractor(student_backbone, feature_layers)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.eval()
    return STFPM(teacher=teacher, student=student)
