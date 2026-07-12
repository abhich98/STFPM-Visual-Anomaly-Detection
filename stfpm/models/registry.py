from __future__ import annotations

from typing import Any

import torch

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


def build_inference_wrapper(config: dict[str, Any], device: torch.device):
    """Build an STFPM model, load the student checkpoint, and wrap it for inference.

    Loads the student weights from ``config["eval"]["checkpoint_path"]``, moves
    the model to ``device``, sets it to eval mode, and returns an
    ``STFPMExportWrapper`` ready for inference / export.

    Returns:
        ``STFPMExportWrapper`` on ``device`` in eval mode.
    """
    # Imported here to avoid a circular import: onnx_export imports registry.
    from stfpm.export.onnx_export import STFPMExportWrapper

    checkpoint_path = config["eval"]["checkpoint_path"]
    image_size = int(config["dataset"]["image_size"])

    model = build_stfpm_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.student.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return STFPMExportWrapper(model, image_size=image_size).to(device).eval()
