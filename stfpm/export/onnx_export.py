from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from stfpm.models.stfpm import STFPM


class STFPMExportWrapper(torch.nn.Module):
    def __init__(self, model: STFPM, image_size: int):
        super().__init__()
        self.model = model
        self.image_size = image_size

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t_feats = self.model.teacher(images)
        s_feats = self.model.student(images)
        score_map = self.model.anomaly_score_map(t_feats, s_feats, out_size=(self.image_size, self.image_size))
        image_score = score_map.flatten(1).max(dim=1).values
        return score_map, image_score


def export_onnx(model: STFPM, config: dict[str, Any], checkpoint_path: str, output_path: str) -> str:
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.student.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    image_size = int(config["dataset"]["image_size"])
    sample = torch.randn(1, 3, image_size, image_size, device=device)
    wrapped = STFPMExportWrapper(model, image_size=image_size).to(device).eval()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_cfg = config["onnx"]
    dynamic = bool(export_cfg["dynamic_batch"])
    dynamic_axes = {"input": {0: "batch"}, "score_map": {0: "batch"}, "image_score": {0: "batch"}} if dynamic else None

    torch.onnx.export(
        wrapped,
        sample,
        str(out_path),
        export_params=True,
        opset_version=int(export_cfg["opset"]),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["score_map", "image_score"],
        dynamic_axes=dynamic_axes,
    )
    return str(out_path)
