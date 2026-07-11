from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, feature_layers: Iterable[str]):
        super().__init__()
        self.backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
        # self.backbone = backbone
        self.feature_layers = set(feature_layers)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        x = images
        for name, module in self.backbone._modules.items():
            x = module(x)
            if name in self.feature_layers:
                outputs.append(x)
        return outputs


class STFPM(torch.nn.Module):
    def __init__(self, teacher: FeatureExtractor, student: FeatureExtractor):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        with torch.no_grad():
            t_feats = self.teacher(images)
        s_feats = self.student(images)
        return t_feats, s_feats

    @staticmethod
    def training_loss(teacher_feats: list[torch.Tensor], student_feats: list[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=student_feats[0].device)
        for t_feat, s_feat in zip(teacher_feats, student_feats, strict=True):
            t_norm = F.normalize(t_feat, dim=1)
            s_norm = F.normalize(s_feat, dim=1)
            loss = loss + torch.sum((t_norm - s_norm) ** 2, dim=1).mean()
        return loss

    @staticmethod
    def anomaly_score_map(
        teacher_feats: list[torch.Tensor],
        student_feats: list[torch.Tensor],
        out_size: tuple[int, int],
    ) -> torch.Tensor:
        score_map = torch.ones((teacher_feats[0].shape[0], 1, out_size[0], out_size[1]), device=teacher_feats[0].device)
        for t_feat, s_feat in zip(teacher_feats, student_feats, strict=True):
            t_norm = F.normalize(t_feat, dim=1)
            s_norm = F.normalize(s_feat, dim=1)
            smap = torch.sum((t_norm - s_norm) ** 2, dim=1, keepdim=True)
            smap = F.interpolate(smap, size=out_size, mode="bilinear", align_corners=False)
            score_map = score_map * smap
        return score_map
