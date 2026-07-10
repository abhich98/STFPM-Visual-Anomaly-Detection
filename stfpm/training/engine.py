from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from stfpm.models.stfpm import STFPM


def _validation_error(model: STFPM, loader, device: torch.device, image_size: int) -> float:
    model.teacher.eval()
    model.student.eval()
    losses = []
    with torch.inference_mode():
        for _, batch_images in loader:
            images = batch_images.to(device, non_blocking=True)
            t_feats = model.teacher(images)
            s_feats = model.student(images)
            score_map = model.anomaly_score_map(t_feats, s_feats, out_size=(image_size // 4, image_size // 4))
            losses.append(score_map.mean().item())
    return float(sum(losses) / max(len(losses), 1))


def train(model: STFPM, train_loader, val_loader, config: dict[str, Any], device: torch.device) -> str:
    train_cfg = config["train"]
    epochs = int(train_cfg["epochs"])
    lr = float(train_cfg["lr"])
    momentum = float(train_cfg["momentum"])
    weight_decay = float(train_cfg["weight_decay"])
    image_size = int(config["dataset"]["image_size"])

    model.to(device)
    model.teacher.eval()
    model.student.train()

    optimizer = torch.optim.SGD(model.student.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    checkpoint_dir = Path(train_cfg["checkpoint_dir"]) / config["dataset"]["category"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pth.tar"

    best_error = float("inf")

    for epoch in range(epochs):
        model.student.train()
        for _, batch_images in train_loader:
            images = batch_images.to(device, non_blocking=True)
            with torch.no_grad():
                t_feats = model.teacher(images)
            s_feats = model.student(images)
            loss = model.training_loss(t_feats, s_feats)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_error = _validation_error(model, val_loader, device, image_size=image_size)
        print(f"[{epoch + 1}/{epochs}] val_error={val_error:.7f}")
        if val_error < best_error:
            best_error = val_error
            checkpoint = {
                "category": config["dataset"]["category"],
                "model": config["model"]["name"],
                "feature_layers": config["model"]["feature_layers"],
                "state_dict": model.student.state_dict(),
            }
            torch.save(checkpoint, best_path)

    return str(best_path)
