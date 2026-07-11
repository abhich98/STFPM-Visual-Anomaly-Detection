from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from .common import ImagePathDataset, build_image_transform
from .mvtec import MVTecEvalDataset, collect_mvtec_eval_samples, collect_mvtec_train_images


def _split_train_val(paths: list[Path], val_split: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not paths:
        return [], []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths))
    rng.shuffle(indices)
    split = int(len(paths) * (1.0 - val_split))
    split = max(1, min(split, len(paths) - 1)) if len(paths) > 1 else 1
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_paths = [paths[int(i)] for i in train_indices]
    val_paths = [paths[int(i)] for i in val_indices] if len(val_indices) > 0 else [paths[int(train_indices[-1])]]
    return train_paths, val_paths


def build_dataloaders(config: dict[str, Any], split: str) -> dict[str, DataLoader]:
    dataset_cfg = config["dataset"]
    name = dataset_cfg["name"].lower()

    if name != "mvtec":
        raise NotImplementedError(
            f"Dataset '{name}' is not implemented yet. Add a new builder in stfpm/data for MIIC or others."
        )
    return build_mvtec_dataloaders(config, split)


def build_mvtec_dataloaders(config: dict[str, Any], split: str) -> dict[str, DataLoader]:
    dataset_cfg = config["dataset"]
    root = Path(dataset_cfg["root"])
    category = dataset_cfg["category"]
    extensions = dataset_cfg["extensions"]
    workers = int(dataset_cfg["num_workers"])
    pin_memory = bool(dataset_cfg["pin_memory"])

    image_size = int(dataset_cfg["image_size"])
    transform = build_image_transform(image_size)

    if split == "train":
        train_batch_size = int(config["train"]["batch_size"])
        image_paths = collect_mvtec_train_images(root, category, extensions)
        train_paths, val_paths = _split_train_val(image_paths, float(config["train"]["val_split"]), int(config["seed"]))
        train_ds = ImagePathDataset(train_paths, transform)
        val_ds = ImagePathDataset(val_paths, transform)
        return {
            "train": DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, drop_last=False, num_workers=workers, pin_memory=pin_memory),
            "val": DataLoader(val_ds, batch_size=train_batch_size, shuffle=False, drop_last=False, num_workers=workers, pin_memory=pin_memory),
        }

    if split == "test":
        eval_batch_size = int(config["eval"]["batch_size"])
        samples = collect_mvtec_eval_samples(root, category, extensions)
        eval_ds = MVTecEvalDataset(samples, transform)
        return {
            "test": DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False, num_workers=workers, pin_memory=pin_memory)
        }

    raise ValueError(f"Unsupported split '{split}'. Use 'train' or 'test'.")
