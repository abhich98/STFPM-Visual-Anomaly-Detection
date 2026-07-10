from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MVTecSample:
    image_path: Path
    label: int
    mask_path: Path | None


class MVTecEvalDataset(Dataset):
    def __init__(self, samples: list[MVTecSample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        return {
            "path": str(sample.image_path),
            "image": self.transform(image),
            "label": sample.label,
            "mask_path": str(sample.mask_path) if sample.mask_path else "",
        }


def _find_mask_path(root: Path, category: str, defect_type: str, image_stem: str) -> Path | None:
    mask_dir = root / category / "ground_truth" / defect_type
    candidate_a = mask_dir / f"{image_stem}_mask.png"
    candidate_b = mask_dir / f"{image_stem}.png"
    if candidate_a.exists():
        return candidate_a
    if candidate_b.exists():
        return candidate_b
    return None


def collect_mvtec_train_images(root: Path, category: str, extensions: list[str]) -> list[Path]:
    train_dir = root / category / "train" / "good"
    paths: list[Path] = []
    for ext in extensions:
        paths.extend(sorted(train_dir.glob(f"*.{ext}")))
    return sorted(paths)


def collect_mvtec_eval_samples(root: Path, category: str, extensions: list[str]) -> list[MVTecSample]:
    test_dir = root / category / "test"
    samples: list[MVTecSample] = []
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        label = 0 if defect_dir.name == "good" else 1
        for ext in extensions:
            for image_path in sorted(defect_dir.glob(f"*.{ext}")):
                mask_path = None
                if label == 1:
                    mask_path = _find_mask_path(root, category, defect_dir.name, image_path.stem)
                samples.append(MVTecSample(image_path=image_path, label=label, mask_path=mask_path))
    return samples


def load_mask(mask_path: str, image_size: int) -> np.ndarray:
    if not mask_path:
        return np.zeros((image_size, image_size), dtype=bool)
    mask = Image.open(mask_path).convert("L").resize((image_size, image_size))
    return np.array(mask) > 0
