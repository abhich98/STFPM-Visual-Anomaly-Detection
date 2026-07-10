from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform: transforms.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        return str(path), self.transform(image)
