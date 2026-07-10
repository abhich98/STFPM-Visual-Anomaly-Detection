from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms


def preprocess_image(image_path: str, image_size: int) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor.numpy().astype(np.float32)


def run_onnx_inference(onnx_path: str, image_path: str, image_size: int) -> dict[str, np.ndarray | float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for ONNX inference. Install it with `pip install onnxruntime`.") from exc

    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_tensor = preprocess_image(image_path, image_size=image_size)
    score_map, image_score = session.run(["score_map", "image_score"], {"input": input_tensor})
    return {"score_map": score_map, "image_score": float(image_score[0])}
