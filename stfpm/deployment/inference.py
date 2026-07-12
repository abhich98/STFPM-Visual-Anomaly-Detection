from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torchvision import transforms

from stfpm.calibration import load_calibration_artifact


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


def run_onnx_inference(
    onnx_path: str,
    image_path: str,
    image_size: int,
    calibration_params_path: str | None = None,
    category: str | None = None,
) -> dict[str, np.ndarray | float | bool]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for ONNX inference. Install it with `pip install onnxruntime`.") from exc

    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_tensor = preprocess_image(image_path, image_size=image_size)
    score_map, image_score = session.run(["score_map", "image_score"], {"input": input_tensor})
    output: dict[str, np.ndarray | float | bool] = {
        "score_map": score_map,
        "image_score": float(image_score[0]),
    }

    if calibration_params_path:
        calibration: dict[str, Any] = load_calibration_artifact(calibration_params_path)
        if category is not None and str(calibration.get("category", category)) != str(category):
            raise ValueError(
                f"Calibration category mismatch: expected '{category}', got '{calibration.get('category')}'."
            )
        if "image_size" in calibration and int(calibration["image_size"]) != int(image_size):
            raise ValueError(
                f"Calibration image_size mismatch: expected {image_size}, got {calibration['image_size']}."
            )

        threshold = float(calibration["threshold"])
        output["threshold"] = threshold
        output["is_anomaly"] = bool(output["image_score"] >= threshold)

    return output
