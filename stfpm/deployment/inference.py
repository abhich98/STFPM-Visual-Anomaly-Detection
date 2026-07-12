from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

from stfpm.deployment.onnx_runtime import load_onnx_session, run_onnx_batch
from stfpm.evaluation.calibration import load_calibration_artifact


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
    use_gpu: bool = False,
) -> dict[str, np.ndarray | float | bool]:
    session = load_onnx_session(onnx_path, use_gpu=use_gpu)
    input_tensor = preprocess_image(image_path, image_size=image_size)
    score_map, image_score = run_onnx_batch(session, input_tensor)
    output: dict[str, np.ndarray | float | bool] = {
        "score_map": score_map,
        "image_score": float(image_score[0]),
    }

    if calibration_params_path:
        calibration: dict[str, Any] = load_calibration_artifact(calibration_params_path)
        if category is not None and str(calibration["category"]) != str(category):
            raise ValueError(
                f"Calibration category mismatch: expected '{category}', got '{calibration['category']}'."
            )
        if int(calibration["image_size"]) != int(image_size):
            raise ValueError(
                f"Calibration image_size mismatch: expected {image_size}, got {calibration['image_size']}."
            )

        threshold = float(calibration["threshold"])
        output["threshold"] = threshold
        output["is_anomaly"] = bool(output["image_score"] >= threshold)

    return output


def save_score_map_overlay(image_path: str, score_map: np.ndarray, output_dir: str, is_anomaly: bool | None = None) -> None:
    
    im_name = Path(image_path).stem + "_overlay"
    im_name += "_anomaly" if is_anomaly else "_normal" if is_anomaly is not None else ""
    output_path = Path(output_dir) / (im_name + ".png")

    image = Image.open(image_path).convert("RGB")
    score_map_resized = Image.fromarray(score_map).resize(image.size, resample=Image.BILINEAR)

    score_map_colored = np.array(score_map_resized.convert("L"))
    heatmap = np.zeros((score_map_colored.shape[0], score_map_colored.shape[1], 3), dtype=np.uint8)
    heatmap[..., 0] = score_map_colored
    heatmap[..., 1] = 0
    heatmap[..., 2] = 255 - score_map_colored
    overlay = Image.blend(image, Image.fromarray(heatmap), alpha=0.5)

    overlay.save(output_path)