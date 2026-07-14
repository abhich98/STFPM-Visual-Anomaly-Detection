from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torchvision import transforms

from stfpm.deployment.onnx_runtime import load_onnx_session, run_onnx_batch
from stfpm.evaluation.calibration import load_calibration_artifact


def _normalize_score_map(score_map: np.ndarray, threshold: float | None = None) -> np.ndarray:

    min_value = float(score_map.min())
    max_value = float(score_map.max()) if threshold is None else threshold

    if max_value - min_value < 1e-12:
        return np.zeros_like(score_map, dtype=np.uint8)

    adjusted = np.clip(score_map, min_value, max_value)
    normalized = (adjusted - min_value) / (max_value - min_value)
    return (normalized * 255.0).astype(np.uint8)


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


def save_score_map_overlay(image_path: str, inference_res: dict, output_dir: str) -> None:
    import cv2

    input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    score_map = inference_res["score_map"][0, 0]
    score_map_resized = cv2.resize(score_map, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_LINEAR)

    threshold = inference_res.get("threshold", None)
    score_map_u8 = _normalize_score_map(score_map_resized, threshold=threshold)
    heatmap = cv2.applyColorMap(score_map_u8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(input_image, 0.6, heatmap, 0.4, 0.0)

    is_anomaly = inference_res.get("is_anomaly", None)
    im_name = Path(image_path).stem + "_overlay"
    im_name += "_anomaly" if is_anomaly else "_normal" if is_anomaly is not None else ""
    output_path = Path(output_dir) / (im_name + ".png")

    # print("I am here!!!")
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))