from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_threshold_at_target_fpr(normal_scores: np.ndarray, target_fpr: float) -> float:
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in [0, 1), got {target_fpr}")
    if normal_scores.size == 0:
        raise ValueError("Cannot calibrate threshold: no normal-image scores were found.")

    quantile = 1.0 - target_fpr
    threshold = np.quantile(normal_scores, quantile, method="higher")

    return float(threshold)


def build_calibration_artifact(
    *,
    category: str,
    image_size: int,
    checkpoint_path: str,
    target_fpr: float,
    threshold: float,
    normal_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "target_fpr",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "image_size": image_size,
        "checkpoint_path": checkpoint_path,
        "target_fpr": float(target_fpr),
        "threshold": float(threshold),
        "normal_count": int(normal_count),
    }


def save_calibration_artifact(path: str, artifact: dict[str, Any]) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
    return str(output_path)


def load_calibration_artifact(path: str) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Calibration artifact not found: {path}")
    with artifact_path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, dict):
        raise ValueError("Calibration artifact must be a JSON object.")
    if "threshold" not in artifact:
        raise ValueError("Calibration artifact missing required field: 'threshold'.")
    return artifact
