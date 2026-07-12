"""
Phase 6 — Validate ONNX numerical equivalence against PyTorch.

Runs the same set of images through both the PyTorch model and the exported
ONNX graph, then compares ``score_map`` and ``image_score`` outputs.

Acceptance criterion (per PLAN.md):
    max absolute error < 1e-5

Usage:
    python validate_onnx.py \
        --default-config configs/default_config.yaml \
        --user-config configs/user_validate.yaml

    # With only default config
    python validate_onnx.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from stfpm.config import get_default_config_path, load_merged_config
from stfpm.data.common import build_image_transform
from stfpm.data.mvtec import MVTecEvalDataset, collect_mvtec_eval_samples
from stfpm.export.onnx_export import STFPMExportWrapper
from stfpm.models import build_stfpm_model
from stfpm.utils import resolve_device, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ONNX outputs against PyTorch")
    parser.add_argument(
        "--default-config",
        type=str,
        default=get_default_config_path(),
        help="Path to default config containing all parameters",
    )
    parser.add_argument(
        "--user-config",
        type=str,
        default=None,
        help="Optional user config with fields to override from default config",
    )
    return parser.parse_args()


def _get_onnx_providers(use_gpu: bool) -> list[str]:
    """Return ONNX Runtime providers, GPU-first with CPU fallback."""
    if use_gpu:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _load_onnx_session(onnx_path: str, providers: list[str]):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for validation. Install with `pip install onnxruntime`."
        ) from exc

    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    session = ort.InferenceSession(onnx_path, providers=providers)
    available = ort.get_available_providers()
    logger.info("ONNX providers requested: %s (available: %s)", providers, available)
    return session


def _run_pytorch(
    wrapper: STFPMExportWrapper, images: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Run PyTorch model and return (score_map, image_score) as numpy arrays."""
    with torch.inference_mode():
        score_map, image_score = wrapper(images)
    return (
        score_map.squeeze(1).cpu().numpy(),
        image_score.cpu().numpy(),
    )


def _run_onnx(session, images_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run ONNX Runtime and return (score_map, image_score) as numpy arrays."""
    score_map, image_score = session.run(
        ["score_map", "image_score"],
        {"input": images_np},
    )
    return score_map.squeeze(1), image_score


def validate(config: dict[str, Any]) -> dict[str, Any]:
    # --- Resolve device ---
    device = resolve_device(config.get("device", "cpu"))
    use_gpu = device.type == "cuda"

    image_size = int(config["dataset"]["image_size"])
    category = config["dataset"]["category"]

    val_cfg = config.get("validation", {})
    batch_size = int(val_cfg.get("batch_size", 32))
    batch_size = max(1, batch_size)
    max_images = val_cfg.get("max_images")
    if max_images is not None:
        max_images = int(max_images)
    tolerance = float(val_cfg.get("tolerance", 1e-5))
    output_json = val_cfg.get("output_json")

    # --- Build PyTorch model ---
    checkpoint_path = config["eval"]["checkpoint_path"]
    model = build_stfpm_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.student.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    wrapper = STFPMExportWrapper(model, image_size=image_size).to(device).eval()

    # --- Load ONNX model ---
    onnx_path = config["onnx"]["output_path"]
    providers = _get_onnx_providers(use_gpu=use_gpu)
    session = _load_onnx_session(onnx_path, providers=providers)

    # --- Collect test images ---
    root = Path(config["dataset"]["root"])
    extensions = config["dataset"].get("extensions", ["png", "jpg", "jpeg"])
    samples = collect_mvtec_eval_samples(root, category, extensions)
    if not samples:
        raise RuntimeError(f"No test images found for category '{category}' in {root}")

    if max_images is not None and max_images > 0:
        samples = samples[:max_images]

    logger.info(
        "Validating %d images (category=%s, image_size=%d, batch_size=%d, device=%s, onnx_providers=%s)",
        len(samples),
        category,
        image_size,
        batch_size,
        device,
        providers,
    )

    # --- Build DataLoader from MVTecEvalDataset ---
    transform = build_image_transform(image_size)
    dataset = MVTecEvalDataset(samples, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=int(config["dataset"].get("num_workers", 0)),
        pin_memory=bool(config["dataset"].get("pin_memory", False)),
    )

    # --- Run validation in batches ---
    pt_score_maps: list[np.ndarray] = []
    onnx_score_maps: list[np.ndarray] = []
    pt_image_scores: list[float] = []
    onnx_image_scores: list[float] = []

    num_images = len(dataset)
    processed = 0
    for batch in loader:
        images = batch["image"].to(device)

        # PyTorch
        pt_map, pt_score = _run_pytorch(wrapper, images)
        # ONNX (expects numpy on CPU)
        onnx_input = images.cpu().numpy().astype(np.float32)
        onnx_map, onnx_score = _run_onnx(session, onnx_input)

        for j in range(pt_map.shape[0]):
            pt_score_maps.append(pt_map[j])
            onnx_score_maps.append(onnx_map[j])
            pt_image_scores.append(float(pt_score[j]))
            onnx_image_scores.append(float(onnx_score[j]))

        processed += pt_map.shape[0]
        if processed % (batch_size * 25) < batch_size or processed >= num_images:
            logger.info("  Processed %d/%d images", processed, num_images)

    # --- Compute errors ---
    pt_maps_arr = np.stack(pt_score_maps, axis=0)
    onnx_maps_arr = np.stack(onnx_score_maps, axis=0)
    pt_scores_arr = np.array(pt_image_scores)
    onnx_scores_arr = np.array(onnx_image_scores)

    map_abs_err = np.abs(pt_maps_arr - onnx_maps_arr)
    score_abs_err = np.abs(pt_scores_arr - onnx_scores_arr)

    results: dict[str, Any] = {
        "category": category,
        "image_size": image_size,
        "num_images": len(samples),
        "batch_size": batch_size,
        "device": str(device),
        "onnx_providers": providers,
        "checkpoint_path": checkpoint_path,
        "onnx_path": onnx_path,
        "tolerance": tolerance,
        "score_map": {
            "max_abs_error": float(map_abs_err.max()),
            "mean_abs_error": float(map_abs_err.mean()),
            "median_abs_error": float(np.median(map_abs_err)),
            "p99_abs_error": float(np.percentile(map_abs_err, 99)),
        },
        "image_score": {
            "max_abs_error": float(score_abs_err.max()),
            "mean_abs_error": float(score_abs_err.mean()),
            "median_abs_error": float(np.median(score_abs_err)),
            "p99_abs_error": float(np.percentile(score_abs_err, 99)),
        },
    }

    map_pass = results["score_map"]["max_abs_error"] < tolerance
    score_pass = results["image_score"]["max_abs_error"] < tolerance
    results["passed"] = bool(map_pass and score_pass)

    return results


def _format_results(results: dict[str, Any]) -> str:
    lines = [
        "",
        "=" * 60,
        "Phase 6 — ONNX Validation Results",
        "=" * 60,
        f"  Category:        {results['category']}",
        f"  Images:          {results['num_images']}",
        f"  Batch size:      {results['batch_size']}",
        f"  Image size:      {results['image_size']}",
        f"  Device:          {results['device']}",
        f"  ONNX providers:  {results['onnx_providers']}",
        f"  Tolerance:       {results['tolerance']:.1e}",
        "-" * 60,
        "  Score Map:",
        f"    max abs error:    {results['score_map']['max_abs_error']:.2e}",
        f"    mean abs error:   {results['score_map']['mean_abs_error']:.2e}",
        f"    median abs error: {results['score_map']['median_abs_error']:.2e}",
        f"    p99 abs error:    {results['score_map']['p99_abs_error']:.2e}",
        "-" * 60,
        "  Image Score:",
        f"    max abs error:    {results['image_score']['max_abs_error']:.2e}",
        f"    mean abs error:   {results['image_score']['mean_abs_error']:.2e}",
        f"    median abs error: {results['image_score']['median_abs_error']:.2e}",
        f"    p99 abs error:    {results['image_score']['p99_abs_error']:.2e}",
        "-" * 60,
    ]

    verdict = "PASSED ✓" if results["passed"] else "FAILED ✗"
    lines.append(f"  Verdict: {verdict}")
    lines.append("=" * 60)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_merged_config(args.default_config, args.user_config)
    set_seed(int(config.get("seed", 0)))

    results = validate(config)

    report = _format_results(results)
    print(report)

    output_json = config.get("validation", {}).get("output_json")
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        logger.info("Saved validation results to %s", output_path)

    if not results["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
