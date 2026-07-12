"""Shared ONNX Runtime helpers used by inference, validation, and benchmarking.

Centralizes provider selection, session creation, and batch inference so that
``stfpm/deployment/inference.py``, ``scripts/validate_onnx.py``, and
``scripts/benchmark.py`` all behave consistently (including GPU support).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Canonical input/output names for the exported STFPM ONNX graph.
INPUT_NAME = "input"
OUTPUT_NAMES = ["score_map", "image_score"]


def get_onnx_providers(use_gpu: bool) -> list[str]:
    """Return ONNX Runtime providers, GPU-first with CPU fallback.

    Args:
        use_gpu: If True, prefer CUDAExecutionProvider with CPU fallback.
            If False, use CPUExecutionProvider only.
    """
    if use_gpu:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_onnx_session(
    onnx_path: str | Path,
    providers: list[str] | None = None,
    use_gpu: bool = False,
):
    """Create an ONNX Runtime inference session.

    Args:
        onnx_path: Path to the exported ONNX model file.
        providers: Optional explicit provider list. If None, providers are
            selected based on ``use_gpu``.
        use_gpu: Ignored if ``providers`` is provided. Otherwise selects
            GPU-first providers when True, CPU-only when False.

    Raises:
        RuntimeError: If onnxruntime is not installed.
        FileNotFoundError: If the ONNX file does not exist.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for ONNX inference. "
            "Install it with `pip install onnxruntime` (or onnxruntime-gpu)."
        ) from exc

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    if providers is None:
        providers = get_onnx_providers(use_gpu=use_gpu)

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    available = ort.get_available_providers()
    logger.info("ONNX providers requested: %s (available: %s)", providers, available)
    return session


def run_onnx_batch(
    session,
    images_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a batch through an ONNX Runtime session.

    Args:
        session: An ``ort.InferenceSession`` returned by ``load_onnx_session``.
        images_np: Input batch as a float32 numpy array of shape
            ``(N, 3, H, W)``.

    Returns:
        A tuple ``(score_map, image_score)`` where ``score_map`` has shape
        ``(N, 1, H, W)`` and ``image_score`` has shape ``(N,)``.
    """
    images_np = np.ascontiguousarray(images_np, dtype=np.float32)
    score_map, image_score = session.run(OUTPUT_NAMES, {INPUT_NAME: images_np})
    return score_map, image_score
