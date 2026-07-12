"""
Phase 7 — Benchmark inference performance: PyTorch vs ONNX Runtime.

Measures latency, throughput, peak memory, and model size for each backend.
Uses random tensors by default (pure inference speed) or real images from
the test set if configured.

Usage:
    python benchmark.py
    python benchmark.py --user-config configs/user_benchmark.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stfpm.config import load_merged_config
from stfpm.data.common import build_image_transform
from stfpm.data.mvtec import MVTecEvalDataset, collect_mvtec_eval_samples
from stfpm.export.onnx_export import STFPMExportWrapper
from stfpm.models import build_stfpm_model
from stfpm.utils import resolve_device, set_seed
from torch.utils.data import DataLoader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark STFPM inference (PyTorch vs ONNX)")
    parser.add_argument(
        "--default-config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to default config containing all parameters",
    )
    parser.add_argument(
        "--user-config",
        type=str,
        default=None,
        help="Optional user config with fields to override from default config",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def _get_system_info() -> dict[str, str]:
    info: dict[str, str] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }

    try:
        import torch as _torch
        info["torch_version"] = _torch.__version__
        info["torchvision_version"] = _torchvision_version()
    except Exception:
        pass

    try:
        import onnxruntime as _ort
        info["onnxruntime_version"] = _ort.__version__
        info["onnx_available_providers"] = ",".join(_ort.get_available_providers())
    except Exception:
        pass

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = str(torch.cuda.device_count())
        try:
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_mb"] = str(props.total_memory // (1024 * 1024))
        except Exception:
            pass

    return info


def _torchvision_version() -> str:
    import torchvision
    return torchvision.__version__


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def _get_peak_memory_mb(device: torch.device) -> float:
    """Return peak memory in MB. Uses tracemalloc for CPU, torch.cuda for GPU."""
    if device.type == "cuda":
        try:
            return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        except Exception:
            return 0.0
    else:
        import tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        return peak / (1024 * 1024)


def _reset_memory_tracking(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    else:
        import tracemalloc
        tracemalloc.stop()
        tracemalloc.start()


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def _prepare_inputs(
    config: dict[str, Any],
    batch_size: int,
    num_iters: int,
    use_real_images: bool,
    device: torch.device,
) -> list[torch.Tensor]:
    """Prepare a list of input batches for benchmarking."""
    image_size = int(config["dataset"]["image_size"])

    if not use_real_images:
        logger.info("Using random tensors (pure inference speed)")
        return [torch.randn(batch_size, 3, image_size, image_size, device=device) for _ in range(num_iters)]

    # Use real images from test set
    logger.info("Using real images from test set")
    root = Path(config["dataset"]["root"])
    category = config["dataset"]["category"]
    extensions = config["dataset"].get("extensions", ["png", "jpg", "jpeg"])
    samples = collect_mvtec_eval_samples(root, category, extensions)
    if not samples:
        raise RuntimeError(f"No test images found for category '{category}' in {root}")

    transform = build_image_transform(image_size)
    dataset = MVTecEvalDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    batches: list[torch.Tensor] = []
    for batch in loader:
        batches.append(batch["image"].to(device))
        if len(batches) >= num_iters:
            break

    if not batches:
        raise RuntimeError("Not enough images for even one batch. Reduce batch_size.")

    # Repeat last batch if we don't have enough
    while len(batches) < num_iters:
        batches.append(batches[-1])

    return batches[:num_iters]


# ---------------------------------------------------------------------------
# PyTorch benchmark
# ---------------------------------------------------------------------------

def _benchmark_pytorch(
    config: dict[str, Any],
    inputs: list[torch.Tensor],
    device: torch.device,
    warmup_iters: int,
    timed_iters: int,
) -> dict[str, Any]:
    logger.info("Benchmarking PyTorch...")

    checkpoint_path = config["eval"]["checkpoint_path"]
    image_size = int(config["dataset"]["image_size"])

    model = build_stfpm_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.student.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    wrapper = STFPMExportWrapper(model, image_size=image_size).to(device).eval()

    # Warmup
    logger.info("  Warmup: %d iterations", warmup_iters)
    with torch.inference_mode():
        for i in range(warmup_iters):
            _ = wrapper(inputs[i % len(inputs)])
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed run
    logger.info("  Timed: %d iterations", timed_iters)
    _reset_memory_tracking(device)
    latencies: list[float] = []

    with torch.inference_mode():
        for i in range(timed_iters):
            inp = inputs[i % len(inputs)]
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = wrapper(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)  # ms

    peak_memory = _get_peak_memory_mb(device)

    latencies_arr = np.array(latencies)
    batch_size = inputs[0].shape[0]
    total_images = timed_iters * batch_size
    total_time_s = latencies_arr.sum() / 1000.0

    results: dict[str, Any] = {
        "backend": "pytorch",
        "device": str(device),
        "batch_size": batch_size,
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "latency_ms": {
            "mean": float(latencies_arr.mean()),
            "std": float(latencies_arr.std()),
            "min": float(latencies_arr.min()),
            "max": float(latencies_arr.max()),
            "p50": float(np.percentile(latencies_arr, 50)),
            "p95": float(np.percentile(latencies_arr, 95)),
            "p99": float(np.percentile(latencies_arr, 99)),
        },
        "throughput_img_per_sec": float(total_images / total_time_s) if total_time_s > 0 else 0.0,
        "peak_memory_mb": float(peak_memory),
    }

    # Cleanup
    del model, wrapper
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# ONNX Runtime benchmark
# ---------------------------------------------------------------------------

def _get_onnx_providers(use_gpu: bool) -> list[str]:
    if use_gpu:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _benchmark_onnx(
    config: dict[str, Any],
    inputs: list[torch.Tensor],
    device: torch.device,
    warmup_iters: int,
    timed_iters: int,
) -> dict[str, Any]:
    logger.info("Benchmarking ONNX Runtime...")

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed, skipping ONNX benchmark")
        return {"backend": "onnx", "error": "onnxruntime not installed"}

    onnx_path = config["onnx"]["output_path"]
    if not Path(onnx_path).exists():
        logger.warning("ONNX file not found: %s, skipping ONNX benchmark", onnx_path)
        return {"backend": "onnx", "error": f"ONNX file not found: {onnx_path}"}

    use_gpu = device.type == "cuda"
    providers = _get_onnx_providers(use_gpu=use_gpu)
    session = ort.InferenceSession(onnx_path, providers=providers)
    logger.info("  ONNX providers: %s", providers)

    # Convert inputs to numpy (ONNX Runtime always takes numpy)
    inputs_np = [inp.cpu().numpy().astype(np.float32) for inp in inputs]

    # Warmup
    logger.info("  Warmup: %d iterations", warmup_iters)
    for i in range(warmup_iters):
        _ = session.run(["score_map", "image_score"], {"input": inputs_np[i % len(inputs_np)]})

    # Timed run
    logger.info("  Timed: %d iterations", timed_iters)
    _reset_memory_tracking(device)
    latencies: list[float] = []

    for i in range(timed_iters):
        inp = inputs_np[i % len(inputs_np)]
        start = time.perf_counter()
        _ = session.run(["score_map", "image_score"], {"input": inp})
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # ms

    peak_memory = _get_peak_memory_mb(device)

    latencies_arr = np.array(latencies)
    batch_size = inputs_np[0].shape[0]
    total_images = timed_iters * batch_size
    total_time_s = latencies_arr.sum() / 1000.0

    results: dict[str, Any] = {
        "backend": "onnx",
        "device": str(device),
        "providers": providers,
        "batch_size": batch_size,
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "latency_ms": {
            "mean": float(latencies_arr.mean()),
            "std": float(latencies_arr.std()),
            "min": float(latencies_arr.min()),
            "max": float(latencies_arr.max()),
            "p50": float(np.percentile(latencies_arr, 50)),
            "p95": float(np.percentile(latencies_arr, 95)),
            "p99": float(np.percentile(latencies_arr, 99)),
        },
        "throughput_img_per_sec": float(total_images / total_time_s) if total_time_s > 0 else 0.0,
        "peak_memory_mb": float(peak_memory),
    }

    del session
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Model size
# ---------------------------------------------------------------------------

def _get_model_sizes(config: dict[str, Any]) -> dict[str, float]:
    """Get model file sizes in MB."""
    sizes: dict[str, float] = {}

    checkpoint_path = config.get("eval", {}).get("checkpoint_path")
    if checkpoint_path and Path(checkpoint_path).exists():
        sizes["pytorch_checkpoint_mb"] = Path(checkpoint_path).stat().st_size / (1024 * 1024)

    onnx_path = config.get("onnx", {}).get("output_path")
    if onnx_path and Path(onnx_path).exists():
        sizes["onnx_model_mb"] = Path(onnx_path).stat().st_size / (1024 * 1024)

    return sizes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_results(results: dict[str, Any]) -> str:
    lines: list[str] = [
        "",
        "=" * 70,
        "Phase 7 — Inference Benchmark Results",
        "=" * 70,
    ]

    # System info
    sys_info = results["system_info"]
    lines.append("System Info:")
    for key, value in sys_info.items():
        lines.append(f"  {key:.<30} {value}")
    lines.append("-" * 70)

    # Config
    lines.append(f"  Batch size:      {results['batch_size']}")
    lines.append(f"  Warmup iters:    {results['warmup_iters']}")
    lines.append(f"  Timed iters:     {results['timed_iters']}")
    lines.append(f"  Image size:      {results['image_size']}")
    lines.append(f"  Device:          {results['device']}")
    lines.append(f"  Use real images: {results['use_real_images']}")
    lines.append("-" * 70)

    # Per-backend results
    for bench in results["benchmarks"]:
        if "error" in bench:
            lines.append(f"  {bench['backend'].upper()}: SKIPPED ({bench['error']})")
            continue

        lines.append(f"  {bench['backend'].upper()} (device={bench['device']}):")
        lat = bench["latency_ms"]
        lines.append(f"    Latency (ms):")
        lines.append(f"      mean:   {lat['mean']:.3f}")
        lines.append(f"      std:    {lat['std']:.3f}")
        lines.append(f"      min:    {lat['min']:.3f}")
        lines.append(f"      max:    {lat['max']:.3f}")
        lines.append(f"      p50:    {lat['p50']:.3f}")
        lines.append(f"      p95:    {lat['p95']:.3f}")
        lines.append(f"      p99:    {lat['p99']:.3f}")
        lines.append(f"    Throughput:   {bench['throughput_img_per_sec']:.1f} img/sec")
        lines.append(f"    Peak memory:  {bench['peak_memory_mb']:.1f} MB")
        lines.append("-" * 70)

    # Model sizes
    sizes = results["model_sizes"]
    if sizes:
        lines.append("  Model Sizes:")
        for key, value in sizes.items():
            lines.append(f"    {key}: {value:.2f} MB")
        lines.append("-" * 70)

    # Comparison table
    valid_benchmarks = [b for b in results["benchmarks"] if "error" not in b]
    if len(valid_benchmarks) >= 2:
        lines.append("  Comparison:")
        lines.append(f"    {'Backend':<15} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Throughput':<15} {'Memory (MB)':<12}")
        lines.append(f"    {'-'*15} {'-'*12} {'-'*12} {'-'*15} {'-'*12}")
        for bench in valid_benchmarks:
            lat = bench["latency_ms"]
            lines.append(
                f"    {bench['backend']:<15} {lat['mean']:<12.3f} {lat['p95']:<12.3f} "
                f"{bench['throughput_img_per_sec']:<15.1f} {bench['peak_memory_mb']:<12.1f}"
            )
        lines.append("-" * 70)

    lines.append("=" * 70)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    device = resolve_device(config.get("device", "cpu"))
    image_size = int(config["dataset"]["image_size"])

    bench_cfg = config.get("benchmark", {})
    warmup_iters = int(bench_cfg.get("warmup_iters", 10))
    timed_iters = int(bench_cfg.get("timed_iters", 100))
    batch_size = int(bench_cfg.get("batch_size", 1))
    batch_size = max(1, batch_size)
    backends = bench_cfg.get("backends", ["pytorch", "onnx"])
    use_real_images = bool(bench_cfg.get("use_real_images", False))

    logger.info(
        "Benchmark config: batch_size=%d, warmup=%d, timed=%d, backends=%s, real_images=%s",
        batch_size,
        warmup_iters,
        timed_iters,
        backends,
        use_real_images,
    )

    # Prepare inputs
    inputs = _prepare_inputs(
        config=config,
        batch_size=batch_size,
        num_iters=max(warmup_iters, timed_iters),
        use_real_images=use_real_images,
        device=device,
    )

    # Start memory tracking
    import tracemalloc
    tracemalloc.start()

    # Run benchmarks
    benchmarks: list[dict[str, Any]] = []
    if "pytorch" in backends:
        benchmarks.append(
            _benchmark_pytorch(config, inputs, device, warmup_iters, timed_iters)
        )

    if "onnx" in backends:
        benchmarks.append(
            _benchmark_onnx(config, inputs, device, warmup_iters, timed_iters)
        )

    tracemalloc.stop()

    # Model sizes
    model_sizes = _get_model_sizes(config)

    # System info
    system_info = _get_system_info()

    results: dict[str, Any] = {
        "system_info": system_info,
        "image_size": image_size,
        "batch_size": batch_size,
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "device": str(device),
        "use_real_images": use_real_images,
        "benchmarks": benchmarks,
        "model_sizes": model_sizes,
    }

    return results


def main() -> None:
    args = parse_args()
    config = load_merged_config(args.default_config, args.user_config)
    set_seed(int(config.get("seed", 0)))

    results = run_benchmark(config)

    report = _format_results(results)
    print(report)

    output_json = config.get("benchmark", {}).get("output_json")
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        logger.info("Saved benchmark results to %s", output_path)


if __name__ == "__main__":
    main()
