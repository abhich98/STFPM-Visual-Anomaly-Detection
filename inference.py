from __future__ import annotations

import argparse
import logging

from stfpm.config import load_merged_config
from stfpm.deployment.inference import run_onnx_inference


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX inference for STFPM")
    parser.add_argument(
        "--default-config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to default inference config containing all parameters",
    )
    parser.add_argument(
        "--user-config",
        type=str,
        default=None,
        help="Optional user config with fields to override from default config",
    )
    parser.add_argument("--onnx", type=str, default=None, help="Path to ONNX file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--calibration-params",
        type=str,
        default=None,
        help="Optional calibration JSON produced by evaluate.py for is_anomaly decision",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_merged_config(args.default_config, args.user_config)
    onnx_path = args.onnx or config["onnx"]["output_path"]
    image_size = int(config["dataset"]["image_size"])
    result = run_onnx_inference(
        onnx_path,
        args.image,
        image_size,
        calibration_params_path=args.calibration_params,
        category=str(config["dataset"]["category"]),
    )
    logger.info(f"Image score: {result['image_score']:.6f}")
    if "is_anomaly" in result:
        logger.info("Decision: is_anomaly=%s (threshold=%.6f)", result["is_anomaly"], result["threshold"])


if __name__ == "__main__":
    main()
