from __future__ import annotations

import argparse
import logging

from stfpm.config import load_merged_config
from stfpm.export import export_onnx
from stfpm.models import build_stfpm_model
from stfpm.utils import set_seed


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export STFPM to ONNX")
    parser.add_argument(
        "--default-config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to default export config containing all parameters",
    )
    parser.add_argument(
        "--user-config",
        type=str,
        default=None,
        help="Optional user config with fields to override from default config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_merged_config(args.default_config, args.user_config)
    set_seed(int(config["seed"]))
    model = build_stfpm_model(config)

    checkpoint_path = config["eval"]["checkpoint_path"]
    if not checkpoint_path:
        raise ValueError("Checkpoint is required. Provide --checkpoint or eval.checkpoint_path in config.")

    output_path = config["onnx"]["output_path"]
    exported = export_onnx(model, config, checkpoint_path, output_path)
    logger.info(f"Exported ONNX: {exported}")


if __name__ == "__main__":
    main()
