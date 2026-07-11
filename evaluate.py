from __future__ import annotations

import argparse
import logging

from stfpm.config import load_merged_config
from stfpm.data import build_dataloaders
from stfpm.evaluation import evaluate_checkpoint
from stfpm.models import build_stfpm_model
from stfpm.utils import resolve_device, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate STFPM model")
    parser.add_argument(
        "--default-config",
        type=str,
        default="configs/eval_mvtec.yaml",
        help="Path to default eval config containing all parameters",
    )
    parser.add_argument(
        "--user-config",
        type=str,
        default=None,
        help="Optional user config with fields to override from default config",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_merged_config(args.default_config, args.user_config)
    set_seed(int(config["seed"]))
    device = resolve_device(config["device"])

    loaders = build_dataloaders(config, split="test")
    model = build_stfpm_model(config)

    checkpoint_path = args.checkpoint or config["eval"]["checkpoint_path"]
    if not checkpoint_path:
        raise ValueError("Checkpoint is required. Provide --checkpoint or eval.checkpoint_path in config.")

    metrics = evaluate_checkpoint(model, loaders["test"], config, checkpoint_path, device)
    category = config["dataset"]["category"]
    logger.info(
        "Category: {category}\tPixel-AUC: {pixel_auc:.6f}\tImage-AUC: {image_auc:.6f}\tPRO: {pro:.6f}".format(
            category=category,
            **metrics,
        )
    )


if __name__ == "__main__":
    main()

