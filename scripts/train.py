from __future__ import annotations

import argparse
import logging

from stfpm.config import get_default_config_path, load_merged_config
from stfpm.data import build_dataloaders
from stfpm.models import build_stfpm_model
from stfpm.training import train
from stfpm.utils import resolve_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STFPM model")
    parser.add_argument(
        "--default-config",
        type=str,
        default=get_default_config_path(),
        help="Path to default train config containing all parameters",
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
    device = resolve_device(config["device"])

    loaders = build_dataloaders(config, split="train")
    model = build_stfpm_model(config)
    ckpt = train(model, loaders["train"], loaders["val"], config, device)
    logger.info(f"Saved best checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
