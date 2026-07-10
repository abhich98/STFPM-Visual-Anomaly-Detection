from __future__ import annotations

import random
import logging

import numpy as np
import torch


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None) -> torch.device:
    req = (requested or "cpu").lower()
    if req.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")

    logger.info(f"Using device: {req}")
    return torch.device(req)
