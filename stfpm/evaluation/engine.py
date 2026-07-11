from __future__ import annotations

from typing import Any
import logging
import numpy as np
import torch

from tqdm import tqdm

from stfpm.data.mvtec import load_mask
from stfpm.metrics import evaluate
from stfpm.models.stfpm import STFPM


logger = logging.getLogger(__name__)


def evaluate_checkpoint(model: STFPM, loader, config: dict[str, Any], checkpoint_path: str, device: torch.device) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.student.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.teacher.eval()
    model.student.eval()

    image_size = int(config["dataset"]["image_size"])
    score_maps: list[np.ndarray] = []
    image_labels: list[int] = []
    pixel_labels: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().tolist()
            mask_paths = batch["mask_path"]

            with torch.no_grad():
                t_feats = model.teacher(images)
                s_feats = model.student(images)
            score = model.anomaly_score_map(t_feats, s_feats, out_size=(image_size, image_size))
            score_np = score.squeeze(1).cpu().numpy()

            for i in range(score_np.shape[0]):
                score_maps.append(score_np[i])
                image_labels.append(int(labels[i]))
                pixel_labels.append(load_mask(mask_paths[i], image_size=image_size))

    logger.info("Calculating metrics...")
    score_arr = np.stack(score_maps, axis=0)
    pixel_arr = np.stack(pixel_labels, axis=0)
    image_label_arr = np.array(image_labels, dtype=np.int64)

    pixel_auc = float(evaluate(pixel_arr.flatten(), score_arr.flatten(), metric="roc"))
    image_auc = float(evaluate(image_label_arr, score_arr.max(-1).max(-1), metric="roc"))
    pro_score = float(evaluate(pixel_arr.astype(bool), score_arr, metric="pro", config=config["eval"]))
    return {"pixel_auc": pixel_auc, "image_auc": image_auc, "pro": pro_score}
