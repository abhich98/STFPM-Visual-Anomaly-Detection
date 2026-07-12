from __future__ import annotations

from typing import Any
import logging
import numpy as np
import torch

from tqdm import tqdm

from stfpm.evaluation.calibration import (
    build_calibration_artifact,
    compute_threshold_at_target_fpr,
    save_calibration_artifact,
)
from stfpm.evaluation.visualizations import save_evaluation_plots
from stfpm.data.mvtec import load_mask
from stfpm.metrics import evaluate
from stfpm.models.stfpm import STFPM


logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    model: STFPM,
    loader,
    config: dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
    calibration_cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
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
    image_scores = score_arr.max(-1).max(-1)

    pixel_auc = float(evaluate(pixel_arr.flatten(), score_arr.flatten(), metric="roc"))
    image_auc = float(evaluate(image_label_arr, image_scores, metric="roc"))
    pro_score = float(evaluate(pixel_arr.astype(bool), score_arr, metric="pro", config=config["eval"]))

    metrics: dict[str, float] = {
        "pixel_auc": pixel_auc,
        "image_auc": image_auc,
        "pro": pro_score,
    }

    if calibration_cfg and bool(calibration_cfg["enabled"]):
        target_fpr = float(calibration_cfg["target_fpr"])
        normal_scores = image_scores[image_label_arr == 0]
        threshold = compute_threshold_at_target_fpr(normal_scores, target_fpr)

        pred_is_anomaly = image_scores >= threshold
        true_is_anomaly = image_label_arr.astype(bool)

        tp = int(np.logical_and(pred_is_anomaly, true_is_anomaly).sum())
        fp = int(np.logical_and(pred_is_anomaly, ~true_is_anomaly).sum())
        fn = int(np.logical_and(~pred_is_anomaly, true_is_anomaly).sum())

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics["calibrated_threshold"] = float(threshold)
        metrics["calibrated_target_fpr"] = float(target_fpr)
        metrics["calibrated_precision"] = precision
        metrics["calibrated_recall"] = recall
        metrics["calibrated_f1"] = f1

        output_path = calibration_cfg["output_path"]
        if output_path:
            artifact = build_calibration_artifact(
                category=str(config["dataset"]["category"]),
                image_size=int(config["dataset"]["image_size"]),
                checkpoint_path=checkpoint_path,
                target_fpr=target_fpr,
                threshold=threshold,
                normal_count=int(normal_scores.size),
            )
            saved_path = save_calibration_artifact(str(output_path), artifact)
            logger.info("Saved calibration artifact: %s", saved_path)

    # --- Generate and save plots ---
    plots_dir = config["eval"]["plots_dir"]
    if plots_dir:
        calibrated_threshold_val = metrics["calibrated_threshold"]
        calibrated_preds = None
        if calibrated_threshold_val is not None:
            calibrated_preds = (image_scores >= calibrated_threshold_val).astype(np.int64)

        saved_plots = save_evaluation_plots(
            image_labels=image_label_arr,
            image_scores=image_scores,
            pixel_labels=pixel_arr.flatten(),
            pixel_scores=score_arr.flatten(),
            category=str(config["dataset"]["category"]),
            output_dir=str(plots_dir),
            image_auc=image_auc,
            pixel_auc=pixel_auc,
            calibrated_threshold=calibrated_threshold_val,
            calibrated_predictions=calibrated_preds,
        )
        for name, path in saved_plots.items():
            logger.info("Saved plot: %s -> %s", name, path)

    return metrics
