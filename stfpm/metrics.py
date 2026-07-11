from __future__ import annotations

import numpy as np
from skimage import measure
from sklearn.metrics import auc, roc_curve
from joblib import Parallel, delayed


def evaluate(labels: np.ndarray, scores: np.ndarray, metric: str = "roc", config: dict | None = None) -> float:
    if metric == "pro":
        return pro(labels, scores, config)
    if metric == "roc":
        return roc(labels, scores)
    raise NotImplementedError("Check the evaluation metric.")


def roc(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


def rescale(values: np.ndarray) -> np.ndarray:
    value_min = values.min()
    value_max = values.max()
    denom = value_max - value_min
    if denom == 0:
        return np.zeros_like(values, dtype=np.float64)
    return (values - value_min) / denom


def pro(masks: np.ndarray, scores: np.ndarray, config: dict | None = None) -> float:
    if config is not None:
        n_jobs = config["n_jobs"]
        max_step = config["pro_max_step"]
    else:
        n_jobs = -1
        max_step = 4000

    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step if max_step > 0 else 0.0

    # binary_score_maps = np.zeros_like(scores, dtype=bool)

    def compute_pro_fpr(
        thresh: float,
        scores: np.ndarray = scores,
        masks: np.ndarray = masks,
    ) -> tuple[float, float]:

        binary_score_maps = np.zeros_like(scores, dtype=bool)
        binary_score_maps[scores <= thresh] = 0
        binary_score_maps[scores > thresh] = 1

        pro_values = []
        for index in range(len(binary_score_maps)):
            label_map = measure.label(masks[index], connectivity=2)
            props = measure.regionprops(label_map, binary_score_maps[index])
            for prop in props:
                pro_values.append(prop.image_intensity.sum() / prop.area)

        pros_mean_val = float(np.mean(pro_values)) if pro_values else 0.0

        masks_neg = ~masks
        neg_sum = masks_neg.sum()
        fpr_val = np.logical_and(masks_neg, binary_score_maps).sum() / neg_sum if neg_sum > 0 else 0.0
        return pros_mean_val, fpr_val

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_pro_fpr)(
            max_th - step * delta) 
        for step in range(max_step)
    )

    pros_mean: list[float] = []
    fprs: list[float] = []
    for result in results:
        pros_mean_val, fpr_val = result
        pros_mean.append(pros_mean_val)
        fprs.append(fpr_val)

    pros_mean_arr = np.array(pros_mean)
    fprs_arr = np.array(fprs)

    expect_fpr = 0.3
    idx = fprs_arr <= expect_fpr
    if idx.sum() < 2:
        return 0.0

    fprs_selected = rescale(fprs_arr[idx])
    pros_mean_selected = rescale(pros_mean_arr[idx])
    return float(auc(fprs_selected, pros_mean_selected))
