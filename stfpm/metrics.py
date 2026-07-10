from __future__ import annotations

import numpy as np
from skimage import measure
from sklearn.metrics import auc, roc_curve


def evaluate(labels: np.ndarray, scores: np.ndarray, metric: str = "roc") -> float:
    if metric == "pro":
        return pro(labels, scores)
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


def pro(masks: np.ndarray, scores: np.ndarray) -> float:
    max_step = 4000
    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step if max_step > 0 else 0

    pros_mean: list[float] = []
    fprs: list[float] = []
    binary_score_maps = np.zeros_like(scores, dtype=bool)

    for step in range(max_step):
        threshold = max_th - step * delta
        binary_score_maps[scores <= threshold] = 0
        binary_score_maps[scores > threshold] = 1

        pro_values = []
        for index in range(len(binary_score_maps)):
            label_map = measure.label(masks[index], connectivity=2)
            props = measure.regionprops(label_map, binary_score_maps[index])
            for prop in props:
                pro_values.append(prop.intensity_image.sum() / prop.area)

        pros_mean.append(float(np.mean(pro_values)) if pro_values else 0.0)

        masks_neg = ~masks
        neg_sum = masks_neg.sum()
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / neg_sum if neg_sum > 0 else 0.0
        fprs.append(float(fpr))

    pros_mean_arr = np.array(pros_mean)
    fprs_arr = np.array(fprs)

    expect_fpr = 0.3
    idx = fprs_arr <= expect_fpr
    if idx.sum() < 2:
        return 0.0

    fprs_selected = rescale(fprs_arr[idx])
    pros_mean_selected = rescale(pros_mean_arr[idx])
    return float(auc(fprs_selected, pros_mean_selected))
