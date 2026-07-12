"""Generate and save evaluation visualizations: confusion matrix and ROC curves."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    category: str = "",
    threshold: float | None = None,
) -> str:
    """Save confusion matrix plot as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            color = "white" if value > cm.max() * 0.5 else "black"
            ax.text(j, i, f"{labels[i][j]}\n{value}", ha="center", va="center", color=color, fontsize=14)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    title = f"Confusion Matrix — {category}" if category else "Confusion Matrix"
    if threshold is not None:
        title += f" (threshold={threshold:.4f})"
    ax.set_title(title)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_roc_curves(
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    pixel_labels: np.ndarray,
    pixel_scores: np.ndarray,
    save_path: str,
    category: str = "",
    image_auc: float | None = None,
    pixel_auc: float | None = None,
) -> str:
    """Save image-level and pixel-level ROC curves as a single PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Image-level ROC ---
    fpr_img, tpr_img, _ = roc_curve(image_labels, image_scores)
    auc_img = float(np.trapezoid(tpr_img, fpr_img)) if image_auc is None else image_auc
    axes[0].plot(fpr_img, tpr_img, color="darkorange", lw=2, label=f"ROC (AUC = {auc_img:.4f})")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    title_img = f"Image-level ROC — {category}" if category else "Image-level ROC"
    axes[0].set_title(title_img)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # --- Pixel-level ROC ---
    fpr_pix, tpr_pix, _ = roc_curve(pixel_labels, pixel_scores)
    auc_pix = float(np.trapezoid(tpr_pix, fpr_pix)) if pixel_auc is None else pixel_auc
    axes[1].plot(fpr_pix, tpr_pix, color="darkgreen", lw=2, label=f"ROC (AUC = {auc_pix:.4f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    title_pix = f"Pixel-level ROC — {category}" if category else "Pixel-level ROC"
    axes[1].set_title(title_pix)
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_evaluation_plots(
    *,
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    pixel_labels: np.ndarray,
    pixel_scores: np.ndarray,
    category: str,
    output_dir: str,
    image_auc: float,
    pixel_auc: float,
    calibrated_threshold: float | None = None,
    calibrated_predictions: np.ndarray | None = None,
) -> dict[str, str]:
    """Generate and save all evaluation plots.

    Returns dict of plot name -> saved path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}

    # ROC curves (always generated)
    roc_path = plot_roc_curves(
        image_labels=image_labels,
        image_scores=image_scores,
        pixel_labels=pixel_labels,
        pixel_scores=pixel_scores,
        save_path=str(output_path / "roc_curves.png"),
        category=category,
        image_auc=image_auc,
        pixel_auc=pixel_auc,
    )
    saved["roc_curves"] = roc_path

    # Confusion matrix (only when calibration threshold is available)
    if calibrated_threshold is not None and calibrated_predictions is not None:
        cm_path = plot_confusion_matrix(
            y_true=image_labels,
            y_pred=calibrated_predictions,
            save_path=str(output_path / "confusion_matrix.png"),
            category=category,
            threshold=calibrated_threshold,
        )
        saved["confusion_matrix"] = cm_path

    return saved
