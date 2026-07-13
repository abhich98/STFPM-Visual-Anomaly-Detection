# STFPM — Student-Teacher Feature Pyramid Matching for Anomaly Detection

A modernized PyTorch implementation of [STFPM](https://arxiv.org/abs/2103.04257v3) (BMVC 2021) for unsupervised visual anomaly detection, extended with an end-to-end MLOps pipeline: training → evaluation → ONNX export → numerical validation → benchmarking → deployment-ready inference.

![architecture](./figs/arch.jpg)

## Overview

**STFPM** trains a *student* network to mimic the feature maps of a pretrained *teacher* network on normal images. At inference time, discrepancies between teacher and student features reveal anomalous regions — no anomaly labels are required during training.

This repo implements the method on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/) dataset and wraps the full lifecycle in a config-driven, package-based codebase.

**Tech stack:** Python ≥ 3.12 · PyTorch · torchvision · ONNX · ONNX Runtime · scikit-learn · scikit-image · OpenCV · uv

## Project Highlights

- 🔧 **Modernized implementation** — refactored the original script into a modular `stfpm/` package with a config-driven CLI (`uv`-managed, type-hinted, Python 3.12+).
- 📈 **Reproduced results** on MVTec AD (pixel-AUC, image-AUC, PRO).
- 📤 **ONNX export** of the full inference pipeline (score map + image score) with dynamic batch support via `torch.onnx.export(dynamo=True)`.
- ✅ **Numerical validation** suite comparing PyTorch vs ONNX outputs across the test set.
- 📊 **Benchmarking suite** measuring latency, throughput, and memory for both backends.
- 🎯 **Calibration** at a target false-positive rate, producing a versioned JSON artifact consumed at inference time.
- 🖼️ **Visualization** — ROC curves, confusion matrices, and anomaly heatmap overlays.

## Results

#### [Original](https://github.com/gdwang08/STFPM) paper (200 epochs, MVTec AD)
| Category   | Pixel-AUC | Image-AUC |   PRO   |
| :--------: | :-------: | :-------: | :-----: |
| carpet     | 0.990292  | 0.964286  | 0.966061|
| transistor | 0.819404  | 0.939167  | 0.880923|

#### Reproduced metrics (this implementation, 120 epochs, MVTec AD)
| Category   | Pixel-AUC | Image-AUC |   PRO   |
| :--------: | :-------: | :-------: | :-----: |
| carpet     | 0.988579  | 0.965490  | 0.963964 |
| transistor | 0.806192  | 0.924583  | 0.878270 |
### Example outputs

<!-- Anomaly heatmap overlay on a defective `transistor` sample:

<p align="center">
  <img src="./results/001_overlay.png" width="45%" alt="anomaly overlay" />
  <img src="./results/001_overlay_normal.png" width="45%" alt="normal overlay" />
</p>

Evaluation plots (ROC curves + confusion matrix) are generated under `results/<category>/`. -->

## Deployment & Benchmarking

```text
PyTorch model  ──►  ONNX export (dynamo, dynamic batch)  ──►  ONNX Runtime  ──►  Benchmark
```

### Inference benchmark (NVIDIA RTX 2000 Ada, CUDA, batch=1, image 256×256)

| Backend       | Mean Latency | P95 Latency | Throughput    | Peak Memory |
| ------------- | -----------: | ----------: | ------------: | ----------: |
| PyTorch       |     6.47 ms  |    7.52 ms  |   154 img/s   |    176 MB   |
| ONNX Runtime  |     2.20 ms  |    2.36 ms  |   456 img/s   |     75 MB   |

**ONNX Runtime is ~3× faster and uses ~57% less memory** than PyTorch for this model.

### Numerical validation (PyTorch vs ONNX, 100 test images)

| Output      | Mean Abs Error | Max Abs Error |
| ----------- | -------------: | ------------: |
| Score map   |     5.9e-06    |    2.8e-04    |
| Image score |     5.6e-05    |    2.2e-04    |

Differences are well within the noise expected from cross-backend float32 CUDA kernels and do not affect anomaly decisions at the calibrated threshold.

## Repository Structure

```text
configs/            # YAML configs (default + per-task overrides)
stfpm/
  data/             # MVTec AD datasets, transforms, dataloader factory
  models/           # STFPM model, ResNet backbones, model registry
  training/         # Training loop (SGD, best-checkpoint saving)
  evaluation/       # Metrics (pixel/image-AUC, PRO), calibration, plots
  export/           # ONNX export wrapper + export logic
  deployment/       # ONNX Runtime inference + shared session helpers
scripts/            # CLI entry points (train, evaluate, export, validate, benchmark, infer)
artifacts/          # Exported ONNX model + calibration artifact
results/            # Evaluation plots, validation JSON, sample overlays
snapshots/          # Trained student checkpoints
```

## Getting Started

### Install

```bash
git clone <repo-url> && cd STFPM-Visual-Anomaly-Detection
uv sync                       # core deps
uv sync --extra onnx-gpu      # + ONNX Runtime (CUDA); use --extra onnx for CPU
```

Download [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and place categories under `mvtec_anomaly_detection/<category>/`.

### Train

```bash
python scripts/train.py --default-config configs/default_config.yaml
# override per-run without editing files:
python scripts/train.py --user-config configs/user_train.yaml
```

### Evaluate

```bash
python scripts/evaluate.py --default-config configs/default_config.yaml
# computes pixel-AUC, image-AUC, PRO; saves ROC + confusion matrix plots
```

### Export to ONNX

```bash
python scripts/export_onnx.py --default-config configs/default_config.yaml
```

### Validate & Benchmark

```bash
python scripts/validate_onnx.py   # PyTorch vs ONNX numerical equivalence
python scripts/benchmark.py       # latency / throughput / memory comparison
```

### Run Inference

```bash
python scripts/inference.py --image mvtec_anomaly_detection/transistor/test/cut_lead/009.png \
    --calibration-params artifacts/calibration.json
```

All scripts except `scripts/inference.py` accept `--user-config` to override any field from `configs/default_config.yaml`.

## References & Acknowledgements

- **Paper:** Wang et al., *Student-Teacher Feature Pyramid Matching for Anomaly Detection*, BMVC 2021. [arXiv:2103.04257](https://arxiv.org/abs/2103.04257v3)
- **Original implementation:** [STFPM](https://github.com/gasharper/STFPM)
- **Dataset:** [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)


If you find this work useful, please cite the original paper:

```bibtex
@inproceedings{wang2021student_teacher,
    title={Student-Teacher Feature Pyramid Matching for Anomaly Detection},
    author={Wang, Guodong and Han, Shumin and Ding, Errui and Huang, Di},
    booktitle={The British Machine Vision Conference (BMVC)},
    year={2021}
}
```
