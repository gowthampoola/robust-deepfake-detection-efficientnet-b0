# Model Checkpoints & Pretrained Weights

This directory holds the PyTorch model weight files (`.pth`) for the binary classification deepfake detectors based on **EfficientNet-B0**.

---

## 1. Checkpoint Files

| File Name | Model Variant | Validation Acc | Description |
|:---|:---|:---:|:---|
| `efficientnet_b0_baseline_best.pth` | Baseline | **99.35%** | EfficientNet-B0 trained on clean images with standard flip augmentation (epoch 5 peak). |
| `efficientnet_b0_robust_best.pth` | Robust | **99.53%** | EfficientNet-B0 fine-tuned with on-the-fly post-processing augmentations (epoch 5 peak). |

---

## 2. Generating Weights via Training

You can train both models directly from scratch using the provided reproducible pipelines:

### Baseline Model:
```bash
python src/train_baseline.py --epochs 5 --batch-size 32 --lr 1e-4
```
This saves the best checkpoint to `models/efficientnet_b0_baseline_best.pth`.

### Robust Model:
```bash
python src/train_robust.py --epochs 5 --batch-size 32 --lr 5e-5 --aug-prob 0.70
```
This initializes from the baseline weights and saves the checkpoint to `models/efficientnet_b0_robust_best.pth`.

---

## 3. Pretrained Weights Download / Linking

If you have already trained the models locally or received pretrained checkpoints:
Place the `.pth` files directly into this `models/` directory:

```text
models/
├── efficientnet_b0_baseline_best.pth
└── efficientnet_b0_robust_best.pth
```

If cloning in a local environment where `deepfake-project/models/` is present, copy or symlink them into this folder.
