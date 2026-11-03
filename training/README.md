# Training Pipelines

This directory contains standalone execution scripts to train and fine-tune the EfficientNet-B0 deepfake detection models.

---

## 1. Environment Verification
Verify your PyTorch environment, GPU acceleration (CUDA), and dataset directory:
```bash
python training/test_setup.py
```

---

## 2. Baseline Model Training
Trains the baseline EfficientNet-B0 on the clean training subset (20,000 images: 10k real + 10k fake) for 5 epochs with standard horizontal flips:
```bash
python training/train_baseline.py --epochs 5 --batch-size 32 --lr 1e-4
```
- **Inputs**: Pristine 224x224 images.
- **Output Checkpoint**: `models/efficientnet_b0_baseline_best.pth`
- **Validation Accuracy**: 99.35% (Peak at epoch 5).

---

## 3. Degradation-Aware Robust Fine-Tuning
Fine-tunes the saved baseline checkpoint for 5 additional epochs under dynamic post-processing corruptions ($p=0.70$ perturbation probability):
```bash
python training/train_robust.py --epochs 5 --batch-size 32 --lr 5e-5 --aug-prob 0.70
```
- **Perturbations Applied**: JPEG compression ($Q \in \{30, 40, 50, 60\}$), spatial downsampling (96, 112, 128), Gaussian blur ($r \in [0.5, 2.0]$), Gaussian noise ($\sigma \in [0.02, 0.08]$), and sharpness/contrast enhancement.
- **Output Checkpoint**: `models/efficientnet_b0_robust_best.pth`
- **Validation Accuracy**: 99.53% (Peak at epoch 5 on clean validation split).
