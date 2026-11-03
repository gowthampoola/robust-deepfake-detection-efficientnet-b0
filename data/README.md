# Dataset Setup & Organization

This repository utilizes the benchmark **140k Real and Fake Faces** dataset created by Bojan Tunguz and xhlulu, derived from NVIDIA's StyleGAN (fake faces) and Flickr-Faces-HQ (real faces).

- **Kaggle Source:** [140k Real and Fake Faces (Kaggle)](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Total Images:** 140,000 balanced faces (70,000 Real, 70,000 Synthesized/Fake)
- **Resolution:** 256x256 pixels, RGB 8-bit

---

## 1. Directory Structure

Download and extract the dataset into the `data/` directory so that the structure matches the standard PyTorch `ImageFolder` layout expected by the codebase:

```text
data/
└── real_vs_fake/
    └── real-vs-fake/
        ├── train/
        │   ├── fake/   [50,000 images]
        │   └── real/   [50,000 images]
        ├── valid/
        │   ├── fake/   [10,000 images]
        │   └── real/   [10,000 images]
        └── test/
            ├── fake/   [10,000 images]
            └── real/   [10,000 images]
```

> **Class Label Mapping:**
> - `fake` (or index `0`): AI-synthesized deepfake face images
> - `real` (or index `1`): Authentic real human face images

---

## 2. Fast Download via Kaggle CLI

If you have the Kaggle CLI configured on your system:

```bash
# Navigate to repository root
cd data

# Download the Kaggle dataset archive
kaggle datasets download -d xhlulu/140k-real-and-fake-faces

# Unzip the downloaded file
unzip -q 140k-real-and-fake-faces.zip -d real_vs_fake/
```

---

## 3. Experimental Subsets

To ensure computationally efficient and reproducible experiments, the training scripts extract deterministic balanced subsets using seed `42`:

- **Training:** 10,000 real + 10,000 fake = **20,000 images**
- **Validation:** 2,000 real + 2,000 fake = **4,000 images**
- **Testing:** 10,000 real + 10,000 fake = **20,000 images** (Full clean test set)

This sampling strategy is handled automatically within `src/train_baseline.py` and `src/train_robust.py`.
