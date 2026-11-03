"""
Diagnostic script to verify PyTorch, CUDA, GPU acceleration, and dataset structure.
Can be executed from repository root or from inside src/.
"""

import sys
from pathlib import Path

# Resolve project root dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ("src", "evaluation", "training") else SCRIPT_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision import datasets, transforms


# -----------------------------
# Paths
# -----------------------------
DATA_DIR = PROJECT_ROOT / "data" / "real_vs_fake" / "real-vs-fake"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("DEEPFAKE DETECTION - ENVIRONMENT & SETUP TEST")
print("=" * 60)

print(f"Repository Root: {PROJECT_ROOT}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
print(f"Active Device:   {device}")

if torch.cuda.is_available():
    print(f"GPU Model:       {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory:      {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# -----------------------------
# Check Dataset Directory
# -----------------------------
print("\n" + "-" * 60)
print("DATASET VERIFICATION")
print("-" * 60)
print(f"Target dataset directory: {DATA_DIR}")

if not DATA_DIR.exists() or not TRAIN_DIR.exists():
    print("\n[NOTE] Dataset not detected at the expected local path.")
    print(f"Expected directory: {DATA_DIR}")
    print("Please follow data/README.md to download and extract the dataset:")
    print("  Kaggle: 'xhlulu/140k-real-and-fake-faces'")
    print("Environment test completed (PyTorch & CUDA are functional).")
    print("=" * 60)
    sys.exit(0)

# -----------------------------
# Image transformation
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# Load datasets
# -----------------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
valid_dataset = datasets.ImageFolder(VALID_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

# -----------------------------
# Dataset information
# -----------------------------
print("\nDataset information:")
print(f"  Classes:         {train_dataset.classes}")
print(f"  Class mapping:   {train_dataset.class_to_idx}")
print(f"  Training images:   {len(train_dataset):,}")
print(f"  Validation images: {len(valid_dataset):,}")
print(f"  Testing images:    {len(test_dataset):,}")

# -----------------------------
# Test sample image
# -----------------------------
image, label = train_dataset[0]
print("\nSample image test:")
print(f"  Tensor shape: {tuple(image.shape)}")
print(f"  Label ID:     {label}")
print(f"  Class:        {train_dataset.classes[label]}")

# -----------------------------
# GPU tensor transfer test
# -----------------------------
batch = image.unsqueeze(0).to(device)
print("\nTensor Device test:")
print(f"  Batch shape:  {tuple(batch.shape)}")
print(f"  Batch device: {batch.device}")

print("\n" + "=" * 60)
print("SETUP TEST COMPLETED SUCCESSFULLY")
print("=" * 60)
