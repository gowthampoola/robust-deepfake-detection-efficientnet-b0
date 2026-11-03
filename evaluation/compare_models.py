"""
Baseline vs Robust EfficientNet-B0 Model Comparison.

Evaluates both baseline and robust models side-by-side across all 6 test conditions,
verifying robustness gains and reproducing the core empirical results of the paper (Table II).
"""

import sys
import argparse
from io import BytesIO
from pathlib import Path

# Resolve project root dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ("src", "evaluation", "training") else SCRIPT_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Baseline vs Robust Model Performance")
    parser.add_argument(
        "--test-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "real_vs_fake" / "real-vs-fake" / "test"),
        help="Path to clean test directory"
    )
    parser.add_argument(
        "--baseline-model", type=str,
        default=str(PROJECT_ROOT / "models" / "efficientnet_b0_baseline_best.pth"),
        help="Path to baseline model checkpoint"
    )
    parser.add_argument(
        "--robust-model", type=str,
        default=str(PROJECT_ROOT / "models" / "efficientnet_b0_robust_best.pth"),
        help="Path to robust model checkpoint"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--image-size", type=int, default=224, help="Input image dimension")
    return parser.parse_args()


# Normalization
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# Post-processing transformations
def jpeg_compression(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=50)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def resize_image(image):
    image = image.resize((112, 112), Image.Resampling.BILINEAR)
    image = image.resize((256, 256), Image.Resampling.BILINEAR)
    return image


def blur_image(image):
    return image.filter(ImageFilter.GaussianBlur(radius=2))


def add_noise(image):
    tensor = transforms.ToTensor()(image)
    noise = torch.randn_like(tensor) * 0.05
    tensor = torch.clamp(tensor + noise, 0.0, 1.0)
    return transforms.ToPILImage()(tensor)


def enhance_image(image):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    return image


class RobustnessDataset(Dataset):
    def __init__(self, root, processing=None, image_size=224):
        self.dataset = datasets.ImageFolder(root, transform=None)
        self.processing = processing
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = image.convert("RGB")

        if self.processing is not None:
            image = self.processing(image)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image = transforms.ToTensor()(image)
        image = normalize(image)
        return image, label


def load_model(checkpoint_path, device):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataset, name, device, batch_size=32):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=name, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("DEEPFAKE DETECTION - BASELINE VS ROBUST MODEL COMPARISON")
    print("=" * 75)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    test_dir = Path(args.test_dir)
    baseline_path = Path(args.baseline_model)
    robust_path = Path(args.robust_model)

    if not test_dir.exists():
        print(f"\n[ERROR] Test directory not found at: {test_dir}")
        print("Please check data/README.md for dataset setup instructions.")
        sys.exit(1)

    if not baseline_path.exists() or not robust_path.exists():
        print("\n[ERROR] One or both required model checkpoints are missing:")
        print(f"  Baseline: {baseline_path} (Exists: {baseline_path.exists()})")
        print(f"  Robust:   {robust_path} (Exists: {robust_path.exists()})")
        print("Please train the models or ensure weights are placed in models/.")
        sys.exit(1)

    # Load models
    print("\nLoading models...")
    baseline_model = load_model(baseline_path, device)
    robust_model = load_model(robust_path, device)
    print("Both models loaded successfully.")

    conditions = {
        "Clean": None,
        "JPEG Compression": jpeg_compression,
        "Resizing": resize_image,
        "Blur": blur_image,
        "Noise": add_noise,
        "Enhancement": enhance_image
    }

    results = {}

    for condition, proc_fn in conditions.items():
        print("\n" + "=" * 75)
        print(f"EVALUATING TEST CONDITION: {condition}")
        print("=" * 75)

        dataset = RobustnessDataset(test_dir, processing=proc_fn, image_size=args.image_size)

        # Baseline evaluation
        if condition == "Noise":
            torch.manual_seed(42)
        base_acc = evaluate_model(baseline_model, dataset, f"Baseline [{condition}]", device, args.batch_size)

        # Robust evaluation
        if condition == "Noise":
            torch.manual_seed(42)
        rob_acc = evaluate_model(robust_model, dataset, f"Robust [{condition}]", device, args.batch_size)

        delta = rob_acc - base_acc
        results[condition] = {
            "baseline": base_acc,
            "robust": rob_acc,
            "delta": delta
        }

        print(f"  Baseline Accuracy : {base_acc:.2f}%")
        print(f"  Robust Accuracy   : {rob_acc:.2f}%")
        print(f"  Improvement       : {delta:+.2f}%")

    # Paper comparison table reproduction
    print("\n")
    print("=" * 75)
    print("BASELINE VS ROBUST - FINAL COMPARATIVE RESULTS (PAPER TABLE II)")
    print("=" * 75)
    print(f"{'Condition':<25} {'Baseline':>12} {'Robust':>12} {'Improvement':>15}")
    print("-" * 75)

    for cond, metrics in results.items():
        print(
            f"{cond:<25} "
            f"{metrics['baseline']:>11.2f}% "
            f"{metrics['robust']:>11.2f}% "
            f"{metrics['delta']:>+14.2f}%"
        )
    print("=" * 75)
    print("COMPARISON COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
