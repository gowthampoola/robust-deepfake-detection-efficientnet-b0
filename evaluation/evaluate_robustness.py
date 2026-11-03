"""
Robustness Evaluation under Real-World Post-Processing Perturbations.

Evaluates a model across 6 distinct conditions:
1. Clean Test Images (Unperturbed)
2. JPEG Compression (Quality = 50)
3. Spatial Resizing (256 -> 112 -> 256)
4. Gaussian Blur (Radius = 2.0)
5. Additive Gaussian Noise (std = 0.05)
6. Contrast (1.5x) and Sharpness (2.0x) Enhancement
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
    parser = argparse.ArgumentParser(description="Evaluate Model Robustness across Perturbations")
    parser.add_argument(
        "--test-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "real_vs_fake" / "real-vs-fake" / "test"),
        help="Path to clean test directory"
    )
    parser.add_argument(
        "--model-path", type=str,
        default=str(PROJECT_ROOT / "models" / "efficientnet_b0_robust_best.pth"),
        help="Path to model checkpoint .pth"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--image-size", type=int, default=224, help="Input image dimension")
    return parser.parse_args()


# ------------------------------------------------------------
# Normalization Transform
# ------------------------------------------------------------
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# ------------------------------------------------------------
# Post-Processing Functions
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Dataset Wrapper
# ------------------------------------------------------------
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

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=name):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    print(f"Results: Total={total:,} | Correct={correct:,} | Accuracy={accuracy:.2f}%")
    return accuracy


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("DEEPFAKE DETECTION - ROBUSTNESS EVALUATION SUITE")
    print("=" * 70)
    print(f"Device:     {device}")
    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name(0)}")

    test_dir = Path(args.test_dir)
    model_path = Path(args.model_path)

    if not test_dir.exists():
        print(f"\n[ERROR] Test dataset not found at: {test_dir}")
        print("Please check data/README.md for download instructions.")
        sys.exit(1)

    if not model_path.exists():
        print(f"\n[ERROR] Model checkpoint not found at: {model_path}")
        print("Please train the robust model or download weights first.")
        sys.exit(1)

    # Load model
    print(f"\nLoading model: {model_path.name}...")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # Conditions
    conditions = [
        ("Clean", None, "CLEAN TEST DATA"),
        ("JPEG Compression", jpeg_compression, "JPEG COMPRESSION - QUALITY 50"),
        ("Resizing", resize_image, "RESIZING - 256 -> 112 -> 256"),
        ("Blur", blur_image, "GAUSSIAN BLUR - RADIUS 2.0"),
        ("Noise", add_noise, "GAUSSIAN NOISE - STD 0.05"),
        ("Enhancement", enhance_image, "CONTRAST (1.5x) + SHARPNESS (2.0x)")
    ]

    results = {}
    for name, proc_fn, title in conditions:
        if name == "Noise":
            torch.manual_seed(42)

        ds = RobustnessDataset(test_dir, processing=proc_fn, image_size=args.image_size)
        results[name] = evaluate_model(model, ds, title, device, batch_size=args.batch_size)

    # Summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Evaluated Checkpoint: {model_path.name}")
    print("-" * 70)
    for name, acc in results.items():
        print(f"  {name:<25} : {acc:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
