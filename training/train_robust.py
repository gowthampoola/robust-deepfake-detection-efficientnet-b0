"""
Robust Deepfake Detection Training using EfficientNet-B0 with Data Augmentation.

Incorporates real-world post-processing perturbations (JPEG compression,
spatial resizing, Gaussian blur, Gaussian noise, contrast/sharpness enhancement)
into the training pipeline to build resistance against online degradation.
"""

import sys
import random
import io
import argparse
from pathlib import Path

# Resolve project root dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ("src", "evaluation", "training") else SCRIPT_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train Robust EfficientNet-B0 Deepfake Detector")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "real_vs_fake" / "real-vs-fake"),
                        help="Path to real_vs_fake root containing train/ and valid/")
    parser.add_argument("--model-dir", type=str, default=str(PROJECT_ROOT / "models"),
                        help="Directory for loading/saving model checkpoints")
    parser.add_argument("--init-weights", type=str, default="baseline", choices=["baseline", "imagenet"],
                        help="Weight initialization: 'baseline' (start from trained baseline) or 'imagenet'")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 0.00005)")
    parser.add_argument("--train-per-class", type=int, default=10000, help="Train samples per class (default: 10000)")
    parser.add_argument("--valid-per-class", type=int, default=2000, help="Valid samples per class (default: 2000)")
    parser.add_argument("--aug-prob", type=float, default=0.70, help="Probability of applying robust augmentation (default: 0.70)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


class RandomRobustAugmentation:
    """
    Randomly applies real-world post-processing effects:
    - JPEG compression (Quality in {30, 40, 50, 60})
    - Resizing (Intermediate {96, 112, 128} and restoration to 224)
    - Gaussian blur (Radius in {0.5, 1.0, 1.5, 2.0})
    - Gaussian noise (Additive std in [0.02, 0.08])
    - Contrast (1.1-1.5x) and Sharpness (1.2-2.0x) enhancement
    """

    def __init__(self, probability=0.70):
        self.probability = probability

    def __call__(self, img):
        if random.random() > self.probability:
            return img

        augmentation = random.choice([
            "jpeg",
            "resize",
            "blur",
            "noise",
            "enhancement"
        ])

        if augmentation == "jpeg":
            quality = random.choice([30, 40, 50, 60])
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")

        elif augmentation == "resize":
            small_size = random.choice([96, 112, 128])
            img = img.resize((small_size, small_size), Image.Resampling.BILINEAR)
            return img.resize((224, 224), Image.Resampling.BILINEAR)

        elif augmentation == "blur":
            radius = random.choice([0.5, 1.0, 1.5, 2.0])
            return img.filter(ImageFilter.GaussianBlur(radius=radius))

        elif augmentation == "noise":
            tensor = transforms.ToTensor()(img)
            noise_strength = random.uniform(0.02, 0.08)
            noise = torch.randn_like(tensor) * noise_strength
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)
            return transforms.ToPILImage()(tensor)

        elif augmentation == "enhancement":
            contrast_factor = random.uniform(1.1, 1.5)
            sharpness_factor = random.uniform(1.2, 2.0)
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)
            return ImageEnhance.Sharpness(img).enhance(sharpness_factor)

        return img


def create_balanced_indices(dataset, samples_per_class, seed=42):
    """Creates a deterministic balanced subset of indices from an ImageFolder dataset."""
    class_indices = {class_id: [] for class_id in range(len(dataset.classes))}

    for index, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(index)

    selected_indices = []
    rng = random.Random(seed)

    for class_id in sorted(class_indices):
        available = list(class_indices[class_id])
        rng.shuffle(available)
        selected = available[:min(samples_per_class, len(available))]
        selected_indices.extend(selected)
        print(f"  {dataset.classes[class_id]}: {len(selected):,} images")

    rng.shuffle(selected_indices)
    return selected_indices


def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("DEEPFAKE DETECTION - ROBUST EFFICIENTNET-B0 TRAINING")
    print("=" * 70)
    print(f"Device:           {device}")
    if torch.cuda.is_available():
        print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"Random Seed:      {args.seed}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Epochs:           {args.epochs}")
    print(f"Augmentation Prob:{args.aug_prob}")
    print(f"Train/Class:      {args.train_per_class}")
    print(f"Valid/Class:      {args.valid_per_class}")

    # Paths
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    baseline_model_path = model_dir / "efficientnet_b0_baseline_best.pth"
    robust_model_path = model_dir / "efficientnet_b0_robust_best.pth"

    if not train_dir.exists() or not valid_dir.exists():
        print(f"\n[ERROR] Dataset directory not found at: {data_dir}")
        print("Please check data/README.md for download instructions.")
        sys.exit(1)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        RandomRobustAugmentation(probability=args.aug_prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    print("\nLoading datasets...")
    train_dataset_full = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset_full = datasets.ImageFolder(valid_dir, transform=valid_transform)

    print(f"  Classes:          {train_dataset_full.classes}")
    print(f"  Class mapping:    {train_dataset_full.class_to_idx}")
    print(f"  Full training:    {len(train_dataset_full):,} images")
    print(f"  Full validation:  {len(valid_dataset_full):,} images")

    print("\nCreating balanced training subset:")
    train_indices = create_balanced_indices(train_dataset_full, args.train_per_class, seed=args.seed)

    print("\nCreating balanced validation subset:")
    valid_indices = create_balanced_indices(valid_dataset_full, args.valid_per_class, seed=args.seed)

    train_dataset = Subset(train_dataset_full, train_indices)
    valid_dataset = Subset(valid_dataset_full, valid_indices)

    print(f"\nFinal dataset sizes:")
    print(f"  Training images:   {len(train_dataset):,}")
    print(f"  Validation images: {len(valid_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Load EfficientNet-B0
    print("\nInitializing EfficientNet-B0...")
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    # Initialize from baseline model if selected and available
    if args.init_weights == "baseline" and baseline_model_path.exists():
        print(f"Loading pretrained baseline weights from: {baseline_model_path}")
        model.load_state_dict(torch.load(baseline_model_path, map_location=device))
        print("Baseline weights loaded successfully for robust fine-tuning.")
    else:
        if args.init_weights == "baseline":
            print(f"[NOTE] Baseline checkpoint not found at: {baseline_model_path}")
            print("Training will start directly from ImageNet pretrained weights.")
        else:
            print("Initializing directly from ImageNet pretrained weights as requested.")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_validation_accuracy = 0.0

    print("\n" + "=" * 70)
    print("ROBUST TRAINING STARTED")
    print("=" * 70)

    for epoch in range(args.epochs):
        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 70)

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = 100.0 * correct / total
            progress.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{accuracy:.2f}%")

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total

        # Validation phase
        model.eval()
        valid_loss_total = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Val]"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                valid_loss_total += loss.item()
                _, predicted = torch.max(outputs, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        valid_loss = valid_loss_total / len(valid_loader)
        valid_accuracy = 100.0 * valid_correct / valid_total

        print("\nEpoch Results:")
        print(f"  Training Loss:       {train_loss:.4f}")
        print(f"  Training Accuracy:   {train_accuracy:.2f}%")
        print(f"  Validation Loss:     {valid_loss:.4f}")
        print(f"  Validation Accuracy: {valid_accuracy:.2f}%")

        if valid_accuracy > best_validation_accuracy:
            best_validation_accuracy = valid_accuracy
            torch.save(model.state_dict(), robust_model_path)
            print(f"  [*] New best robust model saved to: {robust_model_path.name} (Val Acc: {best_validation_accuracy:.2f}%)")

    print("\n" + "=" * 70)
    print("ROBUST TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_validation_accuracy:.2f}%")
    print(f"Robust Checkpoint Saved To: {robust_model_path}")


if __name__ == "__main__":
    main()
