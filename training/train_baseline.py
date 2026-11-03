"""
Baseline EfficientNet-B0 Training for Deepfake Detection.

Trains a standard transfer-learning EfficientNet-B0 model on clean images
with standard data augmentation (RandomHorizontalFlip) to establish a baseline.
"""

import sys
import random
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
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline EfficientNet-B0 Deepfake Detector")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "real_vs_fake" / "real-vs-fake"),
                        help="Path to real_vs_fake root containing train/ and valid/")
    parser.add_argument("--model-dir", type=str, default=str(PROJECT_ROOT / "models"),
                        help="Directory to save the trained model checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--train-per-class", type=int, default=10000, help="Train samples per class (default: 10000)")
    parser.add_argument("--valid-per-class", type=int, default=2000, help="Valid samples per class (default: 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def create_balanced_subset(dataset, images_per_class, seed=42):
    """Creates a deterministic class-balanced subset from an ImageFolder dataset."""
    class_indices = {class_id: [] for class_id in range(len(dataset.classes))}

    for index, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(index)

    selected_indices = []
    rng = random.Random(seed)

    for class_id in sorted(class_indices):
        indices = list(class_indices[class_id])
        rng.shuffle(indices)
        selected = indices[:images_per_class]
        selected_indices.extend(selected)
        print(f"  {dataset.classes[class_id]}: {len(selected):,} images")

    rng.shuffle(selected_indices)
    return Subset(dataset, selected_indices)


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
    print("DEEPFAKE DETECTION - EFFICIENTNET-B0 BASELINE TRAINING")
    print("=" * 70)
    print(f"Device:           {device}")
    if torch.cuda.is_available():
        print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"Random Seed:      {args.seed}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Epochs:           {args.epochs}")
    print(f"Train/Class:      {args.train_per_class}")
    print(f"Valid/Class:      {args.valid_per_class}")

    # Paths
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "efficientnet_b0_baseline_best.pth"

    if not train_dir.exists() or not valid_dir.exists():
        print(f"\n[ERROR] Dataset directory not found at: {data_dir}")
        print("Please check data/README.md for download instructions.")
        sys.exit(1)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
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
    train_dataset = create_balanced_subset(train_dataset_full, args.train_per_class, seed=args.seed)

    print("\nCreating balanced validation subset:")
    valid_dataset = create_balanced_subset(valid_dataset_full, args.valid_per_class, seed=args.seed)

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

    # Model architecture
    print("\nLoading EfficientNet-B0 (pretrained on ImageNet-1k)...")
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)
    print("EfficientNet-B0 loaded successfully.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_validation_accuracy = 0.0

    # Training loop
    for epoch in range(args.epochs):
        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 70)

        # Train phase
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
        validation_loss = 0.0
        validation_correct = 0
        validation_total = 0

        with torch.no_grad():
            progress_val = tqdm(valid_loader, desc=f"Epoch {epoch+1} [Val]")
            for images, labels in progress_val:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                validation_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                validation_total += labels.size(0)
                validation_correct += (predicted == labels).sum().item()

        validation_loss /= len(valid_loader)
        validation_accuracy = 100.0 * validation_correct / validation_total

        print("\nEpoch Results:")
        print(f"  Training Loss:       {train_loss:.4f}")
        print(f"  Training Accuracy:   {train_accuracy:.2f}%")
        print(f"  Validation Loss:     {validation_loss:.4f}")
        print(f"  Validation Accuracy: {validation_accuracy:.2f}%")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"  [*] New best model saved to: {best_model_path.name} (Val Acc: {best_validation_accuracy:.2f}%)")

    print("\n" + "=" * 70)
    print("BASELINE TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_validation_accuracy:.2f}%")
    print(f"Checkpoint Saved To:      {best_model_path}")


if __name__ == "__main__":
    main()
