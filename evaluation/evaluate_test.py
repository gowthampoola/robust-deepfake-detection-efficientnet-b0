"""
Evaluation on Clean Test Dataset.

Evaluates an EfficientNet-B0 model checkpoint on the standard unperturbed
test partition, reporting Accuracy, Precision, Recall, F1-Score,
and Confusion Matrix.
"""

import sys
import argparse
from pathlib import Path

# Resolve project root dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ("src", "evaluation", "training") else SCRIPT_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Model on Clean Test Set")
    parser.add_argument(
        "--test-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "real_vs_fake" / "real-vs-fake" / "test"),
        help="Path to clean test directory"
    )
    parser.add_argument(
        "--model-path", type=str,
        default=str(PROJECT_ROOT / "models" / "efficientnet_b0_baseline_best.pth"),
        help="Path to model checkpoint .pth"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--image-size", type=int, default=224, help="Input image dimension")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("DEEPFAKE DETECTION - FINAL TEST SET EVALUATION")
    print("=" * 70)
    print(f"Device:     {device}")
    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name(0)}")

    test_dir = Path(args.test_dir)
    model_path = Path(args.model_path)

    if not test_dir.exists():
        print(f"\n[ERROR] Test directory not found at: {test_dir}")
        print("Please verify the dataset is extracted in data/real_vs_fake/real-vs-fake/")
        sys.exit(1)

    if not model_path.exists():
        print(f"\n[ERROR] Model checkpoint not found at: {model_path}")
        print("Please train the model first or provide the correct checkpoint path with --model-path.")
        sys.exit(1)

    # Transform
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"\nLoading test dataset from: {test_dir}...")
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    print(f"  Classes:           {test_dataset.classes}")
    print(f"  Class mapping:     {test_dataset.class_to_idx}")
    print(f"  Total test images: {len(test_dataset):,}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"\nLoading model: {model_path.name}...")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    print("\nStarting evaluation...")
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="binary", pos_label=1)
    recall = recall_score(all_labels, all_predictions, average="binary", pos_label=1)
    f1 = f1_score(all_labels, all_predictions, average="binary", pos_label=1)
    cm = confusion_matrix(all_labels, all_predictions)

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(f"Model Checkpoint:      {model_path.name}")
    print(f"Total Test Images:     {total:,}")
    print(f"Correct Predictions:   {correct:,}")
    print(f"Incorrect Predictions: {total - correct:,}")
    print("-" * 70)
    print(f"Accuracy:              {accuracy * 100:.2f}%")
    print(f"Precision (Real):      {precision * 100:.2f}%")
    print(f"Recall (Real):         {recall * 100:.2f}%")
    print(f"F1-Score:              {f1 * 100:.2f}%")

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    print("              Predicted")
    print("             FAKE    REAL")
    print(f"Actual FAKE  {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"Actual REAL  {cm[1][0]:6d}  {cm[1][1]:6d}")

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(all_labels, all_predictions, target_names=["FAKE", "REAL"], digits=4))
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
