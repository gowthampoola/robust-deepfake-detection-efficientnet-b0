"""
Single-Image Prediction using Trained Baseline EfficientNet-B0 Model.

Supports both interactive console prompt and command-line arguments.
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
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_baseline_best.pth"
IMAGE_SIZE = 224
CLASS_NAMES = {0: "FAKE", 1: "REAL"}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Deepfake with Baseline EfficientNet-B0")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input image")
    parser.add_argument("--model-path", "-m", type=str, default=str(MODEL_PATH), help="Path to model weights")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("DEEPFAKE IMAGE PREDICTION (BASELINE EFFICIENTNET-B0)")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\n[ERROR] Model checkpoint not found at: {model_path}")
        print("Please train the baseline model first (python src/train_baseline.py) or download weights.")
        sys.exit(1)

    print(f"\nLoading model: {model_path.name}...")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # Image source
    if args.image:
        image_path_str = args.image.strip().strip('"')
    else:
        image_path_str = input("\nEnter image path: ").strip().strip('"')

    image_path = Path(image_path_str)
    if not image_path.exists():
        print(f"\n[ERROR] Image not found: {image_path}")
        sys.exit(1)

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"\n[ERROR] Could not open image: {e}")
        sys.exit(1)

    print(f"\nEvaluating Image: {image_path.name}")
    print(f"Original resolution: {image.size[0]} x {image.size[1]}")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    idx = predicted_class.item()
    conf_pct = confidence.item() * 100.0
    label = CLASS_NAMES[idx]

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Prediction: {label}")
    print(f"Confidence: {conf_pct:.2f}%")
    print("\nClass Probabilities:")
    print(f"  FAKE: {probabilities[0][0].item() * 100:.2f}%")
    print(f"  REAL: {probabilities[0][1].item() * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
