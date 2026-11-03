"""
Interactive & Scriptable Demonstration: Baseline vs Robust EfficientNet-B0 on Post-Processed Images.

Runs both baseline and robust models on an image before and after selected real-world
post-processing, printing comparative predictions and saving visual artifacts to results/.
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
from torchvision import transforms, models
from PIL import Image, ImageEnhance, ImageFilter

BASELINE_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_baseline_best.pth"
ROBUST_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_robust_best.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
CLASS_NAMES = {0: "FAKE", 1: "REAL"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def jpeg_compression(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=50)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def resize_image(image):
    image = image.resize((112, 112), Image.Resampling.BILINEAR)
    return image.resize((256, 256), Image.Resampling.BILINEAR)


def blur_image(image):
    return image.filter(ImageFilter.GaussianBlur(radius=2))


def add_noise(image):
    torch.manual_seed(42)
    tensor = transforms.ToTensor()(image)
    tensor = torch.clamp(tensor + torch.randn_like(tensor) * 0.05, 0.0, 1.0)
    return transforms.ToPILImage()(tensor)


def enhance_image(image):
    image = ImageEnhance.Contrast(image).enhance(1.5)
    return ImageEnhance.Sharpness(image).enhance(2.0)


POSTPROCESSING = {
    "1": ("Clean (no post-processing)", None, "clean"),
    "2": ("JPEG compression (quality 50)", jpeg_compression, "jpeg_q50"),
    "3": ("Resizing (256 -> 112 -> 256)", resize_image, "resize_112"),
    "4": ("Gaussian blur (radius 2)", blur_image, "gaussian_blur_r2"),
    "5": ("Gaussian noise (scale 0.05)", add_noise, "gaussian_noise_005"),
    "6": ("Contrast + sharpness enhancement", enhance_image, "contrast_sharpness"),
}


def load_model(checkpoint, device):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
    index = int(torch.argmax(probabilities).item())
    return CLASS_NAMES[index], float(probabilities[index]) * 100, {
        "FAKE": float(probabilities[0]) * 100,
        "REAL": float(probabilities[1]) * 100,
    }


def print_prediction(label, result):
    predicted, confidence, probs = result
    print(f"{label}: {predicted:<4} ({confidence:.2f}% confidence) | FAKE: {probs['FAKE']:.2f}% | REAL: {probs['REAL']:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Live Deepfake Robustness Demonstration")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input test image")
    parser.add_argument("--choice", "-c", type=str, default=None, choices=["1", "2", "3", "4", "5", "6"],
                        help="Perturbation choice (1-6)")
    parser.add_argument("--no-plot", action="store_true", help="Do not display matplotlib GUI window")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 75)
    print("LIVE DEEPFAKE ROBUSTNESS DEMO: BASELINE vs ROBUST EFFICIENTNET-B0")
    print("=" * 75)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    if not BASELINE_PATH.exists() or not ROBUST_PATH.exists():
        print("\n[ERROR] One or both required model checkpoints are missing:")
        print(f"  Baseline: {BASELINE_PATH} (Exists: {BASELINE_PATH.exists()})")
        print(f"  Robust:   {ROBUST_PATH} (Exists: {ROBUST_PATH.exists()})")
        print("Please train the models or ensure weights are placed in models/.")
        sys.exit(1)

    # Image selection
    if args.image:
        image_path_str = args.image.strip().strip('"')
    else:
        image_path_str = input("\nEnter path to a test image: ").strip().strip('"')

    image_path = Path(image_path_str).expanduser()
    if not image_path.is_file():
        print(f"\n[ERROR] Image not found: {image_path}")
        sys.exit(1)

    # Condition selection
    if args.choice:
        choice = args.choice
    else:
        print("\nChoose post-processing perturbation:")
        for key, (name, _, _) in POSTPROCESSING.items():
            print(f"  {key}. {name}")
        choice = input("Enter choice (1-6, default 4 for Gaussian blur): ").strip() or "4"

    if choice not in POSTPROCESSING:
        print(f"[ERROR] Invalid choice '{choice}'. Must be between 1 and 6.")
        sys.exit(1)

    effect_name, effect, safe_suffix = POSTPROCESSING[choice]

    with Image.open(image_path) as opened:
        original = opened.convert("RGB")
    processed = original.copy() if effect is None else effect(original.copy())

    print("\nLoading checkpoints...")
    baseline = load_model(BASELINE_PATH, device)
    robust = load_model(ROBUST_PATH, device)
    print("Both models loaded successfully.")

    print("\n" + "=" * 75)
    print("CLEAN IMAGE PREDICTIONS")
    print("=" * 75)
    print_prediction("Baseline", predict(baseline, original, device))
    print_prediction("Robust  ", predict(robust, original, device))

    print("\n" + "=" * 75)
    print(f"POST-PROCESSED IMAGE PREDICTIONS ({effect_name})")
    print("=" * 75)
    print_prediction("Baseline", predict(baseline, processed, device))
    print_prediction("Robust  ", predict(robust, processed, device))

    # Save visual preview
    preview_path = RESULTS_DIR / f"demo_processed_{safe_suffix}.jpg"
    processed.save(preview_path, quality=95)
    print(f"\nProcessed preview saved to: {preview_path}")

    if not args.no_plot:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original)
            axes[0].set_title("Original Clean Image")
            axes[0].axis("off")
            axes[1].imshow(processed)
            axes[1].set_title(f"Post-Processed: {effect_name}")
            axes[1].axis("off")
            fig.suptitle("Robustness Comparison Before and After Post-Processing")
            fig.tight_layout()
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
