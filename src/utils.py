"""
Common utilities: device configuration, seed initialization, post-processing transformations,
and evaluation metrics.
"""

from pathlib import Path
from io import BytesIO
import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter


# Default ImageNet statistics
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_NAMES = {0: "FAKE", 1: "REAL"}


def set_seed(seed=42):
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Returns CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# Real-World Post-Processing Transformations
# ------------------------------------------------------------------------------

def jpeg_compression(image, quality=50):
    """
    Simulates JPEG compression artifacts in memory via DCT quantization.

    Args:
        image (PIL.Image): Input image.
        quality (int): Compression quality factor (1-100). Default: 50.

    Returns:
        PIL.Image: Compressed RGB image.
    """
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def resize_image(image, downscale_size=(112, 112), original_size=(256, 256)):
    """
    Simulates spatial resizing / transmission downsampling followed by restoration.

    Args:
        image (PIL.Image): Input image.
        downscale_size (tuple): Intermediate downscaled resolution. Default: (112, 112).
        original_size (tuple): Target restored resolution. Default: (256, 256).

    Returns:
        PIL.Image: Resized and restored RGB image.
    """
    image = image.resize(downscale_size, Image.Resampling.BILINEAR)
    image = image.resize(original_size, Image.Resampling.BILINEAR)
    return image


def blur_image(image, radius=2.0):
    """
    Simulates optical blur or anti-aliasing filtering.

    Args:
        image (PIL.Image): Input image.
        radius (float): Gaussian blur kernel radius. Default: 2.0.

    Returns:
        PIL.Image: Blurred RGB image.
    """
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def add_noise(image, std=0.05, seed=None):
    """
    Adds additive zero-mean white Gaussian noise to normalized pixel intensities.

    Args:
        image (PIL.Image): Input image.
        std (float): Noise standard deviation. Default: 0.05.
        seed (int, optional): Optional seed for reproducible noise generation.

    Returns:
        PIL.Image: Noisy RGB image clamped to [0.0, 1.0].
    """
    if seed is not None:
        torch.manual_seed(seed)
    tensor = transforms.ToTensor()(image)
    noise = torch.randn_like(tensor) * std
    tensor = torch.clamp(tensor + noise, 0.0, 1.0)
    return transforms.ToPILImage()(tensor)


def enhance_image(image, contrast_factor=1.5, sharpness_factor=2.0):
    """
    Simulates contrast expansion followed by high-boost unsharp masking.

    Args:
        image (PIL.Image): Input image.
        contrast_factor (float): Contrast expansion factor. Default: 1.5.
        sharpness_factor (float): Sharpness amplification factor. Default: 2.0.

    Returns:
        PIL.Image: Enhanced RGB image.
    """
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness_factor)
    return image


def calculate_relative_error_reduction(baseline_acc, robust_acc):
    """
    Calculates Relative Error Reduction (RER %):
    RER = (Error_baseline - Error_robust) / Error_baseline * 100
    """
    err_base = 1.0 - (baseline_acc / 100.0 if baseline_acc > 1.0 else baseline_acc)
    err_rob = 1.0 - (robust_acc / 100.0 if robust_acc > 1.0 else robust_acc)
    if err_base <= 0:
        return 0.0
    return ((err_base - err_rob) / err_base) * 100.0
