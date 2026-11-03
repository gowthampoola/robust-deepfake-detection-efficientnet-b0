"""
Dataset loading, balanced subset sampling, and on-the-fly robust augmentations.
"""

from pathlib import Path
import random
import io
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image, ImageEnhance, ImageFilter

try:
    from src.utils import (
        IMAGE_SIZE,
        MEAN,
        STD,
        jpeg_compression,
        resize_image,
        blur_image,
        add_noise,
        enhance_image
    )
except ImportError:
    from utils import (
        IMAGE_SIZE,
        MEAN,
        STD,
        jpeg_compression,
        resize_image,
        blur_image,
        add_noise,
        enhance_image
    )


# Standard transforms
train_clean_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


class RandomRobustAugmentation:
    """
    Randomly applies one of five real-world post-processing effects during training:
    - JPEG compression (Q in {30, 40, 50, 60})
    - Spatial downscaling (to {96, 112, 128} and restoration to 224)
    - Gaussian blur (r in {0.5, 1.0, 1.5, 2.0})
    - Additive Gaussian noise (std in [0.02, 0.08])
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
            smaller_size = random.choice([96, 112, 128])
            img = img.resize((smaller_size, smaller_size), Image.Resampling.BILINEAR)
            return img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

        elif augmentation == "blur":
            radius = random.choice([0.5, 1.0, 1.5, 2.0])
            return img.filter(ImageFilter.GaussianBlur(radius=radius))

        elif augmentation == "noise":
            tensor = transforms.ToTensor()(img)
            noise_level = random.uniform(0.02, 0.08)
            noise = torch.randn_like(tensor) * noise_level
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)
            return transforms.ToPILImage()(tensor)

        elif augmentation == "enhancement":
            contrast_factor = random.uniform(1.1, 1.5)
            sharpness_factor = random.uniform(1.2, 2.0)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
            enhancer = ImageEnhance.Sharpness(img)
            return enhancer.enhance(sharpness_factor)

        return img


class RobustnessDataset(Dataset):
    """
    Dataset wrapper applying a deterministic post-processing operation before normalization.
    """

    def __init__(self, base_dataset, post_processing_fn=None, transform=None):
        self.base_dataset = base_dataset
        self.post_processing_fn = post_processing_fn
        self.transform = transform if transform is not None else eval_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        path, label = self.base_dataset.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.post_processing_fn is not None:
                img = self.post_processing_fn(img)
            tensor = self.transform(img)
            return tensor, label


def create_balanced_subset(dataset, images_per_class, seed=42):
    """
    Creates a class-balanced subset from an ImageFolder dataset.

    Args:
        dataset (datasets.ImageFolder): Original dataset.
        images_per_class (int): Number of images to sample per class.
        seed (int): Random seed for reproducibility.

    Returns:
        Subset: Balanced PyTorch dataset subset.
    """
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

    rng.shuffle(selected_indices)
    return Subset(dataset, selected_indices)
