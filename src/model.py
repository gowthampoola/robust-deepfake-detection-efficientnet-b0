"""
Model definition and loader utilities for EfficientNet-B0 Deepfake Detection.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models


def build_efficientnet_b0(num_classes=2, pretrained=True, dropout_rate=0.20):
    """
    Builds the EfficientNet-B0 binary deepfake classification model.

    Args:
        num_classes (int): Number of output classes (default: 2 -> FAKE vs REAL).
        pretrained (bool): Whether to initialize with ImageNet-1k default weights.
        dropout_rate (float): Dropout probability in the classifier head (default: 0.20).

    Returns:
        torch.nn.Module: Configured EfficientNet-B0 model.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # In EfficientNet-B0, classifier is:
    # (0): Dropout(p=0.2, inplace=True)
    # (1): Linear(in_features=1280, out_features=1000, bias=True)
    in_features = model.classifier[1].in_features
    if dropout_rate != 0.2:
        model.classifier[0] = nn.Dropout(p=dropout_rate, inplace=True)
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def load_model_checkpoint(checkpoint_path, device=None, num_classes=2):
    """
    Loads an EfficientNet-B0 checkpoint with mapped weights.

    Args:
        checkpoint_path (str or Path): Path to the saved .pth state dict.
        device (torch.device, optional): Device to map tensors to.
        num_classes (int): Number of target classes.

    Returns:
        torch.nn.Module: Loaded model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at: {checkpoint_path}\n"
            f"Please ensure you have trained the model or downloaded the pretrained checkpoint."
        )

    model = build_efficientnet_b0(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model
