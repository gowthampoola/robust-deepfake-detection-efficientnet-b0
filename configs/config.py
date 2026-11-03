"""
Configuration module for the EfficientNet-B0 Deepfake Detection Project.
Provides programmatic access to project defaults and paths.
"""

from pathlib import Path
import os
import yaml

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class ProjectConfig:
    """Project-wide configuration container."""

    # General
    SEED = 42
    IMAGE_SIZE = 224
    NUM_CLASSES = 2
    CLASS_NAMES = {0: "FAKE", 1: "REAL"}

    # Normalization (ImageNet)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Directories
    ROOT_DIR = PROJECT_ROOT
    DATA_DIR = PROJECT_ROOT / "data" / "real_vs_fake" / "real-vs-fake"
    TRAIN_DIR = DATA_DIR / "train"
    VALID_DIR = DATA_DIR / "valid"
    TEST_DIR = DATA_DIR / "test"

    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = PROJECT_ROOT / "logs"

    # Checkpoint paths
    BASELINE_CHECKPOINT = MODELS_DIR / "efficientnet_b0_baseline_best.pth"
    ROBUST_CHECKPOINT = MODELS_DIR / "efficientnet_b0_robust_best.pth"
    SANITY_CHECKPOINT = MODELS_DIR / "efficientnet_b0_baseline_sanity.pth"

    # Subset sizes
    TRAIN_PER_CLASS = 10000
    VALID_PER_CLASS = 2000
    TEST_PER_CLASS = 10000

    # Training parameters
    BATCH_SIZE = 32
    BASELINE_EPOCHS = 5
    BASELINE_LR = 0.0001

    ROBUST_EPOCHS = 5
    ROBUST_LR = 0.00005
    AUGMENTATION_PROBABILITY = 0.70

    # Post-processing stress-test parameters
    JPEG_QUALITY = 50
    RESIZE_DOWNSCALE = 112
    RESIZE_RESTORE = 256
    BLUR_RADIUS = 2.0
    NOISE_STD = 0.05
    CONTRAST_FACTOR = 1.5
    SHARPNESS_FACTOR = 2.0

    @classmethod
    def load_yaml(cls, yaml_path=None):
        """Loads configuration from YAML file if available."""
        if yaml_path is None:
            yaml_path = PROJECT_ROOT / "configs" / "config.yaml"
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return None
