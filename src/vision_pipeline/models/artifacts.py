from pathlib import Path

import torch

from vision_pipeline.exceptions import ModelArtifactError
from vision_pipeline.models.preprocessing import ImagePreprocessor

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".webp"}


def discover_support_set(root: str | Path) -> list[tuple[Path, str]]:
    support_root = Path(root)
    if not support_root.is_dir():
        raise ModelArtifactError(f"Support directory not found: {support_root}")

    items = []
    for class_dir in sorted(path for path in support_root.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((image_path, class_dir.name))

    if not items:
        raise ModelArtifactError(f"No support images found under: {support_root}")
    return items


def load_support_set(
    root: str | Path,
    preprocessor: ImagePreprocessor,
) -> tuple[torch.Tensor, list[str]]:
    support_items = discover_support_set(root)
    images = preprocessor.batch([path for path, _ in support_items])
    labels = [label for _, label in support_items]
    return images, labels
