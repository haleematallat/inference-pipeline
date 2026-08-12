from pathlib import Path

import numpy as np
import torch
from PIL import Image


class ImagePreprocessor:
    def __init__(
        self,
        size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.size = size
        self.normalize = normalize
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: str | Path | Image.Image) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened_image:
                rgb_image = opened_image.convert("RGB")
                return self._to_tensor(rgb_image)
        return self._to_tensor(image.convert("RGB"))

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        if self.normalize:
            tensor = (tensor - self.mean) / self.std
        return tensor

    def batch(self, images: list[str | Path | Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty((0, 3, self.size[0], self.size[1]), dtype=torch.float32)
        return torch.stack([self(image) for image in images])
