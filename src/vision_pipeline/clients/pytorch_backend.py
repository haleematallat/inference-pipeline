from pathlib import Path

from PIL import Image
from torch import nn

from vision_pipeline.clients.base import InferenceBackend
from vision_pipeline.clients.factory import InferenceBackendFactory
from vision_pipeline.common.results import Prediction
from vision_pipeline.config.models import BackendConfig
from vision_pipeline.models.artifacts import load_support_set
from vision_pipeline.models.encoders import StatisticalEncoder, TorchvisionEncoder
from vision_pipeline.models.preprocessing import ImagePreprocessor
from vision_pipeline.models.prototypical_network import PrototypicalNetwork


@InferenceBackendFactory.register("local_pytorch")
class PyTorchBackend(InferenceBackend):
    def __init__(self, config: BackendConfig):
        super().__init__()
        self.config = config
        self.preprocessor = ImagePreprocessor(
            size=config.preprocessing.size,
            normalize=config.preprocessing.normalize,
            mean=config.preprocessing.mean,
            std=config.preprocessing.std,
        )
        if config.encoder.type == "statistical":
            encoder: nn.Module = StatisticalEncoder()
        else:
            encoder = TorchvisionEncoder(
                name=config.encoder.name,
                pretrained=config.encoder.pretrained,
            )
        self.model = PrototypicalNetwork(
            encoder=encoder,
            device=config.device,
            similarity_metric=config.similarity_metric,
            top_k=config.top_k,
            rejection_threshold=config.rejection_threshold,
        )
        if config.encoder.checkpoint is not None:
            self.model.load_checkpoint(config.encoder.checkpoint)

    def start(self) -> None:
        support_images, labels = load_support_set(self.config.support_dir, self.preprocessor)
        self.model.fit_support(support_images, labels)
        self.started = True

    def predict(self, images: list[str | Path | Image.Image]) -> list[list[Prediction]]:
        if not self.started:
            raise RuntimeError("backend has not been started")
        if not images:
            return []

        predictions = []
        for start in range(0, len(images), self.config.batch_size):
            batch_images = images[start : start + self.config.batch_size]
            batch = self.preprocessor.batch(batch_images)
            predictions.extend(self.model.predict(batch))
        return predictions

    def stop(self) -> None:
        self.model.invalidate_prototypes()
        self.started = False
