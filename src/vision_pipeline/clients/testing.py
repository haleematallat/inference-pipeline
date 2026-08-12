from pathlib import Path

from PIL import Image

from vision_pipeline.clients.base import InferenceBackend
from vision_pipeline.common.results import Prediction


class DeterministicTestBackend(InferenceBackend):
    def __init__(self, class_name: str = "test-class", similarity: float = 1.0):
        super().__init__()
        self.class_name = class_name
        self.similarity = similarity

    def predict(self, images: list[str | Path | Image.Image]) -> list[list[Prediction]]:
        if not self.started:
            raise RuntimeError("backend has not been started")
        return [[Prediction(self.class_name, self.similarity, accepted=True)] for _ in images]
