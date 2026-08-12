from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

from vision_pipeline.common.results import Prediction


class InferenceBackend(ABC):
    def __init__(self):
        self.started = False

    def start(self) -> None:
        self.started = True

    @abstractmethod
    def predict(self, images: list[str | Path | Image.Image]) -> list[list[Prediction]]:
        raise NotImplementedError

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "InferenceBackend":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
