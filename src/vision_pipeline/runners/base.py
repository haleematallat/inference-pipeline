from abc import ABC, abstractmethod

from vision_pipeline.common.results import InferResult


class IInferenceRunner(ABC):
    def __init__(self, name: str):
        self.name = name
        self.started = False

    def start(self) -> None:
        self.started = True

    @abstractmethod
    def process_frame(self, input_frame: InferResult) -> InferResult:
        raise NotImplementedError

    def process_tracks(self, input_frame: InferResult) -> InferResult:
        return input_frame

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.stop()
