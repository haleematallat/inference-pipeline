from abc import ABC, abstractmethod

from vision_pipeline.common.results import InferResult


class IStreamHandler(ABC):
    def __init__(self):
        self.started = False

    def start(self) -> None:
        self.started = True

    @abstractmethod
    def get_data(self, timeout: float = 0) -> InferResult | None:
        raise NotImplementedError

    @abstractmethod
    def push_data(self, infer_result: InferResult) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.stop()
