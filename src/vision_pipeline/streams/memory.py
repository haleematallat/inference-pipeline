from queue import Empty, Queue

from vision_pipeline.common.results import InferResult
from vision_pipeline.streams.base import IStreamHandler


class InMemoryStreamHandler(IStreamHandler):
    def __init__(self, capacity: int = 64):
        super().__init__()
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.input_queue: Queue[InferResult] = Queue(maxsize=capacity)
        self.outputs: list[InferResult] = []

    def put_input(self, infer_result: InferResult) -> None:
        self.input_queue.put_nowait(infer_result)

    def get_data(self, timeout: float = 0) -> InferResult | None:
        if not self.started:
            raise RuntimeError("stream handler has not been started")
        try:
            return self.input_queue.get(timeout=timeout) if timeout > 0 else self.input_queue.get_nowait()
        except Empty:
            return None

    def push_data(self, infer_result: InferResult) -> None:
        if not self.started:
            raise RuntimeError("stream handler has not been started")
        self.outputs.append(infer_result)
