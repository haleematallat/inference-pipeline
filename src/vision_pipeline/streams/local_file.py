import json
from pathlib import Path
from typing import TextIO

from vision_pipeline.common.results import InferResult
from vision_pipeline.config.models import StreamConfig
from vision_pipeline.exceptions import StreamError
from vision_pipeline.models.artifacts import IMAGE_EXTENSIONS
from vision_pipeline.streams.base import IStreamHandler
from vision_pipeline.streams.factory import StreamHandlerFactory


@StreamHandlerFactory.register("local_file")
class LocalFileStreamHandler(IStreamHandler):
    def __init__(self, config: StreamConfig):
        super().__init__()
        self.input_dir = config.input_dir
        self.output_path = config.output_path
        self.capacity = config.capacity
        self.input_paths: list[Path] = []
        self.index = 0
        self.output_file: TextIO | None = None

    def start(self) -> None:
        if not self.input_dir.is_dir():
            raise StreamError(f"Input directory not found: {self.input_dir}")
        self.input_paths = sorted(
            path
            for path in self.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.input_paths:
            raise StreamError(f"No input images found in: {self.input_dir}")
        if len(self.input_paths) > self.capacity:
            raise StreamError(
                f"Input contains {len(self.input_paths)} images but stream capacity is {self.capacity}"
            )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_path.open("w", encoding="utf-8")
        self.index = 0
        self.started = True

    def get_data(self, timeout: float = 0) -> InferResult | None:
        if not self.started:
            raise RuntimeError("stream handler has not been started")
        if self.index >= len(self.input_paths):
            return None
        image_path = self.input_paths[self.index]
        result = InferResult(timestamp=self.index, source=str(image_path))
        self.index += 1
        return result

    def push_data(self, infer_result: InferResult) -> None:
        if not self.started or self.output_file is None:
            raise RuntimeError("stream handler has not been started")
        self.output_file.write(json.dumps(infer_result.to_dict(), sort_keys=True) + "\n")
        self.output_file.flush()

    def stop(self) -> None:
        if self.output_file is not None:
            self.output_file.close()
            self.output_file = None
        self.started = False
