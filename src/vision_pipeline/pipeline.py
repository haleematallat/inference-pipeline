import logging

from vision_pipeline.runners.base import IInferenceRunner
from vision_pipeline.streams.base import IStreamHandler

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(
        self,
        stream_handler: IStreamHandler,
        inference_runners: list[IInferenceRunner],
    ):
        if not inference_runners:
            raise ValueError("at least one inference runner is required")
        self.stream_handler = stream_handler
        self.inference_runners = inference_runners
        self.keep_running = False

    def start(self) -> None:
        self.stream_handler.start()
        try:
            for inference_runner in self.inference_runners:
                inference_runner.start()
        except Exception:
            self.stop()
            raise
        self.keep_running = True

    def run(self) -> int:
        processed = 0
        self.start()
        try:
            while self.keep_running:
                infer_result = self.stream_handler.get_data()
                if infer_result is None:
                    break
                for inference_runner in self.inference_runners:
                    infer_result = inference_runner.process_frame(infer_result)
                    infer_result = inference_runner.process_tracks(infer_result)
                self.stream_handler.push_data(infer_result)
                processed += 1
        finally:
            self.stop()
        logger.info("[PipelineRunner] processed %s inputs", processed)
        return processed

    def stop(self) -> None:
        self.keep_running = False
        for inference_runner in reversed(self.inference_runners):
            inference_runner.close()
        self.stream_handler.close()
