from vision_pipeline.clients.factory import InferenceBackendFactory
from vision_pipeline.common.results import InferResult
from vision_pipeline.config.models import RunnerConfig
from vision_pipeline.runners.base import IInferenceRunner
from vision_pipeline.runners.factory import InferenceRunnerFactory


@InferenceRunnerFactory.register("few_shot_classifier")
class FewShotClassificationRunner(IInferenceRunner):
    def __init__(self, config: RunnerConfig):
        super().__init__(name=config.name)
        self.backend = InferenceBackendFactory.create(config.backend)

    def start(self) -> None:
        self.backend.start()
        self.started = True

    def process_frame(self, input_frame: InferResult) -> InferResult:
        if not self.started:
            raise RuntimeError(f"runner has not been started: {self.name}")
        if not input_frame.source:
            return input_frame

        result = self.backend.predict([input_frame.source])
        input_frame.predictions = result[0] if result else []
        input_frame.metadata["runner"] = self.name
        return input_frame

    def stop(self) -> None:
        self.backend.close()
        self.started = False
