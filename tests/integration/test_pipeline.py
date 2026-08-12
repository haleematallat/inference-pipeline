from vision_pipeline.common.results import InferResult
from vision_pipeline.pipeline import PipelineRunner
from vision_pipeline.runners.base import IInferenceRunner
from vision_pipeline.streams.memory import InMemoryStreamHandler


class RecordingRunner(IInferenceRunner):
    def process_frame(self, input_frame):
        input_frame.metadata.setdefault("order", []).append(self.name)
        return input_frame


def test_pipeline_runner_ordering():
    stream = InMemoryStreamHandler()
    stream.put_input(InferResult(1, "image.ppm"))
    pipeline = PipelineRunner(
        stream,
        [RecordingRunner("first"), RecordingRunner("second")],
    )
    assert pipeline.run() == 1
    assert stream.outputs[0].metadata["order"] == ["first", "second"]


def test_pipeline_graceful_shutdown():
    stream = InMemoryStreamHandler()
    runner = RecordingRunner("runner")
    pipeline = PipelineRunner(stream, [runner])
    pipeline.run()
    assert stream.started is False
    assert runner.started is False
    assert pipeline.keep_running is False


def test_pipeline_processes_multiple_inputs():
    stream = InMemoryStreamHandler()
    stream.put_input(InferResult(1, "one.ppm"))
    stream.put_input(InferResult(2, "two.ppm"))
    pipeline = PipelineRunner(stream, [RecordingRunner("runner")])
    assert pipeline.run() == 2
    assert len(stream.outputs) == 2
