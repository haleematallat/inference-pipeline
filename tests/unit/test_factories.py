from pathlib import Path

import pytest

from vision_pipeline.clients.factory import InferenceBackendFactory
from vision_pipeline.config.models import BackendConfig, RunnerConfig
from vision_pipeline.exceptions import FactoryError
from vision_pipeline.runners.base import IInferenceRunner
from vision_pipeline.runners.factory import InferenceRunnerFactory


class TemporaryRunner(IInferenceRunner):
    def __init__(self, config=None):
        super().__init__(name="temporary")

    def process_frame(self, input_frame):
        return input_frame


def test_builtin_runner_is_registered():
    runner_class = InferenceRunnerFactory.get_runner("few_shot_classifier")
    assert runner_class.__name__ == "FewShotClassificationRunner"


def test_triton_backend_is_registered():
    backend_class = InferenceBackendFactory.get_backend("triton")
    assert backend_class.__name__ == "TritonBackend"


def test_custom_runner_registration():
    key = "temporary-runner"
    InferenceRunnerFactory.runners.pop(key, None)
    registered = InferenceRunnerFactory.register(key)(TemporaryRunner)
    assert registered is TemporaryRunner
    assert InferenceRunnerFactory.get_runner(key) is TemporaryRunner
    InferenceRunnerFactory.runners.pop(key)


def test_duplicate_runner_registration_is_rejected():
    key = "duplicate-runner"
    InferenceRunnerFactory.runners.pop(key, None)
    InferenceRunnerFactory.register(key)(TemporaryRunner)
    with pytest.raises(FactoryError, match="already registered"):
        InferenceRunnerFactory.register(key)(TemporaryRunner)
    InferenceRunnerFactory.runners.pop(key)


def test_unknown_runner_lists_supported_types():
    with pytest.raises(FactoryError, match="few_shot_classifier"):
        InferenceRunnerFactory.get_runner("missing")


def test_factory_constructs_runner():
    config = RunnerConfig(
        type="few_shot_classifier",
        name="test-runner",
        backend=BackendConfig(support_dir=Path("support")),
    )
    runner = InferenceRunnerFactory.create(config)
    assert runner.name == "test-runner"
