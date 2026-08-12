from __future__ import annotations

import importlib
from collections.abc import Callable

from vision_pipeline.config.models import RunnerConfig
from vision_pipeline.exceptions import FactoryError
from vision_pipeline.runners.base import IInferenceRunner


class InferenceRunnerFactory:
    runners: dict[str, Callable[[RunnerConfig], IInferenceRunner]] = {}
    _builtins_loaded = False

    @classmethod
    def register(cls, runner_type: str) -> Callable[[type[IInferenceRunner]], type[IInferenceRunner]]:
        key = runner_type.strip().lower()
        if not key:
            raise FactoryError("runner type cannot be empty")

        def decorator(runner_class: type[IInferenceRunner]) -> type[IInferenceRunner]:
            if key in cls.runners:
                raise FactoryError(f"Inference runner already registered: {key}")
            cls.runners[key] = runner_class
            return runner_class

        return decorator

    @classmethod
    def _load_builtins(cls) -> None:
        if cls._builtins_loaded:
            return
        importlib.import_module("vision_pipeline.runners.few_shot_classification")
        cls._builtins_loaded = True

    @classmethod
    def get_runner(cls, runner_type: str) -> Callable[[RunnerConfig], IInferenceRunner]:
        cls._load_builtins()
        key = runner_type.strip().lower()
        if key not in cls.runners:
            supported = ", ".join(sorted(cls.runners)) or "none"
            raise FactoryError(f"Unknown inference runner '{runner_type}'. Supported: {supported}")
        return cls.runners[key]

    @classmethod
    def create(cls, config: RunnerConfig) -> IInferenceRunner:
        return cls.get_runner(config.type)(config)
