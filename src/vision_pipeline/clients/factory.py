from __future__ import annotations

import importlib
from collections.abc import Callable

from vision_pipeline.clients.base import InferenceBackend
from vision_pipeline.config.models import BackendConfig
from vision_pipeline.exceptions import FactoryError


class InferenceBackendFactory:
    backends: dict[str, Callable[[BackendConfig], InferenceBackend]] = {}
    _builtins_loaded = False

    @classmethod
    def register(cls, backend_type: str) -> Callable[[type[InferenceBackend]], type[InferenceBackend]]:
        key = backend_type.strip().lower()
        if not key:
            raise FactoryError("backend type cannot be empty")

        def decorator(backend_class: type[InferenceBackend]) -> type[InferenceBackend]:
            if key in cls.backends:
                raise FactoryError(f"Inference backend already registered: {key}")
            cls.backends[key] = backend_class
            return backend_class

        return decorator

    @classmethod
    def _load_builtins(cls) -> None:
        if cls._builtins_loaded:
            return
        importlib.import_module("vision_pipeline.clients.pytorch_backend")
        importlib.import_module("vision_pipeline.clients.triton_backend")
        cls._builtins_loaded = True

    @classmethod
    def get_backend(cls, backend_type: str) -> Callable[[BackendConfig], InferenceBackend]:
        cls._load_builtins()
        key = backend_type.strip().lower()
        if key not in cls.backends:
            supported = ", ".join(sorted(cls.backends)) or "none"
            raise FactoryError(f"Unknown inference backend '{backend_type}'. Supported: {supported}")
        return cls.backends[key]

    @classmethod
    def create(cls, config: BackendConfig) -> InferenceBackend:
        return cls.get_backend(config.type)(config)
