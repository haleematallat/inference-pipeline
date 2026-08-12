from __future__ import annotations

import importlib
from collections.abc import Callable

from vision_pipeline.config.models import StreamConfig
from vision_pipeline.exceptions import FactoryError
from vision_pipeline.streams.base import IStreamHandler


class StreamHandlerFactory:
    handlers: dict[str, Callable[[StreamConfig], IStreamHandler]] = {}
    _builtins_loaded = False

    @classmethod
    def register(cls, handler_type: str) -> Callable[[type[IStreamHandler]], type[IStreamHandler]]:
        key = handler_type.strip().lower()
        if not key:
            raise FactoryError("stream handler type cannot be empty")

        def decorator(handler_class: type[IStreamHandler]) -> type[IStreamHandler]:
            if key in cls.handlers:
                raise FactoryError(f"Stream handler already registered: {key}")
            cls.handlers[key] = handler_class
            return handler_class

        return decorator

    @classmethod
    def _load_builtins(cls) -> None:
        if cls._builtins_loaded:
            return
        importlib.import_module("vision_pipeline.streams.local_file")
        cls._builtins_loaded = True

    @classmethod
    def get_handler(cls, handler_type: str) -> Callable[[StreamConfig], IStreamHandler]:
        cls._load_builtins()
        key = handler_type.strip().lower()
        if key not in cls.handlers:
            supported = ", ".join(sorted(cls.handlers)) or "none"
            raise FactoryError(f"Unknown stream handler '{handler_type}'. Supported: {supported}")
        return cls.handlers[key]

    @classmethod
    def create(cls, config: StreamConfig) -> IStreamHandler:
        return cls.get_handler(config.type)(config)
