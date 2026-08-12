import os
from pathlib import Path
from typing import Any

import yaml

from vision_pipeline.config.models import PipelineConfig
from vision_pipeline.exceptions import ConfigurationError


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _resolve_path(value: Path, base_path: Path) -> Path:
    if value.is_absolute():
        return value
    return (base_path / value).resolve()


def _resolve_paths(config: PipelineConfig, config_path: Path) -> PipelineConfig:
    base_path = config_path.parent
    config.stream.input_dir = _resolve_path(config.stream.input_dir, base_path)
    config.stream.output_path = _resolve_path(config.stream.output_path, base_path)
    for runner in config.runners:
        runner.backend.support_dir = _resolve_path(runner.backend.support_dir, base_path)
        checkpoint = runner.backend.encoder.checkpoint
        if checkpoint is not None:
            runner.backend.encoder.checkpoint = _resolve_path(checkpoint, base_path)
    return config


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file) or {}
        config = PipelineConfig.model_validate(_expand_environment(raw_config))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Invalid configuration: {error}") from error

    return _resolve_paths(config, config_path)
