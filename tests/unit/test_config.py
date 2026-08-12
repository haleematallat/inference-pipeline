from pathlib import Path

import pytest

from vision_pipeline.config.loader import load_config
from vision_pipeline.exceptions import ConfigurationError


def test_demo_config_is_valid():
    config = load_config(Path(__file__).parents[2] / "configs" / "demo.yaml")
    assert config.runners[0].type == "few_shot_classifier"
    assert config.runners[0].backend.device == "cpu"


def test_relative_paths_are_resolved():
    config_path = Path(__file__).parents[2] / "configs" / "demo.yaml"
    config = load_config(config_path)
    assert config.stream.input_dir.is_absolute()
    assert config.runners[0].backend.support_dir.is_absolute()


def test_triton_config_is_valid():
    config = load_config(Path(__file__).parents[2] / "configs" / "triton.yaml")
    backend = config.runners[0].backend
    assert backend.type == "triton"
    assert backend.triton is not None
    assert backend.triton.model_name == "image_encoder"


def test_missing_config_raises(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_unknown_config_field_is_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
stream:
  input_dir: input
  output_path: output.jsonl
  unexpected: true
runners:
  - type: few_shot_classifier
    name: test
    backend:
      support_dir: support
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="unexpected"):
        load_config(config_path)


def test_triton_settings_are_required(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
stream:
  input_dir: input
  output_path: output.jsonl
runners:
  - type: few_shot_classifier
    name: test
    backend:
      type: triton
      support_dir: support
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="triton settings are required"):
        load_config(config_path)
