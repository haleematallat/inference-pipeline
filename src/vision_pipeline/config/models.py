from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size: tuple[int, int] = (224, 224)
    normalize: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class EncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["statistical", "torchvision"] = "statistical"
    name: str = "mobilenet_v3_small"
    pretrained: bool = False
    checkpoint: Path | None = None


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "local_pytorch"
    device: str = "cpu"
    batch_size: int = Field(default=8, ge=1)
    support_dir: Path
    similarity_metric: Literal["cosine", "euclidean"] = "cosine"
    top_k: int = Field(default=1, ge=1)
    rejection_threshold: float = 0.0
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)

    @model_validator(mode="after")
    def validate_device(self) -> BackendConfig:
        if not (self.device == "cpu" or self.device == "cuda" or self.device.startswith("cuda:")):
            raise ValueError("device must be cpu, cuda, or cuda:<index>")
        return self


class RunnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    name: str
    backend: BackendConfig


class StreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "local_file"
    input_dir: Path
    output_path: Path
    capacity: int = Field(default=64, ge=1)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 7
    logging_level: str = "INFO"
    stream: StreamConfig
    runners: list[RunnerConfig] = Field(min_length=1)
