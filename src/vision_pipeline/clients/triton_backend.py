from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from vision_pipeline.clients.base import InferenceBackend
from vision_pipeline.clients.factory import InferenceBackendFactory
from vision_pipeline.common.results import Prediction
from vision_pipeline.config.models import BackendConfig, TritonConfig
from vision_pipeline.exceptions import ConfigurationError, InferenceError
from vision_pipeline.models.artifacts import load_support_set
from vision_pipeline.models.preprocessing import ImagePreprocessor
from vision_pipeline.models.prototypical_network import PrototypicalNetwork


class TritonEncoder(nn.Module):
    def __init__(self, config: TritonConfig):
        super().__init__()
        self.config = config
        self.client_module: Any = None
        self.client: Any = None

    def _load_client_module(self) -> Any:
        try:
            import tritonclient.grpc as grpcclient
        except ImportError as error:
            raise ConfigurationError(
                "Triton backend requires installation with the 'triton' extra"
            ) from error
        return grpcclient

    def start(self) -> None:
        self.client_module = self._load_client_module()
        self.client = self.client_module.InferenceServerClient(url=self.config.server_url)
        deadline = time.monotonic() + self.config.connect_timeout
        last_error = None
        while time.monotonic() < deadline:
            try:
                if self.client.is_server_ready() and self.client.is_model_ready(
                    self.config.model_name,
                    self.config.model_version,
                ):
                    return
            except Exception as error:
                last_error = error
            time.sleep(0.1)
        self.close()
        raise InferenceError("Triton server or model did not become ready") from last_error

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.client is None or self.client_module is None:
            raise RuntimeError("Triton encoder has not been started")
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")

        array = np.ascontiguousarray(images.detach().cpu().numpy(), dtype=np.float32)
        infer_input = self.client_module.InferInput(self.config.input_name, array.shape, "FP32")
        infer_input.set_data_from_numpy(array)
        output = self.client_module.InferRequestedOutput(self.config.output_name)
        try:
            response = self.client.infer(
                model_name=self.config.model_name,
                model_version=self.config.model_version,
                inputs=[infer_input],
                outputs=[output],
                client_timeout=self.config.request_timeout,
            )
        except Exception as error:
            raise InferenceError("Triton inference request failed") from error

        embeddings = response.as_numpy(self.config.output_name)
        if embeddings is None:
            raise InferenceError(f"Triton response is missing output: {self.config.output_name}")
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != images.shape[0]:
            raise InferenceError("Triton returned an invalid embedding shape")
        if not np.isfinite(embeddings).all():
            raise InferenceError("Triton returned non-finite embeddings")
        return torch.from_numpy(embeddings.copy())

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
        self.client = None
        self.client_module = None


@InferenceBackendFactory.register("triton")
class TritonBackend(InferenceBackend):
    def __init__(self, config: BackendConfig):
        super().__init__()
        if config.triton is None:
            raise ConfigurationError("triton settings are required for the triton backend")
        self.config = config
        self.preprocessor = ImagePreprocessor(
            size=config.preprocessing.size,
            normalize=config.preprocessing.normalize,
            mean=config.preprocessing.mean,
            std=config.preprocessing.std,
        )
        self.encoder = TritonEncoder(config.triton)
        self.model = PrototypicalNetwork(
            encoder=self.encoder,
            device="cpu",
            similarity_metric=config.similarity_metric,
            top_k=config.top_k,
            rejection_threshold=config.rejection_threshold,
        )

    def start(self) -> None:
        self.encoder.start()
        try:
            support_images, labels = load_support_set(self.config.support_dir, self.preprocessor)
            self.model.fit_support(support_images, labels)
        except Exception:
            self.encoder.close()
            raise
        self.started = True

    def predict(self, images: list[str | Path | Image.Image]) -> list[list[Prediction]]:
        if not self.started:
            raise RuntimeError("backend has not been started")
        if not images:
            return []

        predictions = []
        for start in range(0, len(images), self.config.batch_size):
            batch_images = images[start : start + self.config.batch_size]
            batch = self.preprocessor.batch(batch_images)
            predictions.extend(self.model.predict(batch))
        return predictions

    def stop(self) -> None:
        self.model.invalidate_prototypes()
        self.encoder.close()
        self.started = False
