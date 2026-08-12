from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn

from vision_pipeline.common.results import Prediction
from vision_pipeline.exceptions import ModelArtifactError


class PrototypicalNetwork:
    def __init__(
        self,
        encoder: nn.Module,
        device: str = "cpu",
        similarity_metric: str = "cosine",
        top_k: int = 1,
        rejection_threshold: float = 0.0,
    ):
        if similarity_metric not in {"cosine", "euclidean"}:
            raise ValueError("similarity_metric must be cosine or euclidean")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but unavailable: {device}")

        self.encoder = encoder.to(device).eval()
        self.device = torch.device(device)
        self.similarity_metric = similarity_metric
        self.top_k = top_k
        self.rejection_threshold = rejection_threshold
        self.class_names: list[str] = []
        self.prototypes: torch.Tensor | None = None

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        if images.shape[0] == 0:
            return torch.empty((0, 0), dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            embeddings = self.encoder(images.to(self.device))
            embeddings = embeddings.flatten(start_dim=1)
            return functional.normalize(embeddings, p=2, dim=1)

    def fit_support(self, support_images: torch.Tensor, labels: list[str]) -> torch.Tensor:
        if support_images.shape[0] == 0:
            raise ValueError("support_images cannot be empty")
        if support_images.shape[0] != len(labels):
            raise ValueError("support image and label counts do not match")

        support_embeddings = self.embed(support_images)
        self.class_names = list(dict.fromkeys(labels))
        prototypes = []
        for class_name in self.class_names:
            indices = [index for index, label in enumerate(labels) if label == class_name]
            class_embeddings = support_embeddings[indices]
            prototype = class_embeddings.mean(dim=0, keepdim=True)
            prototypes.append(functional.normalize(prototype, p=2, dim=1))
        self.prototypes = torch.cat(prototypes, dim=0)
        return self.prototypes

    def invalidate_prototypes(self) -> None:
        self.class_names = []
        self.prototypes = None

    def predict(self, query_images: torch.Tensor) -> list[list[Prediction]]:
        if self.prototypes is None:
            raise RuntimeError("support prototypes have not been calculated")
        if query_images.shape[0] == 0:
            return []

        query_embeddings = self.embed(query_images)
        if self.similarity_metric == "cosine":
            scores = query_embeddings @ self.prototypes.T
        else:
            scores = -torch.cdist(query_embeddings, self.prototypes, p=2)

        result = []
        top_k = min(self.top_k, len(self.class_names))
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)
        for query_scores, query_indices in zip(top_scores, top_indices, strict=True):
            predictions = []
            for rank, (score, class_index) in enumerate(
                zip(query_scores.tolist(), query_indices.tolist(), strict=True), start=1
            ):
                predictions.append(
                    Prediction(
                        class_name=self.class_names[class_index],
                        similarity=score,
                        accepted=rank == 1 and score >= self.rejection_threshold,
                        rank=rank,
                    )
                )
            result.append(predictions)
        return result

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise ModelArtifactError(f"Checkpoint not found: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise ModelArtifactError(f"Could not load checkpoint: {path}") from error

        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else None
        if not isinstance(state_dict, dict):
            raise ModelArtifactError("Checkpoint must contain a state_dict")
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval()
        self.invalidate_prototypes()
