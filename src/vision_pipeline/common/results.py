from __future__ import annotations

from typing import Any


class Prediction:
    def __init__(self, class_name: str, similarity: float, accepted: bool, rank: int = 1):
        self.class_name = class_name
        self.similarity = float(similarity)
        self.accepted = bool(accepted)
        self.rank = rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "similarity": self.similarity,
            "accepted": self.accepted,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Prediction:
        return cls(
            class_name=str(value["class_name"]),
            similarity=float(value["similarity"]),
            accepted=bool(value["accepted"]),
            rank=int(value.get("rank", 1)),
        )


class InferResult:
    def __init__(
        self,
        timestamp: int,
        source: str,
        predictions: list[Prediction] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.timestamp = timestamp
        self.source = source
        self.predictions = predictions or []
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InferResult:
        return cls(
            timestamp=int(value["timestamp"]),
            source=str(value["source"]),
            predictions=[Prediction.from_dict(item) for item in value.get("predictions", [])],
            metadata=dict(value.get("metadata", {})),
        )
