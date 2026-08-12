from pathlib import Path

import pytest
import torch
from torch import nn

from vision_pipeline.exceptions import ModelArtifactError
from vision_pipeline.models.artifacts import discover_support_set, load_support_set
from vision_pipeline.models.encoders import StatisticalEncoder
from vision_pipeline.models.preprocessing import ImagePreprocessor
from vision_pipeline.models.prototypical_network import PrototypicalNetwork


class MeanEncoder(nn.Module):
    def forward(self, images):
        return images.mean(dim=(2, 3))


def make_images():
    return torch.tensor(
        [
            [[[1.0]], [[0.0]], [[0.0]]],
            [[[0.8]], [[0.2]], [[0.0]]],
            [[[0.0]], [[0.0]], [[1.0]]],
            [[[0.0]], [[0.2]], [[0.8]]],
        ]
    )


def fitted_model(**kwargs):
    model = PrototypicalNetwork(MeanEncoder(), **kwargs)
    model.fit_support(make_images(), ["red", "red", "blue", "blue"])
    return model


def test_statistical_encoder_output_shape():
    encoder = StatisticalEncoder()
    output = encoder(torch.ones((2, 3, 8, 8)))
    assert output.shape == (2, 18)


def test_embedding_normalization():
    model = PrototypicalNetwork(MeanEncoder())
    embeddings = model.embed(make_images())
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(4))


def test_prototype_calculation():
    model = fitted_model()
    assert model.prototypes is not None
    assert model.prototypes.shape == (2, 3)
    assert model.class_names == ["red", "blue"]


def test_top_k_ranking():
    model = fitted_model(top_k=2)
    predictions = model.predict(make_images()[:1])[0]
    assert [prediction.class_name for prediction in predictions] == ["red", "blue"]
    assert [prediction.rank for prediction in predictions] == [1, 2]


def test_unknown_class_rejection():
    model = fitted_model(rejection_threshold=1.01)
    prediction = model.predict(make_images()[:1])[0][0]
    assert prediction.accepted is False


def test_known_class_is_accepted():
    model = fitted_model(rejection_threshold=0.8)
    prediction = model.predict(make_images()[:1])[0][0]
    assert prediction.class_name == "red"
    assert prediction.accepted is True


def test_batched_predictions():
    model = fitted_model()
    predictions = model.predict(make_images())
    assert len(predictions) == 4
    assert predictions[3][0].class_name == "blue"


def test_empty_query_returns_empty_result():
    model = fitted_model()
    empty = torch.empty((0, 3, 1, 1))
    assert model.predict(empty) == []


def test_prototype_cache_invalidation():
    model = fitted_model()
    model.invalidate_prototypes()
    assert model.prototypes is None
    with pytest.raises(RuntimeError, match="not been calculated"):
        model.predict(make_images()[:1])


def test_cpu_device_execution():
    model = fitted_model(device="cpu")
    assert model.prototypes is not None
    assert model.prototypes.device.type == "cpu"


def test_euclidean_similarity():
    model = fitted_model(similarity_metric="euclidean", rejection_threshold=-0.5)
    prediction = model.predict(make_images()[2:3])[0][0]
    assert prediction.class_name == "blue"
    assert prediction.accepted is True


def test_support_set_discovery(tmp_path, solid_image):
    class_dir = tmp_path / "class-a"
    class_dir.mkdir()
    solid_image((255, 0, 0)).save(class_dir / "image.png")
    items = discover_support_set(tmp_path)
    assert items == [(class_dir / "image.png", "class-a")]


def test_support_set_loading(tmp_path, solid_image):
    class_dir = tmp_path / "class-a"
    class_dir.mkdir()
    solid_image((255, 0, 0)).save(class_dir / "image.png")
    images, labels = load_support_set(
        tmp_path,
        ImagePreprocessor(size=(8, 8), normalize=False),
    )
    assert images.shape == (1, 3, 8, 8)
    assert labels == ["class-a"]


def test_support_count_must_match_labels():
    model = PrototypicalNetwork(MeanEncoder())
    with pytest.raises(ValueError, match="counts do not match"):
        model.fit_support(make_images(), ["red"])


def test_missing_checkpoint_is_rejected(tmp_path):
    model = PrototypicalNetwork(MeanEncoder())
    with pytest.raises(ModelArtifactError, match="Checkpoint not found"):
        model.load_checkpoint(Path(tmp_path) / "missing.pt")
