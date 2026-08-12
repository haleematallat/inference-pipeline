import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vision_pipeline.clients.triton_backend import TritonBackend, TritonEncoder
from vision_pipeline.config.models import BackendConfig, PreprocessingConfig, TritonConfig
from vision_pipeline.exceptions import InferenceError
from vision_pipeline.models.encoders import StatisticalEncoder


class FakeInferInput:
    def __init__(self, name, shape, datatype):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.data = None

    def set_data_from_numpy(self, data):
        self.data = data


class FakeRequestedOutput:
    def __init__(self, name):
        self.name = name


class FakeResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def as_numpy(self, name):
        return self.embeddings if name == "embeddings" else None


class FakeClient:
    def __init__(self, url):
        self.url = url
        self.closed = False

    def is_server_ready(self):
        return True

    def is_model_ready(self, model_name, model_version):
        return model_name == "image_encoder" and model_version == "1"

    def infer(self, model_name, model_version, inputs, outputs, client_timeout):
        images = torch.from_numpy(inputs[0].data)
        embeddings = StatisticalEncoder()(images).numpy()
        return FakeResponse(embeddings)

    def close(self):
        self.closed = True


class FakeGrpc:
    InferInput = FakeInferInput
    InferRequestedOutput = FakeRequestedOutput
    InferenceServerClient = FakeClient


def make_encoder():
    encoder = TritonEncoder(TritonConfig())
    encoder._load_client_module = lambda: FakeGrpc
    return encoder


def test_triton_encoder_request():
    encoder = make_encoder()
    encoder.start()
    images = torch.rand((2, 3, 8, 8))
    expected = StatisticalEncoder()(images)
    assert torch.allclose(encoder(images), expected)


def test_triton_encoder_close():
    encoder = make_encoder()
    encoder.start()
    client = encoder.client
    encoder.close()
    assert client.closed is True
    assert encoder.client is None


def test_triton_encoder_rejects_invalid_output():
    encoder = make_encoder()
    encoder.start()
    encoder.client.infer = lambda **kwargs: FakeResponse(np.zeros((1, 18), dtype=np.float32))
    with pytest.raises(InferenceError, match="invalid embedding shape"):
        encoder(torch.rand((2, 3, 8, 8)))


def test_triton_backend_predictions(tmp_path, solid_image):
    support_dir = tmp_path / "support"
    (support_dir / "amber").mkdir(parents=True)
    (support_dir / "blue").mkdir()
    solid_image((225, 145, 25)).save(support_dir / "amber" / "one.ppm")
    solid_image((25, 65, 215)).save(support_dir / "blue" / "one.ppm")
    config = BackendConfig(
        type="triton",
        support_dir=support_dir,
        top_k=2,
        rejection_threshold=0.75,
        preprocessing=PreprocessingConfig(size=(8, 8), normalize=False),
        triton=TritonConfig(),
    )
    backend = TritonBackend(config)
    backend.encoder._load_client_module = lambda: FakeGrpc
    backend.start()
    predictions = backend.predict([solid_image((220, 140, 25))])
    assert predictions[0][0].class_name == "amber"
    assert predictions[0][0].accepted is True
    backend.stop()


def test_python_backend_matches_pytorch(monkeypatch):
    class Tensor:
        def __init__(self, name, data):
            self.name = name
            self.data = data

        def as_numpy(self):
            return self.data

    class InferenceResponse:
        def __init__(self, output_tensors):
            self.output_tensors = output_tensors

    class Request:
        def __init__(self, images):
            self.images = Tensor("images", images)

    module = SimpleNamespace(
        Tensor=Tensor,
        InferenceResponse=InferenceResponse,
        TritonModelException=RuntimeError,
        get_input_tensor_by_name=lambda request, name: request.images,
    )
    monkeypatch.setitem(sys.modules, "triton_python_backend_utils", module)
    model_path = Path(__file__).parents[2] / "triton_models" / "image_encoder" / "1" / "model.py"
    spec = importlib.util.spec_from_file_location("triton_statistical_encoder", model_path)
    assert spec is not None and spec.loader is not None
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    images = np.random.default_rng(7).random((2, 3, 8, 8), dtype=np.float32)
    response = model_module.TritonPythonModel().execute([Request(images)])[0]
    actual = response.output_tensors[0].data
    expected = StatisticalEncoder()(torch.from_numpy(images)).numpy()
    assert np.allclose(actual, expected, atol=1e-6)
