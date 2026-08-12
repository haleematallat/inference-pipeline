# Vision Inference Pipeline

This repository is a standalone computer-vision inference framework built around the same
runner, client, stream-handler, and pipeline boundaries used in larger streaming systems. The
default path performs real few-shot image classification with PyTorch: it embeds support
images, builds one prototype per class, embeds query images, ranks prototypes by similarity,
applies an unknown-class threshold, and writes deterministic JSONL output.

## Capabilities

- Local PyTorch inference on CPU or CUDA.
- Prototypical few-shot classification with cached support prototypes.
- Cosine and Euclidean scoring, top-k output, and configurable rejection.
- Registry-based inference runners, backends, and stream handlers.
- Validated YAML configuration.
- Local-file and bounded in-memory streams.
- Optional Redis integration with a JSON serialization boundary.
- Optional pretrained ResNet18 or MobileNetV3 encoders through torchvision.
- Reproducible offline demo and deterministic tests.

## Architecture

~~~mermaid
flowchart TD
    A["Local files or stream"] --> B["StreamHandler"]
    B --> C["PipelineRunner"]
    C --> D["InferenceRunnerFactory"]
    D --> E["FewShotClassificationRunner"]
    E --> F["InferenceBackendFactory"]
    F --> G["PyTorchBackend"]
    G --> H["Encoder and prototypes"]
    H --> I["Prediction JSONL"]
~~~

PipelineRunner preserves runner order. Each runner receives and returns an InferResult, so
additional stages can enrich the same result without coupling the pipeline to one task.

## Repository structure

~~~text
src/vision_pipeline/
├── clients/       inference backends and backend factory
├── common/        shared result objects
├── config/        validated configuration and loader
├── models/        encoders, preprocessing, support artifacts, prototype model
├── runners/       runner interface, registry, few-shot runner
├── streams/       stream interface and implementations
├── cli.py
├── exceptions.py
└── pipeline.py
configs/           runnable and optional-integration examples
examples/          small offline support and query sets
tests/             unit, integration, and smoke tests
~~~

## Installation

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
~~~

Python 3.10 or newer is required. The default demo does not download weights or require
network access. Install the vision, redis, or triton extras only for those integrations.

## Standalone quick start

~~~bash
vision-pipeline demo --config configs/demo.yaml
~~~

The command reads the query images in examples/query and writes
outputs/predictions.jsonl. The expected top-ranked classes are amber and blue.

## Support-set format

Each direct child directory is a class. Every supported image inside it becomes a support
example.

~~~text
support/
├── class-a/
│   ├── example-1.jpg
│   └── example-2.jpg
└── class-b/
    └── example-1.jpg
~~~

Prototypes are calculated once when the backend starts and remain cached until shutdown or
explicit invalidation.

## Configuration

The complete demo configuration is in configs/demo.yaml. The central settings are:

~~~yaml
backend:
  type: local_pytorch
  device: cpu
  support_dir: ../examples/support
  similarity_metric: cosine
  top_k: 2
  rejection_threshold: 0.75
  encoder:
    type: statistical
  preprocessing:
    size: [32, 32]
    normalize: false
~~~

The statistical encoder is the offline demonstration encoder. It computes deterministic color
and spatial statistics with PyTorch. For general imagery, select a torchvision encoder:

~~~yaml
encoder:
  type: torchvision
  name: mobilenet_v3_small
  pretrained: true
~~~

Pretrained torchvision weights are downloaded on first use. Use a local checkpoint for
reproducible or offline deployment.

## Output

~~~json
{
  "class_name": "amber",
  "similarity": 0.999,
  "accepted": true,
  "rank": 1
}
~~~

Similarity is a cosine score or negative Euclidean distance, not a calibrated probability.
The rejection threshold must be selected with representative validation data.

## Extending the pipeline

Runners register with a stable key:

~~~python
@InferenceRunnerFactory.register("my_runner")
class MyRunner(IInferenceRunner): ...
~~~

The factory rejects duplicate keys and reports supported keys for an unknown runner. New
runners do not require a conditional branch. A backend follows the same pattern: implement
InferenceBackend, register it with InferenceBackendFactory, and accept validated configuration.
Keep optional imports inside the implementation module.

## Redis and Triton

RedisStreamHandler is available in vision_pipeline.streams.redis_stream. It accepts a Redis URL
at runtime, keeps output queues bounded, and serializes JSON-compatible InferResult data.
configs/redis.example.yaml obtains REDIS_URL from the environment.

The Triton dependency group and safe configuration template are included as an extension point.
A Triton backend is not registered because this repository does not include a portable model
repository.

## Testing and quality checks

~~~bash
python -m compileall src
python -c "import vision_pipeline"
ruff check .
mypy
pytest -q
~~~

Tests use a deterministic encoder and require no GPU, Redis, Triton, camera, internet access,
or model download.

## Containers

~~~bash
docker compose config
docker compose run --rm vision-pipeline
~~~

The compose file builds only the standalone pipeline and writes results to outputs.

## Design decisions and tradeoffs

- Explicit runner, backend, stream, and pipeline boundaries isolate model code from transport
  and orchestration.
- The default stream is synchronous, making ordering, failure propagation, and cleanup clear.
- Prototype calculation is separate from query inference to avoid recomputing the support set.
- Checkpoints are loaded as state_dict data with weights_only enabled; pickled model objects
  are not accepted.
- The local-file pipeline processes one result at a time, while the backend itself supports
  batched prediction.

## Performance

Support images are embedded only at startup. Query inference is batched according to batch_size.
CUDA can be selected in configuration. Benchmark preprocessing, encoder latency, batch size,
and stream backpressure on the intended hardware.

## Known limitations

- The sample classes are simple colors and are not an accuracy benchmark.
- Similarity thresholds are not calibrated automatically.
- The optional Triton configuration is an extension point, not a working default backend.
- No training workflow is included; this sample concentrates on correct, reusable inference.


