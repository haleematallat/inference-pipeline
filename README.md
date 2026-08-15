# Vision Inference Pipeline

This repository is a standalone computer-vision inference framework built around the same
runner, client, stream-handler, and pipeline boundaries used in larger streaming systems. The
default path performs real few-shot image classification with PyTorch: it embeds support
images, builds one prototype per class, embeds query images, ranks prototypes by similarity,
applies an unknown-class threshold, and writes deterministic JSONL output.

This is a few-shot pipeline, not a zero-shot model. Its model path follows the established
prototypical-network method; the work here is the standalone implementation and the surrounding
inference architecture, not a claim of a new learning algorithm.

## Capabilities

- Local PyTorch inference on CPU or CUDA.
- Prototypical few-shot classification with cached support prototypes.
- Cosine and Euclidean scoring, top-k output, and configurable rejection.
- Registry-based inference runners, backends, and stream handlers.
- Validated YAML configuration.
- Local-file and bounded in-memory streams.
- Optional Redis integration with a JSON serialization boundary.
- Triton gRPC inference with a runnable CPU model repository.
- Optional pretrained ResNet18 or MobileNetV3 encoders through torchvision.
- Reproducible offline demo and deterministic tests.

## Baseline and contribution

The direct baseline embeds the support set and rebuilds class prototypes before every query
batch. This pipeline calculates them once when the backend starts, then reuses them until the
support set changes. The result is less repeated work without changing the classifier's output.

The repository was implemented as a standalone codebase rather than forked from a research
implementation. It applies an existing few-shot method and adds the parts needed to run it as a
maintainable inference service: validated configuration, replaceable runners and backends,
stream boundaries, rejection handling, local and Triton execution, Docker, and CI. It does not
claim an accuracy improvement over the original prototypical-network paper.

## Architecture

~~~mermaid
flowchart TD
    A["Local files or stream"] --> B["StreamHandler"]
    B --> C["PipelineRunner"]
    C --> D["InferenceRunnerFactory"]
    D --> E["FewShotClassificationRunner"]
    E --> F["InferenceBackendFactory"]
    F --> G["PyTorch or Triton backend"]
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
triton_models/     runnable Triton Python model repository
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

The most useful adaptation points are:

- `configs/demo.yaml` for a new support set, encoder, threshold, or batch size.
- `src/vision_pipeline/runners/few_shot_classification.py` for task-level behavior.
- `src/vision_pipeline/clients/base.py` and `clients/factory.py` for another inference backend.
- `src/vision_pipeline/streams/base.py` and `streams/factory.py` for another source or sink.
- `src/vision_pipeline/models/encoders.py` for another PyTorch feature extractor.

## Redis

RedisStreamHandler is available in vision_pipeline.streams.redis_stream. It accepts a Redis URL
at runtime, keeps output queues bounded, and serializes JSON-compatible InferResult data.
configs/redis.example.yaml obtains REDIS_URL from the environment.

## Triton

The Triton path runs the same few-shot pipeline with embedding inference served over gRPC. The
tracked Python model implements the statistical encoder, while prototype construction, caching,
ranking, and rejection remain in the pipeline process.

~~~bash
docker compose --profile triton up --build --abort-on-container-exit triton-demo
~~~

The command starts the public NVIDIA Triton image, waits for the image_encoder model to become
ready, runs configs/triton.yaml, and writes outputs/triton_predictions.jsonl. It does not require
a GPU or external model weights. Ports 8000, 8001, and 8002 expose Triton's HTTP, gRPC, and metrics
endpoints.

To use an existing server, install the optional client and change triton.server_url:

~~~bash
pip install -e ".[triton]"
vision-pipeline demo --config configs/triton.yaml
~~~

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

The Triton tests cover gRPC request construction, startup and shutdown, invalid server output,
few-shot predictions, and numerical parity between the PyTorch encoder and the served Python
model. They run without starting Triton:

~~~bash
pytest -q tests/unit/test_triton_backend.py
docker compose --profile triton config
~~~

GitHub Actions runs the full test suite and validates both the default and Triton Compose
configurations on every push and pull request.

## Containers

~~~bash
docker compose config
docker compose run --rm vision-pipeline
docker compose --profile triton config
~~~

The compose file builds only the standalone pipeline and writes results to outputs.

## Design decisions and tradeoffs

- Explicit runner, backend, stream, and pipeline boundaries isolate model code from transport
  and orchestration.
- The default stream is synchronous, making ordering, failure propagation, and cleanup clear.
- Prototype calculation is separate from query inference to avoid recomputing the support set.
- Triton serves embeddings rather than final classes, so local and remote backends share the
  same few-shot ranking and rejection behavior.
- Checkpoints are loaded as state_dict data with weights_only enabled; pickled model objects
  are not accepted.
- The local-file pipeline processes one result at a time, while the backend itself supports
  batched prediction.

## Performance

`benchmarks/benchmark_inference.py` compares cached inference with the direct recomputation
baseline. It uses deterministic synthetic tensors so it measures the code path without dataset,
file-system, or network variance.

~~~bash
python benchmarks/benchmark_inference.py
~~~

Reference CPU run: AMD EPYC 9V74, PyTorch 2.9.1+cpu, one thread, 5-way 5-shot support,
32 query images of size 32 x 32, 50 warmup iterations, and 300 measured iterations.

| Path | Median batch latency | p95 batch latency | Throughput |
| --- | ---: | ---: | ---: |
| Cached prototypes | 0.742 ms | 0.856 ms | 43,127 images/s |
| Recompute prototypes per batch | 1.410 ms | 1.572 ms | 22,697 images/s |

Prototype setup took a median 0.615 ms. Moving it out of the query path produced a 1.90x median
speedup for this small deterministic encoder. These are core model-path measurements, not
end-to-end service numbers; preprocessing, disk I/O, Triton transport, and real encoder cost are
excluded. Run the script on deployment hardware before selecting a batch size or latency target.

The statistical demo encoder has no learned parameters. A GFLOP total is not reported for it
because its work is tensor reductions rather than the convolution and matrix operations counted
by common model profilers. For the optional 224 x 224 torchvision encoders, the published model
costs are 0.06 GFLOPs for MobileNetV3 Small and 1.81 GFLOPs for ResNet18. Those are encoder
reference values, not measurements of this complete pipeline.

CUDA can be selected for local PyTorch inference. Triton can combine requests with dynamic
batching, but adds serialization and network latency; local and remote paths should be compared
under the same concurrency and batch settings.

## Related approaches

| Approach | Class information at inference | Comparison | Status here |
| --- | --- | --- | --- |
| Prototypical Networks | Labelled support images | Distance to each class mean | Implemented |
| Matching Networks | Labelled support images | Attention over support examples | Future comparison |
| Relation Networks | Labelled support images | Learned relation function | Future comparison |
| CLIP | Class text prompts | Image-text similarity | Future zero-shot baseline |

Docker and Triton are deliberate deployment boundaries around the few-shot method. They are not
substitutes for model comparison, so evaluation against the alternatives above remains separate
work.

## Future improvements

- Replace the color demonstration set with a representative image dataset and report measured
  classification, rejection, and per-class metrics.
- Compare the current prototypical classifier with Matching Networks and Relation Networks under
  the same few-shot episodes, plus a CLIP zero-shot baseline using the same class vocabulary.
- Add threshold calibration against a held-out validation set instead of selecting similarity
  thresholds manually.
- Add a training and fine-tuning workflow that exports compatible encoder checkpoints with
  reproducible dataset and experiment configuration.
- Extend the Triton model repository with a GPU-backed encoder and benchmark local PyTorch and
  remote inference under the same batch and concurrency settings.
- Add a container-level integration test that starts Triton, runs the complete demo, and compares
  its predictions with the local backend.
- Add asynchronous stream processing and explicit backpressure policies for sustained workloads.

## License

Licensed under the MIT License. See `LICENSE`.

## References

1. Snell, Swersky, and Zemel, [Prototypical Networks for Few-shot Learning](https://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning), NeurIPS 2017.
2. Vinyals et al., [Matching Networks for One Shot Learning](https://proceedings.neurips.cc/paper/2016/hash/90e1357833654983612fb05e3ec9148c-Abstract.html), NeurIPS 2016.
3. Sung et al., [Learning to Compare: Relation Network for Few-Shot Learning](https://openaccess.thecvf.com/content_cvpr_2018/html/Sung_Learning_to_Compare_CVPR_2018_paper.html), CVPR 2018.
4. Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html), ICML 2021.
5. PyTorch, [MobileNetV3 Small model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).
6. PyTorch, [ResNet18 model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
7. NVIDIA, [Triton dynamic batcher documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html).
