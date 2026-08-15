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

The implementation baseline embeds the support set and rebuilds class prototypes before every
query batch. This pipeline calculates them when the backend starts, then reuses them until the
support set changes. The resulting latency reduction is expected from removing repeated support
encoding; the benchmark below validates that the implementation follows that cost model rather
than presenting prototype caching as a new optimization.

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

The classifier can also be used directly with any encoder that returns one feature vector per
image. This complete example uses channel means:

~~~python
from torch import nn
from vision_pipeline.models import PrototypicalNetwork

class ChannelMeanEncoder(nn.Module):
    def forward(self, images):
        return images.mean(dim=(2, 3))

model = PrototypicalNetwork(ChannelMeanEncoder(), similarity_metric="cosine")
model.fit_support(support_tensor, ["cat", "cat", "dog", "dog"])
predictions = model.predict(query_tensor)
~~~

An alternative backend can extend an existing implementation and register without changing the
pipeline or runner:

~~~python
from vision_pipeline.clients.factory import InferenceBackendFactory
from vision_pipeline.clients.pytorch_backend import PyTorchBackend

@InferenceBackendFactory.register("top1_local")
class Top1LocalBackend(PyTorchBackend):
    def predict(self, images):
        return [ranked[:1] for ranked in super().predict(images)]
~~~

Set `backend.type` to `top1_local`; the factory supplies it to the existing few-shot runner.

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

`benchmarks/benchmark_inference.py` validates cached inference against the direct recomputation
baseline with deterministic tensors. If support and query images have the same per-image encoder
cost and encoder work dominates, the expected speedup is:

$$
\frac{T_q + T_s}{T_q} = 1 + \frac{N_s}{N_q}
$$

~~~bash
python benchmarks/benchmark_inference.py
~~~

Reference CPU run: AMD EPYC 9V74, PyTorch 2.9.1+cpu, one thread, five classes, 32 queries,
32 x 32 inputs, 30 warmup iterations, and 200 measured iterations.

| Shots per class | Support images | Predicted | Measured | Cached median | Recomputed median |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 1.156x | 1.390x | 0.751 ms | 1.044 ms |
| 5 | 25 | 1.781x | 1.874x | 0.757 ms | 1.418 ms |
| 10 | 50 | 2.562x | 2.541x | 0.745 ms | 1.893 ms |
| 20 | 100 | 4.125x | 3.839x | 0.754 ms | 2.894 ms |

The relation is an encoder-only expectation, not an exact end-to-end identity. Prototype
construction, scoring, Python dispatch, and measurement noise explain the remaining difference.
The sweep is evidence that support work moves out of the query path as intended, not a claim of
an algorithmic improvement.

The same 5-way 5-shot workload with an untrained MobileNetV3 Small feature extractor at 224 x
224 measured 123.404 ms cached and 214.334 ms with recomputation: 1.737x observed against 1.781x
predicted. Weights do not affect compute cost; the untrained model avoids a download in the
latency benchmark. These are model-core measurements and exclude image decoding, stream I/O,
Triton transport, and concurrency.

`torch.utils.flop_counter.FlopCounterMode` reports 0.110 GFLOPs per image for this repository's
MobileNetV3 Small feature extractor at 224 x 224. The statistical encoder reports zero because
the profiler does not count its reduction operators. An explicit estimate of its means, variance
terms, and quadrant reductions gives 18,435 scalar operations, or approximately 0.000018 GFLOPs,
per 32 x 32 image. FLOP conventions differ, so the script exposes both the profiler result and
the explicit reduction estimate.

Run the realistic encoder benchmark with:

~~~bash
python benchmarks/benchmark_inference.py \
  --encoder mobilenet_v3_small --image-size 224 --shots 5
~~~

## Omniglot evaluation

The repository now includes a closed-set accuracy evaluation on the Omniglot evaluation split.
It uses 100 fixed 5-way 5-shot episodes, 15 queries per class, and seed 7. The statistical encoder
is deterministic and untrained; MobileNetV3 Small uses frozen ImageNet-1K weights without
fine-tuning on Omniglot.

| Encoder | Metric | Mean accuracy | Episode standard deviation | 95% CI |
| --- | --- | ---: | ---: | ---: |
| Statistical, no weights | Cosine | 63.23% | 10.87% | +/- 2.13% |
| Statistical, no weights | Euclidean | 63.23% | 10.87% | +/- 2.13% |
| MobileNetV3 Small, ImageNet-1K | Cosine | 92.52% | 4.92% | +/- 0.96% |
| MobileNetV3 Small, ImageNet-1K | Euclidean | 92.52% | 4.92% | +/- 0.96% |

Cosine and Euclidean produce identical class rankings here because query embeddings and class
prototypes are L2-normalized. Their score scales still differ, so rejection thresholds are not
interchangeable. Rejection is disabled for this closed-set table and requires separate unknown
classes plus held-out calibration data.

~~~bash
pip install -e ".[vision]"
python benchmarks/evaluate_omniglot.py --download
~~~

This is a frozen-encoder system check, not a reproduction of the trained Omniglot result from the
Prototypical Networks paper. The script downloads the evaluation split and official torchvision
weights on first use, then emits the protocol and results as JSON.

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

## Related tools

| Tool | Primary focus | Difference from this repository |
| --- | --- | --- |
| EasyFSL | Few-shot methods, tasks, and research examples | Broader method coverage; this project focuses on configurable inference and Triton parity |
| learn2learn | Meta-learning algorithms and benchmark utilities | Includes training and research workflows; this project is a smaller inference system |
| Ray Serve | General distributed model serving | Provides scaling and deployment primitives without few-shot prototype semantics |
| TorchServe | PyTorch model serving | Model archive and serving workflow; currently under limited maintenance |

The closest method-level neighbour is EasyFSL. The closest serving alternatives are Triton,
which is implemented here, and general serving frameworks such as Ray Serve. The distinguishing
scope is the connection between prototype-based classification, validated configuration, stream
abstractions, and tested local/Triton embedding parity.

## Future improvements

- Evaluate a representative application dataset with per-class metrics and confidence intervals;
  Omniglot currently verifies only generic closed-set few-shot behavior.
- Compare the current prototypical classifier with Matching Networks and Relation Networks under
  the same few-shot episodes, plus a CLIP zero-shot baseline using the same class vocabulary.
- Add unknown classes to the episode protocol and report rejection AUROC, false-accept rate, and
  thresholds calibrated on a separate validation split.
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
8. Lake, Salakhutdinov, and Tenenbaum, [Human-level concept learning through probabilistic program induction](https://www.science.org/doi/10.1126/science.aab3050), Science 2015.
9. Sicara, [EasyFSL: Easy Few-Shot Learning](https://github.com/sicara/easy-few-shot-learning).
10. Arnold et al., [learn2learn: A Library for Meta-Learning Research](https://arxiv.org/abs/2008.12284), 2020.
11. Ray, [Ray Serve documentation](https://docs.ray.io/en/latest/serve/index.html).
12. PyTorch, [TorchServe documentation](https://docs.pytorch.org/serve/).
