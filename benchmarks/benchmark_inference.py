from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from statistics import median
from time import perf_counter_ns

import torch

from vision_pipeline.models import PrototypicalNetwork, StatisticalEncoder


def create_inputs(
    class_count: int,
    shots: int,
    batch_size: int,
    image_size: int,
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    support = []
    labels = []
    class_colors = torch.rand((class_count, 3, 1, 1), generator=generator)
    for class_index, color in enumerate(class_colors):
        for _ in range(shots):
            noise = torch.rand((3, image_size, image_size), generator=generator) * 0.05
            support.append((color + noise).clamp(0.0, 1.0))
            labels.append(f"class-{class_index}")

    query_classes = torch.arange(batch_size) % class_count
    query = []
    for class_index in query_classes:
        noise = torch.rand((3, image_size, image_size), generator=generator) * 0.05
        query.append((class_colors[class_index] + noise).clamp(0.0, 1.0))
    return torch.stack(support), labels, torch.stack(query)


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def measure(
    operation: Callable[[], object],
    warmup: int,
    iterations: int,
    device: str,
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    synchronize(device)

    samples = []
    for _ in range(iterations):
        started_at = perf_counter_ns()
        operation()
        synchronize(device)
        samples.append((perf_counter_ns() - started_at) / 1_000_000)

    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "median_ms": round(median(samples), 4),
        "p95_ms": round(ordered[p95_index], 4),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.threads)
    support, labels, query = create_inputs(
        args.classes,
        args.shots,
        args.batch_size,
        args.image_size,
    )
    support = support.to(args.device)
    query = query.to(args.device)

    cached_model = PrototypicalNetwork(
        StatisticalEncoder(),
        device=args.device,
        top_k=1,
    )
    uncached_model = PrototypicalNetwork(
        StatisticalEncoder(),
        device=args.device,
        top_k=1,
    )
    cached_model.fit_support(support, labels)

    cached = measure(
        lambda: cached_model.predict(query),
        args.warmup,
        args.iterations,
        args.device,
    )

    def predict_without_cache() -> object:
        uncached_model.fit_support(support, labels)
        return uncached_model.predict(query)

    uncached = measure(
        predict_without_cache,
        args.warmup,
        args.iterations,
        args.device,
    )
    prototype_setup = measure(
        lambda: cached_model.fit_support(support, labels),
        args.warmup,
        args.iterations,
        args.device,
    )

    cached["images_per_second"] = round(args.batch_size / (cached["median_ms"] / 1000), 1)
    uncached["images_per_second"] = round(args.batch_size / (uncached["median_ms"] / 1000), 1)

    return {
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": args.device,
            "threads": args.threads,
        },
        "workload": {
            "classes": args.classes,
            "shots_per_class": args.shots,
            "query_batch_size": args.batch_size,
            "image_size": [3, args.image_size, args.image_size],
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "encoder": {
            "name": "statistical",
            "parameters": sum(parameter.numel() for parameter in cached_model.encoder.parameters()),
        },
        "cached_prototypes": cached,
        "recomputed_prototypes": uncached,
        "prototype_setup": prototype_setup,
        "median_speedup": round(uncached["median_ms"] / cached["median_ms"], 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_benchmark(parse_args()), indent=2))


if __name__ == "__main__":
    main()
