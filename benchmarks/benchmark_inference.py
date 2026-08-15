from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from statistics import median
from time import perf_counter_ns

import torch
from torch import nn

from vision_pipeline.models import PrototypicalNetwork, StatisticalEncoder, TorchvisionEncoder


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


def create_encoder(name: str, pretrained: bool) -> nn.Module:
    if name == "statistical":
        return StatisticalEncoder()
    return TorchvisionEncoder(name=name, pretrained=pretrained)


def statistical_encoder_flops(image_size: int) -> int:
    pixels = image_size * image_size
    return 18 * pixels + 3


def profile_encoder_flops(encoder: nn.Module, image_size: int, device: str) -> int | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None

    encoder = encoder.to(device).eval()
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        encoder(torch.rand((1, 3, image_size, image_size), device=device))
    return int(counter.get_total_flops())


def benchmark_shots(args: argparse.Namespace, shots: int) -> dict[str, object]:
    support, labels, query = create_inputs(
        args.classes,
        shots,
        args.batch_size,
        args.image_size,
    )
    support = support.to(args.device)
    query = query.to(args.device)

    cached_model = PrototypicalNetwork(
        create_encoder(args.encoder, args.pretrained),
        device=args.device,
        top_k=1,
    )
    recomputed_model = PrototypicalNetwork(
        create_encoder(args.encoder, args.pretrained),
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

    def predict_with_recomputation() -> object:
        recomputed_model.fit_support(support, labels)
        return recomputed_model.predict(query)

    recomputed = measure(
        predict_with_recomputation,
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
    predicted_speedup = 1 + (args.classes * shots) / args.batch_size

    return {
        "shots_per_class": shots,
        "support_images": args.classes * shots,
        "predicted_speedup": round(predicted_speedup, 3),
        "measured_speedup": round(recomputed["median_ms"] / cached["median_ms"], 3),
        "cached_prototypes": cached,
        "recomputed_prototypes": recomputed,
        "prototype_setup": prototype_setup,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.classes < 1 or args.batch_size < 1 or args.image_size < 2:
        raise ValueError("classes and batch size must be positive; image size must be at least two")
    if any(shots < 1 for shots in args.shots):
        raise ValueError("shots must be at least one")
    if args.warmup < 0 or args.iterations < 1 or args.threads < 1:
        raise ValueError("warmup cannot be negative; iterations and threads must be positive")

    torch.set_num_threads(args.threads)
    encoder = create_encoder(args.encoder, args.pretrained)
    encoder_flops = statistical_encoder_flops(args.image_size) if args.encoder == "statistical" else None
    profiled_flops = profile_encoder_flops(encoder, args.image_size, args.device)
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
            "name": args.encoder,
            "pretrained": args.pretrained,
            "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
            "profiled_flops_per_image": profiled_flops,
            "profiled_gflops_per_image": (
                round(profiled_flops / 1_000_000_000, 6) if profiled_flops is not None else None
            ),
            "estimated_flops_per_image": encoder_flops,
            "estimated_gflops_per_image": (
                round(encoder_flops / 1_000_000_000, 9) if encoder_flops is not None else None
            ),
        },
        "speedup_model": "1 + support_images / query_images",
        "measurements": [benchmark_shots(args, shots) for shots in args.shots],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument(
        "--encoder",
        choices=["statistical", "mobilenet_v3_small", "resnet18"],
        default="statistical",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_benchmark(parse_args()), indent=2))


if __name__ == "__main__":
    main()
