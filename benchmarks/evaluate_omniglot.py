from __future__ import annotations

import argparse
import json
import math
import platform
import random
from collections import defaultdict
from statistics import mean, stdev

import torch
import torch.nn.functional as functional

from vision_pipeline.models import StatisticalEncoder, TorchvisionEncoder


def load_dataset(root: str, download: bool) -> tuple[object, dict[int, list[int]]]:
    try:
        from torchvision.datasets import Omniglot
    except ImportError as error:
        raise RuntimeError("Omniglot evaluation requires installation with the 'vision' extra") from error

    dataset = Omniglot(root=root, background=False, download=download)
    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        _, target = dataset[index]
        indices_by_class[target].append(index)
    return dataset, indices_by_class


def create_episodes(
    indices_by_class: dict[int, list[int]],
    ways: int,
    shots: int,
    queries: int,
    episodes: int,
    seed: int,
) -> list[dict[str, list[int]]]:
    if ways < 1:
        raise ValueError("ways must be at least one")
    if ways > len(indices_by_class):
        raise ValueError("ways exceeds the number of available classes")
    if shots < 1 or queries < 1 or episodes < 1:
        raise ValueError("shots, queries, and episodes must be at least one")
    if any(len(indices) < shots + queries for indices in indices_by_class.values()):
        raise ValueError("a class does not contain enough images for the requested episode")

    generator = random.Random(seed)
    class_ids = sorted(indices_by_class)
    result = []
    for _ in range(episodes):
        support_indices = []
        query_indices = []
        support_labels = []
        query_labels = []
        for episode_label, class_id in enumerate(generator.sample(class_ids, ways)):
            selected = generator.sample(indices_by_class[class_id], shots + queries)
            support_indices.extend(selected[:shots])
            query_indices.extend(selected[shots:])
            support_labels.extend([episode_label] * shots)
            query_labels.extend([episode_label] * queries)
        result.append(
            {
                "support_indices": support_indices,
                "query_indices": query_indices,
                "support_labels": support_labels,
                "query_labels": query_labels,
            }
        )
    return result


def create_encoder(name: str, device: str) -> tuple[torch.nn.Module, object]:
    try:
        from torchvision import transforms
        from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights
    except ImportError as error:
        raise RuntimeError("Omniglot evaluation requires installation with the 'vision' extra") from error

    if name == "statistical":
        encoder = StatisticalEncoder()
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32), antialias=True),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
    elif name == "mobilenet_v3_small":
        encoder = TorchvisionEncoder(name=name, pretrained=True)
        transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()
    else:
        encoder = TorchvisionEncoder(name=name, pretrained=True)
        transform = ResNet18_Weights.DEFAULT.transforms()
    return encoder.to(device).eval(), transform


def embed_images(
    dataset: object,
    indices: list[int],
    encoder: torch.nn.Module,
    transform: object,
    batch_size: int,
    device: str,
) -> dict[int, torch.Tensor]:
    embeddings = {}
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        images = []
        for index in batch_indices:
            image, _ = dataset[index]  # type: ignore[index]
            images.append(transform(image.convert("RGB")))  # type: ignore[operator]
        with torch.inference_mode():
            values = encoder(torch.stack(images).to(device)).flatten(start_dim=1)
            values = functional.normalize(values, p=2, dim=1).cpu()
        embeddings.update(zip(batch_indices, values, strict=True))
    return embeddings


def evaluate_metric(
    episodes: list[dict[str, list[int]]],
    embeddings: dict[int, torch.Tensor],
    metric: str,
    ways: int,
) -> dict[str, float]:
    episode_accuracies = []
    for episode in episodes:
        support = torch.stack([embeddings[index] for index in episode["support_indices"]])
        query = torch.stack([embeddings[index] for index in episode["query_indices"]])
        support_labels = torch.tensor(episode["support_labels"])
        query_labels = torch.tensor(episode["query_labels"])
        prototypes = []
        for class_index in range(ways):
            prototype = support[support_labels == class_index].mean(dim=0, keepdim=True)
            prototypes.append(functional.normalize(prototype, p=2, dim=1))
        prototype_tensor = torch.cat(prototypes)
        if metric == "cosine":
            scores = query @ prototype_tensor.T
        else:
            scores = -torch.cdist(query, prototype_tensor, p=2)
        accuracy = (scores.argmax(dim=1) == query_labels).float().mean().item()
        episode_accuracies.append(accuracy * 100)

    average = mean(episode_accuracies)
    standard_deviation = stdev(episode_accuracies) if len(episode_accuracies) > 1 else 0.0
    confidence_interval = 1.96 * standard_deviation / math.sqrt(len(episode_accuracies))
    return {
        "mean_accuracy_percent": round(average, 2),
        "episode_std_percent": round(standard_deviation, 2),
        "ci95_percent": round(confidence_interval, 2),
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    import torchvision

    torch.set_num_threads(args.threads)
    dataset, indices_by_class = load_dataset(args.data_dir, args.download)
    episodes = create_episodes(
        indices_by_class,
        args.ways,
        args.shots,
        args.queries,
        args.episodes,
        args.seed,
    )
    used_indices = sorted(
        {
            index
            for episode in episodes
            for key in ("support_indices", "query_indices")
            for index in episode[key]
        }
    )

    results = []
    for encoder_name in args.encoders:
        encoder, transform = create_encoder(encoder_name, args.device)
        embeddings = embed_images(
            dataset,
            used_indices,
            encoder,
            transform,
            args.batch_size,
            args.device,
        )
        for metric in args.metrics:
            results.append(
                {
                    "encoder": encoder_name,
                    "weights": "ImageNet-1K" if encoder_name != "statistical" else "none",
                    "metric": metric,
                    **evaluate_metric(episodes, embeddings, metric, args.ways),
                }
            )

    return {
        "dataset": "Omniglot evaluation split",
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": args.device,
            "threads": args.threads,
        },
        "protocol": {
            "ways": args.ways,
            "shots": args.shots,
            "queries_per_class": args.queries,
            "episodes": args.episodes,
            "seed": args.seed,
            "rejection_threshold": None,
        },
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--queries", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=["statistical", "mobilenet_v3_small", "resnet18"],
        default=["statistical", "mobilenet_v3_small"],
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["cosine", "euclidean"],
        default=["cosine", "euclidean"],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_evaluation(parse_args()), indent=2))


if __name__ == "__main__":
    main()
