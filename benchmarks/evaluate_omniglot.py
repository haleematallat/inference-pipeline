from __future__ import annotations

import argparse
import json
import math
import platform
import random
from collections.abc import Callable
from functools import partial
from statistics import mean, stdev
from typing import TYPE_CHECKING

import torch
from PIL import Image

from vision_pipeline.models import PrototypicalNetwork, StatisticalEncoder, TorchvisionEncoder

if TYPE_CHECKING:
    from torchvision.datasets import Omniglot


def load_dataset(root: str, download: bool) -> tuple[Omniglot, dict[int, list[int]]]:
    try:
        from torchvision.datasets import Omniglot
    except ImportError as error:
        raise RuntimeError("Omniglot evaluation requires installation with the 'vision' extra") from error

    dataset = Omniglot(root=root, background=False, download=download)
    indices_by_class = {class_index: [] for class_index in range(len(dataset._characters))}
    for index, (_, target) in enumerate(dataset._flat_character_images):
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


def create_encoder(name: str) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
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
    return encoder, transform


def load_image(
    dataset: Omniglot,
    index: int,
    transform: Callable[[Image.Image], torch.Tensor],
) -> torch.Tensor:
    return transform(dataset[index][0].convert("RGB"))


def evaluate_metric(
    episodes: list[dict[str, list[int]]],
    image_loader: Callable[[int], torch.Tensor],
    encoder: torch.nn.Module,
    metric: str,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    model = PrototypicalNetwork(
        encoder,
        device=device,
        similarity_metric=metric,
        rejection_threshold=float("-inf"),
    )
    episode_accuracies = []
    for episode in episodes:
        support = torch.stack([image_loader(index) for index in episode["support_indices"]])
        support_labels = [str(label) for label in episode["support_labels"]]
        model.fit_support(support, support_labels)

        predicted_labels = []
        query_indices = episode["query_indices"]
        for start in range(0, len(query_indices), batch_size):
            batch_indices = query_indices[start : start + batch_size]
            query = torch.stack([image_loader(index) for index in batch_indices])
            predictions = model.predict(query)
            predicted_labels.extend(int(ranked[0].class_name) for ranked in predictions)
        correct = sum(
            predicted == expected
            for predicted, expected in zip(predicted_labels, episode["query_labels"], strict=True)
        )
        episode_accuracies.append(correct / len(predicted_labels) * 100)

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
    results = []
    for encoder_name in args.encoders:
        encoder, transform = create_encoder(encoder_name)
        image_loader = partial(load_image, dataset, transform=transform)
        for metric in args.metrics:
            results.append(
                {
                    "encoder": encoder_name,
                    "weights": "ImageNet-1K" if encoder_name != "statistical" else "none",
                    "metric": metric,
                    **evaluate_metric(
                        episodes,
                        image_loader,
                        encoder,
                        metric,
                        args.device,
                        args.batch_size,
                    ),
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
