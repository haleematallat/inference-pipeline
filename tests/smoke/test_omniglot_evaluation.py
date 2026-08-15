import runpy

import torch
from torch import nn


class FlattenEncoder(nn.Module):
    def forward(self, images):
        return images.flatten(start_dim=1)


def test_episode_evaluation_uses_fixed_support_and_query_sets() -> None:
    evaluation = runpy.run_path("benchmarks/evaluate_omniglot.py", run_name="evaluation")
    create_episodes = evaluation["create_episodes"]
    evaluate_metric = evaluation["evaluate_metric"]
    indices_by_class = {
        class_index: list(range(class_index * 20, (class_index + 1) * 20))
        for class_index in range(5)
    }
    episodes = create_episodes(
        indices_by_class,
        ways=5,
        shots=5,
        queries=5,
        episodes=3,
        seed=7,
    )
    images = {
        index: torch.nn.functional.one_hot(torch.tensor(index // 20), num_classes=5)
        .float()
        .reshape(1, 1, 5)
        for index in range(100)
    }

    result, episode_accuracies = evaluate_metric(
        episodes,
        images.__getitem__,
        FlattenEncoder(),
        metric="cosine",
        device="cpu",
        batch_size=8,
    )

    assert result["mean_accuracy_percent"] == 100.0
    assert result["episode_std_percent"] == 0.0
    assert episode_accuracies == [100.0, 100.0, 100.0]


def test_paired_metric_comparison_uses_episode_differences() -> None:
    evaluation = runpy.run_path("benchmarks/evaluate_omniglot.py", run_name="evaluation")
    compare_metrics = evaluation["compare_metrics"]

    result = compare_metrics([60.0, 70.0, 80.0], [62.0, 72.0, 82.0])

    assert result["mean_difference_pp"] == 2.0
    assert result["ci95_pp"] == 0.0
    assert result["excludes_zero"] is True


def test_omniglot_index_collection_falls_back_to_dataset_reads() -> None:
    evaluation = runpy.run_path("benchmarks/evaluate_omniglot.py", run_name="evaluation")
    collect_class_indices = evaluation["collect_class_indices"]

    class Dataset:
        samples = [(object(), 0), (object(), 1), (object(), 0)]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    assert collect_class_indices(Dataset()) == {0: [0, 2], 1: [1]}


def test_omniglot_index_collection_uses_metadata_without_decoding() -> None:
    evaluation = runpy.run_path("benchmarks/evaluate_omniglot.py", run_name="evaluation")
    collect_class_indices = evaluation["collect_class_indices"]

    class Dataset:
        _characters = ["b", "a"]
        _flat_character_images = [
            ("two.png", 0),
            ("two.png", 1),
            ("one.png", 0),
            ("one.png", 1),
        ]

        def __getitem__(self, index):
            raise AssertionError("image should not be decoded")

    assert collect_class_indices(Dataset()) == {0: [3, 1], 1: [2, 0]}
