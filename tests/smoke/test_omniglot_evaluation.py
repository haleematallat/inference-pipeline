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

    result = evaluate_metric(
        episodes,
        images.__getitem__,
        FlattenEncoder(),
        metric="cosine",
        device="cpu",
        batch_size=8,
    )

    assert result["mean_accuracy_percent"] == 100.0
    assert result["episode_std_percent"] == 0.0
