from __future__ import annotations

import json
import subprocess
import sys


def test_benchmark_runs() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/benchmark_inference.py",
            "--classes",
            "2",
            "--shots",
            "1",
            "--batch-size",
            "2",
            "--image-size",
            "8",
            "--warmup",
            "1",
            "--iterations",
            "2",
        ],
        capture_output=True,
        check=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    measurement = result["measurements"][0]
    assert measurement["cached_prototypes"]["median_ms"] > 0
    assert measurement["recomputed_prototypes"]["median_ms"] > 0
    assert measurement["predicted_speedup"] == 2.0
    assert result["encoder"]["parameters"] == 0
    assert result["encoder"]["estimated_flops_per_image"] == 1155
