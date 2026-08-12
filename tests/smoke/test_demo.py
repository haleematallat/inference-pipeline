import json
from pathlib import Path

from vision_pipeline.cli import build_pipeline


def test_end_to_end_demo():
    repository_root = Path(__file__).parents[2]
    pipeline, output_path = build_pipeline(repository_root / "configs" / "demo.yaml")
    assert pipeline.run() == 2
    output = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert output[0]["predictions"][0]["class_name"] == "amber"
    assert output[1]["predictions"][0]["class_name"] == "blue"
    assert all(item["predictions"][0]["accepted"] for item in output)
