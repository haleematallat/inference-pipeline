import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from vision_pipeline.config.loader import load_config
from vision_pipeline.pipeline import PipelineRunner
from vision_pipeline.runners.factory import InferenceRunnerFactory
from vision_pipeline.streams.factory import StreamHandlerFactory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pipeline(config_path: str | Path) -> tuple[PipelineRunner, Path]:
    config = load_config(config_path)
    logging.basicConfig(
        level=getattr(logging, config.logging_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    set_seed(config.seed)
    stream_handler = StreamHandlerFactory.create(config.stream)
    inference_runners = [InferenceRunnerFactory.create(runner_config) for runner_config in config.runners]
    return PipelineRunner(stream_handler, inference_runners), config.stream.output_path


def run_demo(config_path: str | Path) -> int:
    pipeline, output_path = build_pipeline(config_path)
    processed = pipeline.run()
    print(f"Processed {processed} images")
    print(f"Output: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vision-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo_parser = subparsers.add_parser("demo", help="run the standalone few-shot demo")
    demo_parser.add_argument("--config", default="configs/demo.yaml")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "demo":
        return run_demo(args.config)
    return 1
