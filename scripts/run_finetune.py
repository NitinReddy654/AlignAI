#!/usr/bin/env python3
"""CLI entrypoint for fine-tuning jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alignai.logging_utils import setup_logging
from alignai.training.strategies import TrainingHyperparams
from alignai.training.trainer import run_finetuning_job

logger = setup_logging("alignai.finetune")


def main() -> None:
    parser = argparse.ArgumentParser(description="AlignAI Fine-Tuning Pipeline")
    parser.add_argument("--dataset", required=True, help="Path to JSONL training dataset")
    parser.add_argument("--strategy", choices=["full", "lora", "qlora"], default="lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-version", default="v1")
    args = parser.parse_args()

    hparams = TrainingHyperparams(
        strategy=args.strategy,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    logger.info("Launching %s fine-tuning on %s", args.strategy, args.dataset)
    result = run_finetuning_job(
        dataset_path=args.dataset,
        strategy=args.strategy,
        hparams=hparams,
        output_dir=args.output_dir,
        dataset_version=args.dataset_version,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
