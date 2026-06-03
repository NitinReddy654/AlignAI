"""Fine-tuning orchestration using TRL SFTTrainer."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from alignai.config import get_config
from alignai.cost import estimate_training_cost
from alignai.experiments.registry import ExperimentRecord, ExperimentRegistry
from alignai.logging_utils import setup_logging
from alignai.training.data import build_sft_dataset, load_jsonl_dataset
from alignai.training.strategies import (
    TrainingHyperparams,
    get_base_model_id,
    prepare_model_for_training,
)

logger = setup_logging(__name__)


@dataclass
class TrainingResult:
    """Outcome of a fine-tuning job."""

    experiment_id: str
    output_dir: str
    duration_seconds: float
    cost_estimate: dict
    hyperparams: dict
    status: str

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "output_dir": self.output_dir,
            "duration_seconds": round(self.duration_seconds, 2),
            "cost_estimate": self.cost_estimate,
            "hyperparams": self.hyperparams,
            "status": self.status,
        }


def run_finetuning_job(
    dataset_path: str,
    strategy: str = "lora",
    hparams: Optional[TrainingHyperparams] = None,
    output_dir: Optional[str] = None,
    dataset_version: str = "v1",
) -> TrainingResult:
    """Execute a fine-tuning job and register experiment metadata."""
    cfg = get_config()
    hparams = hparams or TrainingHyperparams(strategy=strategy)
    hparams.strategy = strategy
    experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
    model_id = get_base_model_id(strategy)
    output_dir = output_dir or str(cfg.checkpoints_dir / experiment_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting %s fine-tuning: experiment=%s model=%s", strategy, experiment_id, model_id)

    records = load_jsonl_dataset(dataset_path)
    model, tokenizer, _ = prepare_model_for_training(model_id, strategy, hparams)
    train_dataset = build_sft_dataset(records, tokenizer, hparams.max_seq_length)

    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=hparams.num_epochs,
        per_device_train_batch_size=hparams.batch_size,
        gradient_accumulation_steps=hparams.gradient_accumulation_steps,
        learning_rate=hparams.learning_rate,
        warmup_ratio=hparams.warmup_ratio,
        weight_decay=hparams.weight_decay,
        logging_steps=hparams.logging_steps,
        save_steps=hparams.save_steps,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        seed=cfg.seed,
        report_to="none",
        max_seq_length=hparams.max_seq_length,
        dataset_text_field="text",
    )

    start = time.perf_counter()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    duration = time.perf_counter() - start

    cost = estimate_training_cost(duration, strategy, model_id.split("/")[-1])
    registry = ExperimentRegistry()
    record = ExperimentRecord(
        experiment_id=experiment_id,
        model_version=model_id,
        dataset_version=dataset_version,
        strategy=strategy,
        hyperparams=hparams.to_dict(),
        output_dir=output_dir,
        duration_seconds=duration,
        cost_estimate=cost.to_dict(),
        status="completed",
    )
    registry.save(record)

    logger.info("Fine-tuning complete: %s (%.1fs)", experiment_id, duration)
    return TrainingResult(
        experiment_id=experiment_id,
        output_dir=output_dir,
        duration_seconds=duration,
        cost_estimate=cost.to_dict(),
        hyperparams=hparams.to_dict(),
        status="completed",
    )
