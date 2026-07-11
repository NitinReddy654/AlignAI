"""Tests for distributed fine-tuning configuration."""

from alignai.training.distributed_finetune import (
    DistributedTrainingConfig,
    distributed_environment,
    distributed_training_kwargs,
)


def test_distributed_config_disabled_for_single_process():
    cfg = DistributedTrainingConfig(world_size=1)
    assert cfg.enabled is False


def test_distributed_training_kwargs_for_ddp():
    cfg = DistributedTrainingConfig(strategy="ddp", world_size=2, local_rank=1)
    kwargs = distributed_training_kwargs(cfg)
    assert kwargs["ddp_find_unused_parameters"] is False
    assert kwargs["local_rank"] == 1


def test_distributed_environment_shape():
    env = distributed_environment()
    assert {"world_size", "rank", "local_rank", "is_distributed"} <= set(env)
