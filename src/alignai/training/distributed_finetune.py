"""Distributed fine-tuning helpers for DDP, FSDP, and multi-GPU fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class DistributedTrainingConfig:
    """Runtime configuration for distributed fine-tuning."""

    strategy: str = "ddp"  # ddp | fsdp | data_parallel | none
    backend: str = "nccl"
    init_method: str = "env://"
    world_size: int = int(os.getenv("WORLD_SIZE", "1"))
    rank: int = int(os.getenv("RANK", "0"))
    local_rank: int = int(os.getenv("LOCAL_RANK", "0"))
    mixed_precision: bool = True
    fsdp_cpu_offload: bool = False

    @property
    def enabled(self) -> bool:
        return self.strategy != "none" and self.world_size > 1


def distributed_environment() -> Dict[str, int | bool]:
    """Return standard torch.distributed environment metadata."""
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_distributed": world_size > 1,
    }


def setup_distributed(config: DistributedTrainingConfig) -> bool:
    """Initialize torch.distributed when running with more than one process."""
    if not config.enabled:
        return False

    import torch
    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(config.local_rank)
        dist.init_process_group(
            backend=config.backend,
            init_method=config.init_method,
            world_size=config.world_size,
            rank=config.rank,
        )
    return True


def cleanup_distributed() -> None:
    """Tear down a distributed process group when initialized."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_for_distributed(model: Any, config: DistributedTrainingConfig) -> Any:
    """
    Wrap a model for DDP/FSDP or fall back to DataParallel for single-node multi-GPU.
    """
    if config.strategy == "none":
        return model

    import torch

    if config.strategy == "fsdp":
        from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel

        cpu_offload = CPUOffload(offload_params=config.fsdp_cpu_offload)
        return FullyShardedDataParallel(model, cpu_offload=cpu_offload)

    if config.strategy == "ddp" and config.enabled:
        from torch.nn.parallel import DistributedDataParallel

        device_ids = [config.local_rank] if torch.cuda.is_available() else None
        return DistributedDataParallel(model, device_ids=device_ids)

    if config.strategy == "data_parallel" and torch.cuda.device_count() > 1:
        return torch.nn.DataParallel(model)

    return model


def distributed_training_kwargs(config: DistributedTrainingConfig) -> Dict[str, Any]:
    """Return TrainingArguments/SFTConfig kwargs for distributed training."""
    kwargs: Dict[str, Any] = {}
    if config.strategy == "ddp":
        kwargs["ddp_find_unused_parameters"] = False
    if config.enabled:
        kwargs["local_rank"] = config.local_rank
    if config.strategy == "fsdp":
        kwargs["fsdp"] = "full_shard auto_wrap"
    return kwargs
