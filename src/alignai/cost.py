"""Training cost estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass

from alignai.config import get_config


@dataclass
class CostEstimate:
    """Estimated training cost breakdown."""

    duration_hours: float
    gpu_hour_cost: float
    total_cost_usd: float
    strategy: str
    model_size: str

    def to_dict(self) -> dict:
        return {
            "duration_hours": round(self.duration_hours, 4),
            "gpu_hour_cost": self.gpu_hour_cost,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "strategy": self.strategy,
            "model_size": self.model_size,
        }


def estimate_training_cost(
    duration_seconds: float,
    strategy: str,
    model_size: str = "3B",
) -> CostEstimate:
    """Estimate training cost based on duration and GPU-hour rate."""
    cfg = get_config()
    hours = duration_seconds / 3600.0
    multiplier = {"full": 1.5, "lora": 1.0, "qlora": 0.8}.get(strategy.lower(), 1.0)
    adjusted_hours = hours * multiplier
    total = adjusted_hours * cfg.gpu_hour_cost
    return CostEstimate(
        duration_hours=adjusted_hours,
        gpu_hour_cost=cfg.gpu_hour_cost,
        total_cost_usd=total,
        strategy=strategy,
        model_size=model_size,
    )


def estimate_inference_cost(
    latency_ms: float,
    tokens_generated: int,
    gpu_hour_cost: float | None = None,
) -> dict:
    """Estimate per-request inference cost."""
    cfg = get_config()
    rate = gpu_hour_cost or cfg.gpu_hour_cost
    hours = latency_ms / 3_600_000.0
    return {
        "latency_ms": latency_ms,
        "tokens_generated": tokens_generated,
        "estimated_cost_usd": round(hours * rate, 8),
        "tokens_per_second": round(tokens_generated / (latency_ms / 1000.0), 2) if latency_ms > 0 else 0,
    }
