"""Central configuration with validation for AlignAI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass
class AlignAIConfig:
    """Validated runtime configuration."""

    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_judge_model: str = field(default_factory=lambda: _env("OPENAI_JUDGE_MODEL", "gpt-4o-mini"))
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN"))
    base_model_lora: str = field(
        default_factory=lambda: _env("ALIGNAI_BASE_MODEL_LORA", "Qwen/Qwen2.5-3B-Instruct")
    )
    base_model_fft: str = field(
        default_factory=lambda: _env("ALIGNAI_BASE_MODEL_FFT", "Qwen/Qwen2.5-0.5B-Instruct")
    )
    data_dir: Path = field(default_factory=lambda: Path(_env("ALIGNAI_DATA_DIR", "./data")))
    artifacts_dir: Path = field(
        default_factory=lambda: Path(_env("ALIGNAI_ARTIFACTS_DIR", "./data/artifacts"))
    )
    checkpoints_dir: Path = field(
        default_factory=lambda: Path(_env("ALIGNAI_CHECKPOINTS_DIR", "./checkpoints"))
    )
    experiments_dir: Path = field(
        default_factory=lambda: Path(_env("ALIGNAI_EXPERIMENTS_DIR", "./data/artifacts/experiments"))
    )
    seed: int = field(default_factory=lambda: _env_int("ALIGNAI_SEED", 42))
    max_seq_length: int = field(default_factory=lambda: _env_int("ALIGNAI_MAX_SEQ_LENGTH", 2048))
    gpu_hour_cost: float = field(default_factory=lambda: _env_float("ALIGNAI_GPU_HOUR_COST", 0.50))
    log_level: str = field(default_factory=lambda: _env("ALIGNAI_LOG_LEVEL", "INFO"))

    def validate(self) -> list[str]:
        """Return list of validation errors (empty if valid)."""
        errors: list[str] = []
        if not self.openai_api_key and os.getenv("ALIGNAI_SKIP_OPENAI_VALIDATION") != "1":
            errors.append("OPENAI_API_KEY is required for evaluation features.")
        return errors

    def ensure_dirs(self) -> None:
        """Create required directories."""
        for path in (
            self.data_dir,
            self.artifacts_dir,
            self.checkpoints_dir,
            self.experiments_dir,
            self.artifacts_dir / "evaluations",
            self.artifacts_dir / "reports",
            self.artifacts_dir / "preferences",
            self.artifacts_dir / "datasets",
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_config() -> AlignAIConfig:
    """Return validated configuration singleton."""
    cfg = AlignAIConfig()
    cfg.ensure_dirs()
    return cfg
