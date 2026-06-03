"""Alignment scoring weights and normalization."""

from __future__ import annotations

from typing import Dict

DEFAULT_WEIGHTS: Dict[str, float] = {
    "judge_quality": 0.30,
    "human_preference": 0.20,
    "safety": 0.15,
    "consistency": 0.10,
    "dataset_health": 0.10,
    "instruction_following": 0.10,
    "conciseness": 0.05,
}


def normalize_score(value: float, min_val: float = 1.0, max_val: float = 5.0) -> float:
    """Normalize a 1-5 Likert score to 0-100 scale."""
    clamped = max(min_val, min(max_val, value))
    return ((clamped - min_val) / (max_val - min_val)) * 100
