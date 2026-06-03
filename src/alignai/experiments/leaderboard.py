"""Experiment leaderboard and ranking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from alignai.experiments.registry import ExperimentRecord, ExperimentRegistry


@dataclass
class LeaderboardEntry:
    """Single leaderboard row."""

    experiment_id: str
    strategy: str
    model_version: str
    alignment_score: float
    avg_judge_score: float
    human_win_rate: float
    cost_usd: float
    latency_ms: float
    duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "strategy": self.strategy,
            "model_version": self.model_version,
            "alignment_score": round(self.alignment_score, 2),
            "avg_judge_score": round(self.avg_judge_score, 2),
            "human_win_rate": round(self.human_win_rate, 2),
            "cost_usd": round(self.cost_usd, 4),
            "latency_ms": round(self.latency_ms, 2),
            "duration_seconds": round(self.duration_seconds, 2),
        }


def build_leaderboard(
    experiments: Optional[List[ExperimentRecord]] = None,
    sort_by: str = "alignment_score",
) -> pd.DataFrame:
    """Build sortable leaderboard DataFrame from experiment records."""
    registry = ExperimentRegistry()
    experiments = experiments or registry.list_all()
    entries: List[Dict[str, Any]] = []

    for exp in experiments:
        scores = exp.evaluation_scores or {}
        entries.append(
            LeaderboardEntry(
                experiment_id=exp.experiment_id,
                strategy=exp.strategy,
                model_version=exp.model_version,
                alignment_score=scores.get("alignment_readiness", 0.0),
                avg_judge_score=scores.get("avg_judge_score", 0.0),
                human_win_rate=scores.get("human_win_rate", 0.0),
                cost_usd=exp.cost_estimate.get("total_cost_usd", 0.0),
                latency_ms=scores.get("avg_latency_ms", 0.0),
                duration_seconds=exp.duration_seconds,
            ).to_dict()
        )

    df = pd.DataFrame(entries)
    if not df.empty and sort_by in df.columns:
        ascending = sort_by in ("cost_usd", "latency_ms", "duration_seconds")
        df = df.sort_values(sort_by, ascending=ascending)
    return df


def get_best_models(df: pd.DataFrame) -> Dict[str, Any]:
    """Return best model per category."""
    if df.empty:
        return {}
    return {
        "best_alignment": df.loc[df["alignment_score"].idxmax()].to_dict()
        if "alignment_score" in df.columns
        else {},
        "best_judge_score": df.loc[df["avg_judge_score"].idxmax()].to_dict()
        if "avg_judge_score" in df.columns
        else {},
        "lowest_cost": df.loc[df["cost_usd"].idxmin()].to_dict()
        if "cost_usd" in df.columns
        else {},
        "fastest_inference": df.loc[df["latency_ms"].idxmin()].to_dict()
        if "latency_ms" in df.columns
        else {},
        "best_human_preference": df.loc[df["human_win_rate"].idxmax()].to_dict()
        if "human_win_rate" in df.columns
        else {},
    }
