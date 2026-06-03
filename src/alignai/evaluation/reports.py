"""Exportable evaluation report generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from alignai.config import get_config
from alignai.evaluation.confidence import compute_evaluation_confidence

try:
    from alignai.experiments.recommendation import (
        generate_recommendation_from_registry,
    )
except ImportError:
    generate_recommendation_from_registry = None  # type: ignore


def generate_evaluation_report(
    evaluation_data: Dict[str, Any],
    experiment_id: str,
    human_votes: Optional[Dict[str, Any]] = None,
    alignment_readiness: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a comprehensive evaluation report."""
    judge_results = evaluation_data.get("individual_results", [])
    confidence = compute_evaluation_confidence(judge_results, human_votes)

    report = {
        "report_id": f"report_{experiment_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "experiment_id": experiment_id,
        "model_name": evaluation_data.get("model_name", "unknown"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "final_score": evaluation_data.get("aggregated_metrics", {}).get("avg_judge_score", 0),
        "category_breakdown": evaluation_data.get("aggregated_metrics", {}).get("categories", {}),
        "judge_reasoning_samples": _extract_reasoning_samples(judge_results),
        "human_preference_results": human_votes or {},
        "evaluation_confidence": confidence,
        "alignment_readiness": alignment_readiness or {},
        "supporting_examples": judge_results[:3],
    }
    if generate_recommendation_from_registry is not None:
        cohort_rec = generate_recommendation_from_registry()
        if cohort_rec:
            report["deployment_recommendation"] = cohort_rec.to_dict()
    return report


def _extract_reasoning_samples(results: list, max_samples: int = 3) -> list:
    """Extract justification samples from judge results."""
    samples = []
    for result in results[:max_samples]:
        sample = {"prompt": result.get("prompt", "")}
        for key, val in result.items():
            if key.endswith("_justification"):
                sample[key] = val
        samples.append(sample)
    return samples


def save_report(report: Dict[str, Any], output_dir: Optional[Path] = None) -> Path:
    """Persist report to JSON file."""
    cfg = get_config()
    out_dir = output_dir or cfg.artifacts_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report['report_id']}.json"
    path.write_text(json.dumps(report, indent=2))
    return path
