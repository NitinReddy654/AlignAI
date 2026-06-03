"""Evaluation metrics aggregation."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List

from alignai.evaluation.rubrics import EVALUATION_CATEGORIES


def extract_scores(judge_result: dict) -> Dict[str, float]:
    """Extract numeric scores from a judge JSON result."""
    scores = {}
    for category in EVALUATION_CATEGORIES:
        key = f"{category}_score"
        if key in judge_result:
            scores[category] = float(judge_result[key])
    return scores


def aggregate_judge_results(results: List[dict]) -> Dict[str, Any]:
    """Compute mean, median, stdev for each evaluation category."""
    if not results:
        return {"status": "no_results", "categories": {}}

    all_scores: Dict[str, List[float]] = {cat: [] for cat in EVALUATION_CATEGORIES}
    for result in results:
        scores = extract_scores(result)
        for cat, val in scores.items():
            all_scores[cat].append(val)

    aggregated: Dict[str, Any] = {"status": "completed", "categories": {}, "sample_count": len(results)}
    overall_scores: List[float] = []

    for cat, values in all_scores.items():
        if values:
            cat_stats = {
                "mean": round(statistics.mean(values), 3),
                "median": round(statistics.median(values), 3),
                "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
            aggregated["categories"][cat] = cat_stats
            overall_scores.extend(values)

    if overall_scores:
        aggregated["avg_judge_score"] = round(statistics.mean(overall_scores), 3)
        aggregated["overall_median"] = round(statistics.median(overall_scores), 3)

    return aggregated
