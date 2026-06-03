"""Evaluation confidence engine (heuristic explainability scores)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

CONFIDENCE_DISCLAIMER = (
    "Confidence scores are heuristic explainability scores and not statistical probabilities."
)


def compute_evaluation_confidence(
    judge_results: List[dict],
    human_votes: Optional[Dict[str, Any]] = None,
    sample_size: int = 0,
) -> Dict[str, Any]:
    """
    Compute heuristic evaluation confidence score (0-100).

    Factors: sample size, judge score variance, human agreement, contradictions.
    """
    factors: Dict[str, float] = {}
    evidence: List[str] = []
    limitations: List[str] = []

    effective_sample = sample_size or len(judge_results)
    sample_factor = min(effective_sample / 20.0, 1.0) * 30
    factors["sample_size"] = sample_factor
    if effective_sample < 5:
        limitations.append("Very small evaluation sample size reduces confidence.")
        evidence.append(f"Only {effective_sample} samples evaluated.")
    else:
        evidence.append(f"{effective_sample} evaluation samples provide moderate coverage.")

    variance_factor = 25.0
    if judge_results:
        all_scores = []
        for result in judge_results:
            for key, val in result.items():
                if key.endswith("_score") and isinstance(val, (int, float)):
                    all_scores.append(float(val))
        if len(all_scores) > 1:
            mean = sum(all_scores) / len(all_scores)
            variance = sum((s - mean) ** 2 for s in all_scores) / len(all_scores)
            variance_factor = max(0, 25 - variance * 5)
            if variance > 2:
                evidence.append("High variance in judge scores suggests inconsistent evaluation.")
            else:
                evidence.append("Judge scores show reasonable consistency across categories.")
    factors["judge_agreement"] = variance_factor

    human_factor = 15.0
    contradictions: List[str] = []
    if human_votes:
        total = human_votes.get("total_comparisons", 0)
        win_rate = human_votes.get("win_rate", 0.5)
        if total >= 3:
            margin = abs(win_rate - 0.5) * 2
            human_factor = margin * 25
            evidence.append(f"Human preference margin: {win_rate:.0%} win rate over {total} comparisons.")
        else:
            limitations.append("Insufficient human preference data for strong confidence.")
        if human_votes.get("contradicts_judge"):
            contradictions.append("Human preferences contradict automated judge rankings.")
            human_factor *= 0.5
    factors["human_agreement"] = human_factor

    coverage_factor = min(len(judge_results[0].keys()) if judge_results else 0, 8) / 8 * 20
    factors["category_coverage"] = coverage_factor

    raw_score = sum(factors.values())
    confidence = min(max(round(raw_score), 0), 100)

    return {
        "confidence_score": confidence,
        "factors": {k: round(v, 2) for k, v in factors.items()},
        "supporting_evidence": evidence,
        "contradictory_signals": contradictions,
        "limitations": limitations,
        "disclaimer": CONFIDENCE_DISCLAIMER,
    }
