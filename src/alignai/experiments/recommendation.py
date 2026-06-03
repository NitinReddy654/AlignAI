"""Experiment Recommendation Engine - deployment decision support."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from alignai.config import get_config
from alignai.experiments.registry import ExperimentRecord, ExperimentRegistry
from alignai.logging_utils import setup_logging

logger = setup_logging(__name__)

RECOMMENDATION_DISCLAIMER = (
    "Recommendations are decision-support outputs based on available evaluation artifacts. "
    "They are not automatic deployment decisions. Human review and production validation "
    "are required before rollout."
)

STRATEGY_LABELS = {
    "base": "Base",
    "lora": "LoRA",
    "qlora": "QLoRA",
    "full": "Full Fine-Tuning",
}

# Weights for composite deployment score (sum = 1.0)
SCORE_WEIGHTS = {
    "alignment_readiness": 0.25,
    "judge_quality": 0.20,
    "safety": 0.15,
    "human_win_rate": 0.15,
    "evaluation_confidence": 0.10,
    "cost_efficiency": 0.05,
    "latency_efficiency": 0.05,
    "dataset_health": 0.05,
}


@dataclass
class CandidateMetrics:
    """Normalized metrics for one experiment variant."""

    experiment_id: str
    strategy: str
    display_name: str
    judge_quality: float  # avg judge 1-5, normalized to 0-100 internally
    alignment_readiness: float
    evaluation_confidence: float
    safety_score: float  # 1-5 scale from judge
    human_win_rate: float  # 0-1
    latency_ms: float
    training_cost_usd: float
    inference_cost_usd: float
    dataset_health: float
    model_version: str = ""
    composite_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AlternativeRecommendation:
    """When another model may be preferred."""

    display_name: str
    experiment_id: str
    preferred_when: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeploymentRecommendation:
    """Full recommendation output."""

    recommended_model: str
    experiment_id: str
    strategy: str
    reasons: List[str]
    supporting_evidence: List[str]
    tradeoff_analysis: List[str]
    alternative_recommendations: List[AlternativeRecommendation]
    deployment_warnings: List[str]
    disclaimer: str = RECOMMENDATION_DISCLAIMER
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    candidate_rankings: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["alternative_recommendations"] = [
            a if isinstance(a, dict) else a.to_dict()
            for a in self.alternative_recommendations
        ]
        return data


def strategy_display_name(strategy: str, experiment_id: str) -> str:
    """Human-readable model label, e.g. QLoRA-v3 from exp_qlora03."""
    label = STRATEGY_LABELS.get(strategy.lower(), strategy.title())
    suffix = experiment_id.replace("exp_", "")[-4:] if experiment_id else "01"
    short = {"qlora": "QLoRA", "lora": "LoRA", "full": "FFT", "base": "Base"}.get(
        strategy.lower(), label
    )
    if strategy.lower() in ("qlora", "lora"):
        return f"{short}-v{suffix}"
    if strategy.lower() == "full":
        return f"FFT-v{suffix}"
    if strategy.lower() == "base":
        return "Base"
    return f"{label}-{suffix}"


def _normalize_judge_to_100(score_1_5: float) -> float:
    if score_1_5 <= 0:
        return 0.0
    return min(100.0, max(0.0, ((score_1_5 - 1) / 4) * 100))


def _normalize_safety_to_100(score_1_5: float) -> float:
    return _normalize_judge_to_100(score_1_5)


def _cost_efficiency(cost: float, max_cost: float) -> float:
    if max_cost <= 0:
        return 100.0 if cost <= 0 else 50.0
    return max(0.0, 100.0 * (1 - cost / max_cost))


def _latency_efficiency(latency: float, max_latency: float) -> float:
    if max_latency <= 0:
        return 100.0 if latency <= 0 else 50.0
    return max(0.0, 100.0 * (1 - latency / max_latency))


def extract_candidate_metrics(
    experiment: ExperimentRecord,
    report: Optional[Dict[str, Any]] = None,
    dataset_health: float = 0.0,
) -> CandidateMetrics:
    """Build candidate metrics from experiment record and optional evaluation report."""
    scores = experiment.evaluation_scores or {}
    report = report or {}

    readiness = report.get("alignment_readiness", {})
    if isinstance(readiness, dict):
        alignment = readiness.get("alignment_readiness", scores.get("alignment_readiness", 0))
    else:
        alignment = scores.get("alignment_readiness", 0)

    confidence = report.get("evaluation_confidence", {})
    if isinstance(confidence, dict):
        conf_score = confidence.get("confidence_score", scores.get("evaluation_confidence", 0))
    else:
        conf_score = scores.get("evaluation_confidence", 0)

    categories = report.get("category_breakdown", {})
    safety = scores.get("safety_score", 0)
    if not safety and categories:
        safety_cat = categories.get("safety", {})
        safety = safety_cat.get("mean", 0) if isinstance(safety_cat, dict) else 0

    judge = scores.get("avg_judge_score", report.get("final_score", 0))

    return CandidateMetrics(
        experiment_id=experiment.experiment_id,
        strategy=experiment.strategy.lower(),
        display_name=strategy_display_name(experiment.strategy, experiment.experiment_id),
        judge_quality=judge,
        alignment_readiness=float(alignment),
        evaluation_confidence=float(conf_score),
        safety_score=float(safety),
        human_win_rate=float(scores.get("human_win_rate", 0)),
        latency_ms=float(scores.get("avg_latency_ms", 0)),
        training_cost_usd=float(
            experiment.cost_estimate.get("total_cost_usd", 0)
        ),
        inference_cost_usd=float(scores.get("inference_cost_usd", 0)),
        dataset_health=float(
            scores.get("dataset_health", dataset_health)
        ),
        model_version=experiment.model_version,
    )


def _compute_composite(candidates: List[CandidateMetrics]) -> None:
    """Compute composite deployment score in-place."""
    if not candidates:
        return

    max_cost = max(c.training_cost_usd for c in candidates) or 1.0
    max_latency = max(c.latency_ms for c in candidates) or 1.0
    max_inference = max(c.inference_cost_usd for c in candidates) or 1.0
    max_train_inf = max(max_cost, max_inference)

    for c in candidates:
        judge_100 = _normalize_judge_to_100(c.judge_quality)
        safety_100 = _normalize_safety_to_100(c.safety_score)
        human_100 = c.human_win_rate * 100 if c.human_win_rate <= 1 else c.human_win_rate
        inf_cost = c.inference_cost_usd or c.training_cost_usd * 0.01
        cost_eff = _cost_efficiency(
            c.training_cost_usd + inf_cost, max_train_inf + 1e-9
        )
        lat_eff = _latency_efficiency(c.latency_ms, max_latency)

        c.composite_score = (
            SCORE_WEIGHTS["alignment_readiness"] * min(c.alignment_readiness, 100)
            + SCORE_WEIGHTS["judge_quality"] * judge_100
            + SCORE_WEIGHTS["safety"] * safety_100
            + SCORE_WEIGHTS["human_win_rate"] * min(human_100, 100)
            + SCORE_WEIGHTS["evaluation_confidence"] * min(c.evaluation_confidence, 100)
            + SCORE_WEIGHTS["cost_efficiency"] * cost_eff
            + SCORE_WEIGHTS["latency_efficiency"] * lat_eff
            + SCORE_WEIGHTS["dataset_health"] * min(c.dataset_health, 100)
        )


def _find_by_strategy(
    candidates: List[CandidateMetrics], strategy: str
) -> Optional[CandidateMetrics]:
    for c in candidates:
        if c.strategy == strategy.lower():
            return c
    return None


def _pct_of_reference(value: float, reference: float) -> Optional[float]:
    if reference <= 0:
        return None
    return round((value / reference) * 100, 1)


def _build_reasons(
    winner: CandidateMetrics,
    candidates: List[CandidateMetrics],
) -> List[str]:
    """Generate bullet reasons comparing winner to cohort."""
    reasons: List[str] = []

    best_alignment = max(candidates, key=lambda c: c.alignment_readiness)
    if winner.experiment_id == best_alignment.experiment_id:
        reasons.append(
            f"Highest Alignment Readiness Score ({winner.alignment_readiness:.0f}/100)"
        )

    best_human = max(candidates, key=lambda c: c.human_win_rate)
    if winner.human_win_rate >= best_human.human_win_rate * 0.95 and winner.human_win_rate > 0:
        reasons.append(
            f"Strong human preference win rate ({winner.human_win_rate * 100:.0f}%)"
        )

    lora_ref = _find_by_strategy(candidates, "lora")
    if lora_ref and winner.experiment_id != lora_ref.experiment_id:
        pct = _pct_of_reference(winner.judge_quality, lora_ref.judge_quality)
        if pct and pct >= 90:
            reasons.append(f"{pct:.0f}% of LoRA judge quality")
    elif lora_ref and winner.experiment_id == lora_ref.experiment_id:
        best_judge = max(candidates, key=lambda c: c.judge_quality)
        if winner.experiment_id == best_judge.experiment_id:
            reasons.append("Top judge quality score in cohort")

    cheapest = min(candidates, key=lambda c: c.training_cost_usd)
    if winner.experiment_id == cheapest.experiment_id and len(candidates) > 1:
        others = [c for c in candidates if c.experiment_id != winner.experiment_id]
        if others:
            avg_other = sum(c.training_cost_usd for c in others) / len(others)
            if avg_other > 0:
                savings = (1 - winner.training_cost_usd / avg_other) * 100
                if savings > 10:
                    reasons.append(
                        f"{savings:.0f}% lower estimated training cost vs cohort average"
                    )

    if winner.evaluation_confidence >= 70:
        reasons.append(
            f"Solid evaluation confidence ({winner.evaluation_confidence:.0f}/100)"
        )

    if winner.safety_score >= 4.0:
        reasons.append(f"Strong safety score ({winner.safety_score:.1f}/5)")

    if not reasons:
        reasons.append(
            f"Best overall composite deployment score ({winner.composite_score:.1f}/100)"
        )

    return reasons[:6]


def _build_tradeoffs(
    winner: CandidateMetrics,
    candidates: List[CandidateMetrics],
) -> List[str]:
    """Tradeoff analysis vs other strategies."""
    tradeoffs: List[str] = []
    for c in sorted(candidates, key=lambda x: x.composite_score, reverse=True):
        if c.experiment_id == winner.experiment_id:
            continue
        if c.judge_quality > winner.judge_quality + 0.2:
            tradeoffs.append(
                f"{c.display_name} offers higher judge quality "
                f"({c.judge_quality:.2f} vs {winner.judge_quality:.2f})"
            )
        if c.training_cost_usd < winner.training_cost_usd * 0.7:
            tradeoffs.append(
                f"{c.display_name} has lower training cost "
                f"(${c.training_cost_usd:.4f} vs ${winner.training_cost_usd:.4f})"
            )
        if c.latency_ms > 0 and winner.latency_ms > 0 and c.latency_ms < winner.latency_ms * 0.8:
            tradeoffs.append(
                f"{c.display_name} has lower inference latency "
                f"({c.latency_ms:.0f}ms vs {winner.latency_ms:.0f}ms)"
            )
    if not tradeoffs:
        tradeoffs.append(
            "Recommended model leads on composite score; marginal tradeoffs across single metrics."
        )
    return tradeoffs[:5]


def _build_alternatives(
    winner: CandidateMetrics,
    candidates: List[CandidateMetrics],
) -> List[AlternativeRecommendation]:
    """Suggest when other models may be preferred."""
    alts: List[AlternativeRecommendation] = []
    ranked = sorted(candidates, key=lambda c: c.composite_score, reverse=True)

    for c in ranked[1:4]:
        when: List[str] = []
        if c.judge_quality > winner.judge_quality:
            when.append("Maximum response quality is the top priority")
        if c.training_cost_usd < winner.training_cost_usd:
            when.append("Training budget is severely constrained")
        if c.latency_ms > 0 and (winner.latency_ms <= 0 or c.latency_ms < winner.latency_ms):
            when.append("Lowest inference latency is required")
        if c.safety_score > winner.safety_score + 0.3:
            when.append("Highest safety margin is required for regulated workloads")
        if c.human_win_rate > winner.human_win_rate:
            when.append("Human evaluators consistently preferred this variant")
        if not when:
            when.append("Comparable composite score with different metric emphasis")
        alts.append(
            AlternativeRecommendation(
                display_name=c.display_name,
                experiment_id=c.experiment_id,
                preferred_when=when,
            )
        )
    return alts


def _build_warnings(
    winner: CandidateMetrics,
    candidates: List[CandidateMetrics],
) -> List[str]:
    """Deployment warnings for recommended model."""
    warnings: List[str] = []
    if winner.alignment_readiness < 60:
        warnings.append(
            "Alignment Readiness is below 60 - address gaps before production rollout."
        )
    if winner.evaluation_confidence < 50:
        warnings.append(
            "Evaluation confidence is low - collect more judge and human feedback samples."
        )
    if winner.safety_score < 3.5 and winner.safety_score > 0:
        warnings.append("Safety score is below threshold - review content filters.")
    if winner.human_win_rate == 0 and any(c.human_win_rate > 0 for c in candidates):
        warnings.append(
            "No human preference data for recommended model - run A/B review before deploy."
        )
    if winner.dataset_health < 70 and winner.dataset_health > 0:
        warnings.append("Dataset health score is suboptimal - review training data quality.")
    if len(candidates) < 2:
        warnings.append(
            "Only one evaluated variant available - comparison confidence is limited."
        )
    strategies_present = {c.strategy for c in candidates}
    expected = {"base", "lora", "qlora", "full"}
    missing = expected - strategies_present
    if missing:
        warnings.append(
            f"Incomplete strategy comparison - missing: {', '.join(sorted(missing))}"
        )
    return warnings


def compute_deployment_recommendation(
    candidates: List[CandidateMetrics],
) -> Optional[DeploymentRecommendation]:
    """
    Compare candidates and return deployment recommendation.

    Returns None if no candidates provided.
    """
    if not candidates:
        return None

    _compute_composite(candidates)
    ranked = sorted(candidates, key=lambda c: c.composite_score, reverse=True)
    winner = ranked[0]

    rankings = [
        {
            "rank": i + 1,
            "display_name": c.display_name,
            "experiment_id": c.experiment_id,
            "strategy": c.strategy,
            "composite_score": round(c.composite_score, 2),
            "alignment_readiness": c.alignment_readiness,
            "judge_quality": c.judge_quality,
            "evaluation_confidence": c.evaluation_confidence,
            "safety_score": c.safety_score,
            "human_win_rate": round(c.human_win_rate * 100, 1)
            if c.human_win_rate <= 1
            else c.human_win_rate,
            "latency_ms": c.latency_ms,
            "training_cost_usd": c.training_cost_usd,
            "inference_cost_usd": c.inference_cost_usd,
            "dataset_health": c.dataset_health,
        }
        for i, c in enumerate(ranked)
    ]

    return DeploymentRecommendation(
        recommended_model=winner.display_name,
        experiment_id=winner.experiment_id,
        strategy=winner.strategy,
        reasons=_build_reasons(winner, candidates),
        supporting_evidence=[
            f"Composite deployment score: {winner.composite_score:.1f}/100 (rank 1 of {len(candidates)})",
            f"Judge quality: {winner.judge_quality:.2f}/5",
            f"Alignment readiness: {winner.alignment_readiness:.0f}/100",
            f"Evaluation confidence: {winner.evaluation_confidence:.0f}/100",
            f"Safety: {winner.safety_score:.1f}/5",
            f"Training cost estimate: ${winner.training_cost_usd:.4f}",
            f"Latency: {winner.latency_ms:.0f}ms" if winner.latency_ms else "Latency: not measured",
        ],
        tradeoff_analysis=_build_tradeoffs(winner, candidates),
        alternative_recommendations=_build_alternatives(winner, candidates),
        deployment_warnings=_build_warnings(winner, candidates),
        candidate_rankings=rankings,
    )


def _load_report_for_experiment(
    experiment_id: str, reports_dir: Path
) -> Optional[Dict[str, Any]]:
    """Find latest report JSON for an experiment."""
    if not reports_dir.exists():
        return None
    matches = sorted(
        reports_dir.glob(f"report_{experiment_id}_*.json"),
        reverse=True,
    )
    if matches:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    return None


def generate_recommendation_from_registry(
    registry: Optional[ExperimentRegistry] = None,
    dataset_health: Optional[float] = None,
    min_evaluated: bool = True,
) -> Optional[DeploymentRecommendation]:
    """
    Build recommendation from all experiments with evaluation scores.

    Loads optional report JSON to enrich confidence, safety, and readiness.
    """
    registry = registry or ExperimentRegistry()
    cfg = get_config()
    reports_dir = cfg.artifacts_dir / "reports"

    if dataset_health is None:
        health_path = cfg.artifacts_dir / "datasets"
        if health_path.exists():
            health_files = list(health_path.glob("*_health.json"))
            if health_files:
                data = json.loads(health_files[-1].read_text(encoding="utf-8"))
                dataset_health = data.get("health_score", 0.0)

    experiments = registry.list_all()
    candidates: List[CandidateMetrics] = []

    for exp in experiments:
        if min_evaluated and not exp.evaluation_scores:
            continue
        report = _load_report_for_experiment(exp.experiment_id, reports_dir)
        candidates.append(
            extract_candidate_metrics(
                exp,
                report=report,
                dataset_health=dataset_health or 0.0,
            )
        )

    return compute_deployment_recommendation(candidates)


def save_recommendation_report(
    recommendation: DeploymentRecommendation,
    output_dir: Optional[Path] = None,
) -> Path:
    """Persist exportable deployment recommendation JSON."""
    cfg = get_config()
    out_dir = output_dir or cfg.artifacts_dir / "recommendations"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"deployment_recommendation_{ts}.json"
    path.write_text(json.dumps(recommendation.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved deployment recommendation: %s", path)
    return path
