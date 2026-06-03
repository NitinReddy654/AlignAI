"""Tests for Experiment Recommendation Engine."""

import os

os.environ["ALIGNAI_SKIP_OPENAI_VALIDATION"] = "1"

from alignai.experiments.recommendation import (
    RECOMMENDATION_DISCLAIMER,
    CandidateMetrics,
    compute_deployment_recommendation,
    extract_candidate_metrics,
    generate_recommendation_from_registry,
    strategy_display_name,
)
from alignai.experiments.registry import ExperimentRecord, ExperimentRegistry


def _make_exp(
    exp_id: str,
    strategy: str,
    scores: dict,
    cost: float = 0.1,
    latency: float = 100.0,
) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=exp_id,
        model_version="Qwen/Qwen2.5-3B-Instruct",
        dataset_version="v1",
        strategy=strategy,
        hyperparams={},
        output_dir=f"/tmp/{exp_id}",
        duration_seconds=120.0,
        cost_estimate={"total_cost_usd": cost},
        evaluation_scores=scores,
    )


class TestStrategyDisplayName:
    def test_qlora_label(self):
        name = strategy_display_name("qlora", "exp_qlora03ab")
        assert name.startswith("QLoRA-v")

    def test_base_label(self):
        assert strategy_display_name("base", "exp_base01") == "Base"


class TestExtractCandidateMetrics:
    def test_from_experiment_and_report(self):
        exp = _make_exp(
            "exp_lora01",
            "lora",
            {
                "avg_judge_score": 4.2,
                "alignment_readiness": 85,
                "evaluation_confidence": 72,
                "human_win_rate": 0.65,
                "avg_latency_ms": 150,
                "safety_score": 4.5,
            },
        )
        report = {
            "final_score": 4.2,
            "alignment_readiness": {"alignment_readiness": 85},
            "evaluation_confidence": {"confidence_score": 72},
            "category_breakdown": {"safety": {"mean": 4.8}},
        }
        m = extract_candidate_metrics(exp, report=report, dataset_health=90)
        assert m.display_name.startswith("LoRA")
        assert m.alignment_readiness == 85
        assert m.dataset_health == 90


class TestComputeDeploymentRecommendation:
    def _cohort(self) -> list[CandidateMetrics]:
        return [
            CandidateMetrics(
                experiment_id="exp_base",
                strategy="base",
                display_name="Base",
                judge_quality=3.5,
                alignment_readiness=55,
                evaluation_confidence=60,
                safety_score=4.0,
                human_win_rate=0.3,
                latency_ms=80,
                training_cost_usd=0.0,
                inference_cost_usd=0.001,
                dataset_health=90,
            ),
            CandidateMetrics(
                experiment_id="exp_lora01",
                strategy="lora",
                display_name="LoRA-v1",
                judge_quality=4.5,
                alignment_readiness=88,
                evaluation_confidence=75,
                safety_score=4.6,
                human_win_rate=0.7,
                latency_ms=120,
                training_cost_usd=0.5,
                inference_cost_usd=0.002,
                dataset_health=90,
            ),
            CandidateMetrics(
                experiment_id="exp_qlora02",
                strategy="qlora",
                display_name="QLoRA-v2",
                judge_quality=4.3,
                alignment_readiness=92,
                evaluation_confidence=80,
                safety_score=4.7,
                human_win_rate=0.75,
                latency_ms=130,
                training_cost_usd=0.15,
                inference_cost_usd=0.0015,
                dataset_health=90,
            ),
            CandidateMetrics(
                experiment_id="exp_full03",
                strategy="full",
                display_name="FFT-v3",
                judge_quality=4.6,
                alignment_readiness=85,
                evaluation_confidence=70,
                safety_score=4.5,
                human_win_rate=0.6,
                latency_ms=200,
                training_cost_usd=1.2,
                inference_cost_usd=0.003,
                dataset_health=90,
            ),
        ]

    def test_returns_recommendation(self):
        rec = compute_deployment_recommendation(self._cohort())
        assert rec is not None
        assert rec.recommended_model
        assert rec.disclaimer == RECOMMENDATION_DISCLAIMER
        assert len(rec.reasons) >= 1
        assert len(rec.candidate_rankings) == 4

    def test_qlora_often_wins_cost_alignment_tradeoff(self):
        rec = compute_deployment_recommendation(self._cohort())
        # QLoRA has high alignment, strong human pref, low cost
        assert rec.strategy in ("qlora", "lora", "full")

    def test_alternatives_populated(self):
        rec = compute_deployment_recommendation(self._cohort())
        assert len(rec.alternative_recommendations) >= 1
        assert rec.alternative_recommendations[0].preferred_when

    def test_warnings_low_readiness(self):
        cohort = self._cohort()
        for c in cohort:
            c.alignment_readiness = 45
        rec = compute_deployment_recommendation(cohort)
        assert rec is not None
        assert any("Alignment Readiness" in w for w in rec.deployment_warnings)

    def test_empty_returns_none(self):
        assert compute_deployment_recommendation([]) is None

    def test_export_dict(self):
        rec = compute_deployment_recommendation(self._cohort())
        d = rec.to_dict()
        assert "recommended_model" in d
        assert "disclaimer" in d
        assert "tradeoff_analysis" in d


class TestGenerateFromRegistry:
    def test_registry_integration(self, tmp_path):
        registry = ExperimentRegistry(registry_dir=tmp_path / "experiments")
        for exp_id, strategy, scores, cost in [
            ("exp_base01", "base", {"avg_judge_score": 3.2, "alignment_readiness": 50}, 0),
            ("exp_lora02", "lora", {"avg_judge_score": 4.4, "alignment_readiness": 90}, 0.4),
            ("exp_qlora03", "qlora", {"avg_judge_score": 4.2, "alignment_readiness": 95}, 0.12),
        ]:
            registry.save(
                _make_exp(exp_id, strategy, {**scores, "evaluation_confidence": 70}, cost)
            )

        rec = generate_recommendation_from_registry(registry=registry, dataset_health=88)
        assert rec is not None
        assert rec.experiment_id in ("exp_qlora03", "exp_lora02")
