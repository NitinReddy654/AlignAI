"""Tests for AlignAI core modules."""

import os

os.environ["ALIGNAI_SKIP_OPENAI_VALIDATION"] = "1"

from alignai.alignment.readiness import compute_alignment_readiness
from alignai.alignment.scoring import normalize_score
from alignai.cost import estimate_training_cost
from alignai.evaluation.confidence import CONFIDENCE_DISCLAIMER, compute_evaluation_confidence
from alignai.evaluation.metrics import aggregate_judge_results


class TestNormalizeScore:
    def test_min_score(self):
        assert normalize_score(1.0) == 0.0

    def test_max_score(self):
        assert normalize_score(5.0) == 100.0

    def test_mid_score(self):
        assert normalize_score(3.0) == 50.0


class TestCostEstimation:
    def test_lora_cost(self):
        cost = estimate_training_cost(3600, "lora", "3B")
        assert cost.total_cost_usd > 0
        assert cost.strategy == "lora"

    def test_qlora_cheaper_than_full(self):
        qlora = estimate_training_cost(3600, "qlora", "3B")
        full = estimate_training_cost(3600, "full", "3B")
        assert qlora.total_cost_usd < full.total_cost_usd


class TestJudgeMetrics:
    def test_aggregate_empty(self):
        result = aggregate_judge_results([])
        assert result["status"] == "no_results"

    def test_aggregate_scores(self):
        results = [
            {"correctness_score": 4, "relevance_score": 5, "helpfulness_score": 4,
             "instruction_following_score": 4, "safety_score": 5, "consistency_score": 4,
             "conciseness_score": 3, "tone_alignment_score": 4},
        ]
        agg = aggregate_judge_results(results)
        assert agg["status"] == "completed"
        assert "correctness" in agg["categories"]
        assert agg["avg_judge_score"] > 0


class TestAlignmentReadiness:
    def test_readiness_with_metrics(self):
        metrics = {
            "categories": {
                "correctness": {"mean": 4.5},
                "relevance": {"mean": 4.0},
                "helpfulness": {"mean": 4.0},
                "tone_alignment": {"mean": 4.0},
                "safety": {"mean": 5.0},
                "consistency": {"mean": 4.0},
                "instruction_following": {"mean": 4.0},
                "conciseness": {"mean": 3.5},
            }
        }
        result = compute_alignment_readiness(metrics)
        assert 0 <= result["alignment_readiness"] <= 100
        assert "recommendations" in result
        assert "breakdown" in result


class TestEvaluationConfidence:
    def test_confidence_range(self):
        results = [{"correctness_score": 4, "relevance_score": 5}]
        conf = compute_evaluation_confidence(results, sample_size=10)
        assert 0 <= conf["confidence_score"] <= 100
        assert conf["disclaimer"] == CONFIDENCE_DISCLAIMER

    def test_confidence_with_human_votes(self):
        results = [{"correctness_score": 4}]
        votes = {"total_comparisons": 10, "win_rate": 0.8}
        conf = compute_evaluation_confidence(results, votes, sample_size=10)
        assert conf["confidence_score"] > 0
