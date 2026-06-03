"""Tests for evaluation reports."""

import os

os.environ["ALIGNAI_SKIP_OPENAI_VALIDATION"] = "1"

from alignai.evaluation.reports import generate_evaluation_report


class TestEvaluationReports:
    def test_generate_report(self):
        evaluation = {
            "model_name": "lora_test",
            "individual_results": [
                {"prompt": "test", "response": "answer", "correctness_score": 4,
                 "correctness_justification": "Accurate response."},
            ],
            "aggregated_metrics": {
                "avg_judge_score": 4.0,
                "categories": {"correctness": {"mean": 4.0}},
            },
        }
        report = generate_evaluation_report(evaluation, "exp_test")
        assert "report_id" in report
        assert report["final_score"] == 4.0
        assert "evaluation_confidence" in report
