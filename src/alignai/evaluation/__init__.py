"""Evaluation package."""

from alignai.evaluation.confidence import CONFIDENCE_DISCLAIMER, compute_evaluation_confidence
from alignai.evaluation.judge import evaluate_single_response, run_batch_evaluation
from alignai.evaluation.metrics import aggregate_judge_results
from alignai.evaluation.reports import generate_evaluation_report, save_report
from alignai.evaluation.rubrics import EVALUATION_CATEGORIES

__all__ = [
    "CONFIDENCE_DISCLAIMER",
    "EVALUATION_CATEGORIES",
    "aggregate_judge_results",
    "compute_evaluation_confidence",
    "evaluate_single_response",
    "generate_evaluation_report",
    "run_batch_evaluation",
    "save_report",
]
