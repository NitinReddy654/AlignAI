"""Tests for human preference analytics."""

import os

os.environ["ALIGNAI_SKIP_OPENAI_VALIDATION"] = "1"

from alignai.preference.collector import calculate_preference_analytics, prepare_comparison_pairs


class TestPreferenceCollector:
    def test_prepare_pairs(self):
        results = [
            {"prompt": "Q1", "model_name": "model_a", "response": "resp a"},
            {"prompt": "Q1", "model_name": "model_b", "response": "resp b"},
            {"prompt": "Q2", "model_name": "model_a", "response": "resp a2"},
            {"prompt": "Q2", "model_name": "model_b", "response": "resp b2"},
        ]
        pairs = prepare_comparison_pairs(results)
        assert len(pairs) == 2

    def test_preference_analytics(self):
        votes = {
            ("model_a", "model_b", "Q1"): {"winner": "model_a"},
            ("model_a", "model_b", "Q2"): {"winner": "model_b"},
            ("model_a", "model_b", "Q3"): {"winner": "tie"},
        }
        analytics = calculate_preference_analytics(votes)
        assert analytics["total_votes"] == 3
        assert len(analytics["summary"]) == 2
