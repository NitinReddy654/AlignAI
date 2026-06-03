"""Tests for dataset analysis."""

import json
import os

os.environ["ALIGNAI_SKIP_OPENAI_VALIDATION"] = "1"

from alignai.datasets_analysis.analyzer import analyze_dataset


class TestDatasetAnalyzer:
    def test_analyze_sample_dataset(self, tmp_path):
        data = [
            {"conversation": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]},
            {"conversation": [{"role": "user", "content": "Help me"}, {"role": "assistant", "content": "Sure, how?"}]},
        ]
        path = tmp_path / "test.jsonl"
        with path.open("w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        report = analyze_dataset(path)
        assert report.total_records == 2
        assert report.health_score > 0
        assert report.avg_turns_per_conversation == 2.0

    def test_empty_dataset(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        report = analyze_dataset(path)
        assert report.total_records == 0
        assert report.health_score == 0

    def test_duplicate_detection(self, tmp_path):
        record = {"conversation": [{"role": "user", "content": "Same"}, {"role": "assistant", "content": "Same"}]}
        path = tmp_path / "dup.jsonl"
        with path.open("w") as f:
            f.write(json.dumps(record) + "\n")
            f.write(json.dumps(record) + "\n")

        report = analyze_dataset(path)
        assert report.duplicate_count >= 1
