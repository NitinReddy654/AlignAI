"""Tests for experiment registry and leaderboard."""

import os

os.environ["ALIGNAI_SKIP_OPENAI_VALIDATION"] = "1"

from alignai.experiments.leaderboard import build_leaderboard, get_best_models
from alignai.experiments.registry import ExperimentRecord, ExperimentRegistry


class TestExperimentRegistry:
    def test_save_and_load(self, tmp_path):
        registry = ExperimentRegistry(registry_dir=tmp_path)
        record = ExperimentRecord(
            experiment_id="exp_test01",
            model_version="Qwen/Qwen2.5-3B-Instruct",
            dataset_version="v1",
            strategy="lora",
            hyperparams={"epochs": 3},
            output_dir="/tmp/test",
            duration_seconds=120.0,
            cost_estimate={"total_cost_usd": 0.05},
        )
        registry.save(record)
        loaded = registry.load("exp_test01")
        assert loaded is not None
        assert loaded.strategy == "lora"

    def test_list_all(self, tmp_path):
        registry = ExperimentRegistry(registry_dir=tmp_path)
        for i in range(3):
            registry.save(
                ExperimentRecord(
                    experiment_id=f"exp_{i}",
                    model_version="test",
                    dataset_version="v1",
                    strategy="lora",
                    hyperparams={},
                    output_dir="/tmp",
                    duration_seconds=60,
                    cost_estimate={"total_cost_usd": 0.01},
                )
            )
        assert len(registry.list_all()) == 3


class TestLeaderboard:
    def test_empty_leaderboard(self):
        df = build_leaderboard(experiments=[])
        assert df.empty

    def test_best_models_empty(self):
        import pandas as pd
        assert get_best_models(pd.DataFrame()) == {}
