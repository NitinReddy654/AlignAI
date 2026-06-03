"""Preference package."""

from alignai.preference.collector import (
    calculate_preference_analytics,
    load_evaluation_results,
    prepare_comparison_pairs,
    save_preferences,
)

__all__ = [
    "calculate_preference_analytics",
    "load_evaluation_results",
    "prepare_comparison_pairs",
    "save_preferences",
]
