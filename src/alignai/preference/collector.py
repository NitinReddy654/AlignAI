"""Human preference collection and analytics."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from alignai.config import get_config
from alignai.logging_utils import setup_logging

logger = setup_logging(__name__)


def prepare_comparison_pairs(
    evaluation_results: List[dict],
) -> List[Tuple[str, str, str, str, str]]:
    """Build A/B comparison pairs from evaluation results."""
    prompt_responses: Dict[str, List[dict]] = defaultdict(list)
    for row in evaluation_results:
        prompt = row.get("prompt", "").strip()
        model = row.get("model_name", "").strip()
        resp = row.get("response", "").strip()
        if prompt and model and resp:
            prompt_responses[prompt].append({"model": model, "response": resp})

    pairs = []
    for prompt, resps in prompt_responses.items():
        if len(resps) < 2:
            continue
        for r1, r2 in combinations(resps, 2):
            pairs.append((prompt, r1["model"], r1["response"], r2["model"], r2["response"]))
    return pairs


def calculate_preference_analytics(votes: Dict[tuple, dict]) -> Dict[str, Any]:
    """Compute win rates and head-to-head analytics from human votes."""
    model_wins: Dict[str, int] = defaultdict(int)
    model_ties: Dict[str, int] = defaultdict(int)
    model_losses: Dict[str, int] = defaultdict(int)
    model_comparisons: Dict[str, int] = defaultdict(int)
    all_models: set = set()

    for key, result in votes.items():
        model1, model2 = key[0], key[1]
        all_models.update([model1, model2])
        model_comparisons[model1] += 1
        model_comparisons[model2] += 1
        winner = result.get("winner")

        if winner == model1:
            model_wins[model1] += 1
            model_losses[model2] += 1
        elif winner == model2:
            model_wins[model2] += 1
            model_losses[model1] += 1
        else:
            model_ties[model1] += 1
            model_ties[model2] += 1

    summary = []
    for model in sorted(all_models):
        total = model_comparisons[model]
        win_rate = (model_wins[model] / total * 100) if total else 0
        summary.append(
            {
                "model": model,
                "wins": model_wins[model],
                "ties": model_ties[model],
                "losses": model_losses[model],
                "total_comparisons": total,
                "win_rate": round(win_rate, 1),
            }
        )

    df = pd.DataFrame(summary)
    if not df.empty:
        df = df.sort_values("win_rate", ascending=False)

    top_win_rate = df.iloc[0]["win_rate"] / 100 if not df.empty else 0.0
    return {
        "summary": df.to_dict(orient="records") if not df.empty else [],
        "total_votes": len(votes),
        "win_rate": top_win_rate,
        "total_comparisons": sum(model_comparisons.values()) // 2 if model_comparisons else 0,
        "ranking_distribution": summary,
    }


def save_preferences(votes: dict, session_id: str) -> Path:
    """Persist human preference votes to JSON."""
    cfg = get_config()
    out_dir = cfg.artifacts_dir / "preferences"
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in votes.items()}
    path = out_dir / f"preferences_{session_id}.json"
    path.write_text(json.dumps(serializable, indent=2))
    logger.info("Saved preferences: %s", path)
    return path


def load_evaluation_results(filepath: str | Path) -> List[dict]:
    """Load evaluation results JSON for A/B testing."""
    path = Path(filepath)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "individual_results" in data:
        return data["individual_results"]
    return []
