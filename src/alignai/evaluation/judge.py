"""LLM-as-a-Judge evaluation engine using OpenAI."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from alignai.config import get_config
from alignai.evaluation.metrics import aggregate_judge_results
from alignai.evaluation.rubrics import build_judge_messages
from alignai.logging_utils import setup_logging

logger = setup_logging(__name__)


def _parse_judge_response(content: str) -> dict:
    """Parse JSON from judge response, handling markdown fences."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def evaluate_single_response(
    user_prompt: str,
    assistant_response: str,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
) -> dict:
    """Run LLM-as-judge on a single prompt-response pair."""
    cfg = get_config()
    if client is None:
        from openai import OpenAI

        client = OpenAI(api_key=cfg.openai_api_key)
    judge_model = model or cfg.openai_judge_model
    messages = build_judge_messages(user_prompt, assistant_response)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return _parse_judge_response(response.choices[0].message.content)
        except Exception as e:
            logger.warning("Judge attempt %d failed: %s", attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                return {"error": str(e), "status": "failed"}
    return {"status": "failed"}


def run_batch_evaluation(
    prompt_response_pairs: List[Dict[str, Any]],
    model_name: str = "unknown",
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate a batch of prompt-response pairs."""
    results = []
    for pair in prompt_response_pairs:
        judge_result = evaluate_single_response(
            user_prompt=pair["prompt"],
            assistant_response=pair["response"],
            client=client,
        )
        judge_result["prompt"] = pair["prompt"]
        judge_result["response"] = pair["response"]
        judge_result["model_name"] = model_name
        for metric in (
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "tokens_per_second",
        ):
            if metric in pair:
                judge_result[metric] = pair[metric]
        results.append(judge_result)

    aggregated = aggregate_judge_results(results)
    return {
        "model_name": model_name,
        "individual_results": results,
        "aggregated_metrics": aggregated,
    }
