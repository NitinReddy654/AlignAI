#!/usr/bin/env python3
"""CLI entrypoint for model evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alignai.alignment.readiness import compute_alignment_readiness
from alignai.config import get_config
from alignai.evaluation.judge import run_batch_evaluation
from alignai.evaluation.reports import generate_evaluation_report, save_report
from alignai.experiments.registry import ExperimentRegistry
from alignai.logging_utils import setup_logging
from alignai.models.generation import generate_response
from alignai.models.loaders import (
    load_base_model,
    load_fft_model,
    load_lora_model,
    load_qlora_model,
)

logger = setup_logging("alignai.evaluate")

DEFAULT_EVAL_PROMPTS = [
    "How do I reset my enterprise account password?",
    "What is the refund policy for annual subscriptions?",
    "My API integration is returning 403 errors. How can I troubleshoot this?",
    "Can you explain the difference between Standard and Enterprise plans?",
    "How do I add a new team member to my organization workspace?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="AlignAI Evaluation Pipeline")
    parser.add_argument("--experiment-id", default="latest")
    parser.add_argument("--prompts-file", default=None, help="JSON file with evaluation prompts")
    parser.add_argument("--mock", action="store_true", help="Use mock responses (no model loading)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    cfg = get_config()
    registry = ExperimentRegistry()
    experiment = registry.get_latest() if args.experiment_id == "latest" else registry.load(args.experiment_id)

    if not experiment:
        logger.error("No experiment found.")
        sys.exit(1)

    prompts = DEFAULT_EVAL_PROMPTS
    if args.prompts_file:
        prompts = json.loads(Path(args.prompts_file).read_text())

    pairs = []
    if args.mock:
        for prompt in prompts:
            pairs.append(
                {"prompt": prompt, "response": f"[Synthetic evaluation response: {prompt[:50]}...]"}
            )
    else:
        if experiment.strategy == "full":
            model, tokenizer = load_fft_model(experiment.output_dir)
        elif experiment.strategy == "lora":
            model, tokenizer = load_lora_model(experiment.model_version, experiment.output_dir)
        elif experiment.strategy == "qlora":
            model, tokenizer = load_qlora_model(experiment.model_version, experiment.output_dir)
        else:
            model, tokenizer = load_base_model(experiment.model_version)

        for prompt in prompts:
            generation = generate_response(
                model,
                tokenizer,
                [{"role": "user", "content": prompt}],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            result = generation.to_dict()
            pairs.append({"prompt": prompt, **result})

    model_name = f"{experiment.strategy}_{experiment.experiment_id}"
    evaluation = run_batch_evaluation(pairs, model_name=model_name) if cfg.openai_api_key else {
        "model_name": model_name,
        "individual_results": [
            {
                "prompt": p["prompt"],
                "response": p["response"],
                "correctness_score": 4,
                "relevance_score": 4,
                "helpfulness_score": 3,
                "instruction_following_score": 4,
                "safety_score": 5,
                "consistency_score": 4,
                "conciseness_score": 3,
                "tone_alignment_score": 4,
            }
            for p in pairs
        ],
        "aggregated_metrics": {"status": "mock", "avg_judge_score": 3.9},
    }

    readiness = compute_alignment_readiness(evaluation.get("aggregated_metrics", {}))
    report = generate_evaluation_report(evaluation, experiment.experiment_id, alignment_readiness=readiness)
    report_path = save_report(report)

    agg = evaluation.get("aggregated_metrics", {})
    safety_mean = agg.get("categories", {}).get("safety", {}).get("mean", 0)
    latency_values = [p.get("latency_ms", 0) for p in pairs if p.get("latency_ms")]
    avg_latency_ms = round(sum(latency_values) / len(latency_values), 2) if latency_values else 0
    registry.update_evaluation_scores(experiment.experiment_id, {
        "alignment_readiness": readiness["alignment_readiness"],
        "avg_judge_score": agg.get("avg_judge_score", 0),
        "evaluation_confidence": report.get("evaluation_confidence", {}).get(
            "confidence_score", 0
        ),
        "safety_score": safety_mean,
        "human_win_rate": 0,
        "avg_latency_ms": avg_latency_ms,
        "inference_cost_usd": 0,
        "dataset_health": readiness.get("breakdown", {}).get("dataset_health", 0),
    })

    eval_path = cfg.artifacts_dir / "evaluations" / f"{experiment.experiment_id}_eval.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(evaluation, indent=2))

    print(json.dumps({"report_path": str(report_path), "readiness": readiness}, indent=2))


if __name__ == "__main__":
    main()
