# Deployment Decision Engine

**Author:** Nitin Reddy Bommidi

## Purpose

The Deployment Decision Engine compares completed evaluations across **Base**, **LoRA**, **QLoRA**, and **Full Fine-Tuning** experiments and identifies the strongest deployment candidate.

## Deployment Boundary

> **Decision-support outputs are not automatic deployment approvals.**
> Human review, security review, and production validation are required before any rollout.

## Metrics Compared

| Metric | Source |
|--------|--------|
| Judge quality | `avg_judge_score` (1-5) |
| Alignment readiness | Alignment Readiness Score (0-100) |
| Evaluation confidence | Evaluation Confidence Engine (0-100) |
| Safety | Judge `safety` category mean |
| Human preference | Win rate from A/B comparisons |
| Latency | `avg_latency_ms` per experiment |
| Training cost | `cost_estimate.total_cost_usd` |
| Inference cost | `inference_cost_usd` in evaluation scores |
| Dataset health | Dataset analyzer health score |

## Scoring

A weighted **composite deployment score** ranks candidates:

- Alignment readiness (25%)
- Judge quality (20%)
- Safety (15%)
- Human win rate (15%)
- Evaluation confidence (10%)
- Cost efficiency (5%)
- Latency efficiency (5%)
- Dataset health (5%)

## Output Structure

```json
{
  "recommended_model": "QLoRA-v3",
  "reasons": ["..."],
  "supporting_evidence": ["..."],
  "tradeoff_analysis": ["..."],
  "alternative_recommendations": [{"display_name": "...", "preferred_when": ["..."]}],
  "deployment_warnings": ["..."],
  "disclaimer": "..."
}
```

## UI Surfaces

- **Leaderboard** - decision panel above rankings
- **Evaluation Center** - post-evaluation deployment guidance
- **Reports** - Deployment Decision tab and JSON export

## API

```python
from alignai.experiments.recommendation import (
    generate_recommendation_from_registry,
    save_recommendation_report,
)

rec = generate_recommendation_from_registry()
if rec:
    save_recommendation_report(rec)
```
