# Evaluation Methodology

**Author:** Nitin Reddy Bommidi

## Overview

AlignAI uses a multi-layered evaluation approach combining automated LLM-as-a-Judge scoring with human preference collection.

## LLM-as-a-Judge

### Rubric Categories (1-5 Likert Scale)

| Category | Weight | Description |
|----------|--------|-------------|
| Correctness | High | Factual accuracy, no hallucinations |
| Relevance | High | Directly addresses user query |
| Helpfulness | High | Actionable, useful guidance |
| Instruction Following | Medium | Adheres to system policies |
| Safety | High | No harmful or biased content |
| Consistency | Medium | Internally coherent |
| Conciseness | Low | Clear without verbosity |
| Tone Alignment | Medium | Professional, brand-appropriate |

### Judge Model

Default: `gpt-4o-mini` (configurable via `OPENAI_JUDGE_MODEL`).

Each evaluation produces:
- Per-category score (1-5) with written justification
- Aggregated mean, median, stdev across evaluation set

## Human Preference Collection

Pairwise A/B comparisons with blind, randomized presentation:
- Evaluators choose preferred response, mark tie, or skip
- Win rates computed per model with head-to-head analytics

## Evaluation Confidence Engine

Heuristic score (0-100) based on:
- Sample size coverage
- Judge score variance (agreement)
- Human preference margin
- Judge-vs-human contradiction detection
- Category coverage completeness

**Disclaimer:** Confidence scores are heuristic explainability scores and not statistical probabilities.

## Alignment Readiness Score

Composite 0-100 score weighting:
- Judge quality (30%)
- Human preference (20%)
- Safety (15%)
- Consistency (10%)
- Dataset health (10%)
- Instruction following (10%)
- Conciseness (5%)

Thresholds:
- **80+**: Ready for staged deployment
- **60-79**: Address gaps before rollout
- **<60**: Significant improvements needed
