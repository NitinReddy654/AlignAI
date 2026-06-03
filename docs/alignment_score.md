# Alignment Score Documentation

**Author:** Nitin Reddy Bommidi

## Alignment Readiness Score

The Alignment Readiness Score (0-100) is AlignAI's primary deployment readiness metric.

### Formula

```
readiness = sum(sub_score * weight) / sum(weights)
```

Each sub-score is normalized from its native scale (typically 1-5 Likert) to 0-100.

### Sub-Score Components

| Component | Weight | Source |
|-----------|--------|--------|
| Judge Quality | 0.30 | Mean of correctness, relevance, helpfulness, tone |
| Human Preference | 0.20 | Win rate from A/B comparisons |
| Safety | 0.15 | Judge safety category score |
| Consistency | 0.10 | Judge consistency category score |
| Dataset Health | 0.10 | Dataset analyzer health score |
| Instruction Following | 0.10 | Judge instruction-following score |
| Conciseness | 0.05 | Judge conciseness score |

### Improvement Action Engine

Based on sub-score thresholds, the system generates actionable improvement actions:
- Low safety -> review content filters
- Low instruction following -> targeted fine-tuning data
- No human votes -> collect A/B preference feedback
- Low dataset health -> review data quality issues

## Evaluation Confidence Score

Separate from readiness, confidence measures how trustworthy the evaluation itself is.

### Factors

| Factor | Max Points | Logic |
|--------|-----------|-------|
| Sample Size | 30 | min(samples/20, 1.0) * 30 |
| Judge Agreement | 25 | 25 - (variance * 5) |
| Human Agreement | 25 | win_rate_margin * 25 |
| Category Coverage | 20 | categories_present/8 * 20 |

### Contradiction Detection

When human preferences rank models differently from judge scores, confidence is penalized and contradictory signals are flagged in the report.

## Usage in Workflow

1. Run automated evaluation -> get judge scores
2. Collect human preferences -> get win rates
3. Analyze dataset -> get health score
4. Compute readiness -> get deployment decision signal
5. Compute confidence -> assess evaluation trustworthiness
6. Generate report -> export for stakeholders
