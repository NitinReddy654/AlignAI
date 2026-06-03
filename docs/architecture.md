# AlignAI Architecture

**Author:** Nitin Reddy Bommidi

## Overview

AlignAI follows a **Prepare -> Train -> Evaluate -> Decide** architecture for generative AI enablement. The system separates model training, evaluation, alignment scoring, human review, and reporting into clear modules backed by portable JSON artifacts.

This keeps the platform lightweight enough to run locally while still reflecting the workflow AI teams use to compare model variants and approve deployment candidates.

## System Layers

```text
+---------------------------------------------------------+
|                   Streamlit Platform                    |
|  Datasets | Experiments | Training | Evaluation | Reports |
+---------------------------+-----------------------------+
                            |
+---------------------------v-----------------------------+
|                    CLI Entrypoints                      |
|  run_finetune.py | run_evaluation.py | analyze_dataset  |
+---------------------------+-----------------------------+
                            |
+---------------------------v-----------------------------+
|                   src/alignai Package                   |
|  training | models | evaluation | alignment | preference |
+---------------------------+-----------------------------+
                            |
+---------------------------v-----------------------------+
|              JSON Artifact Store (data/artifacts/)      |
|  experiments | evaluations | reports | preferences | data |
+---------------------------------------------------------+
```

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | Environment configuration with validation |
| `hardware.py` | GPU, MPS, and CPU detection with dtype selection |
| `models/` | Base, full fine-tuned, LoRA, and QLoRA loading plus generation |
| `training/` | Dataset formatting, strategy configs, and TRL SFTTrainer orchestration |
| `evaluation/` | LLM-as-a-judge scoring, metrics, confidence, and reports |
| `alignment/` | Readiness scoring and normalized deployment signals |
| `experiments/` | JSON registry, leaderboard, and deployment candidate ranking |
| `preference/` | Human A/B comparison capture and analytics |
| `datasets_analysis/` | Dataset health reporting and quality checks |

## Data Flow

1. **Dataset analysis:** JSONL conversations are inspected for size, role balance, duplicates, token distribution, and quality issues.
2. **Training:** Full fine-tuning, LoRA, or QLoRA creates checkpoints and experiment metadata.
3. **Generation:** Candidate models produce evaluation responses with latency and token metrics.
4. **Automated judging:** The LLM-as-a-judge rubric scores response quality, safety, and alignment categories.
5. **Human review:** Blind A/B comparisons capture human preference signals.
6. **Readiness scoring:** Alignment and confidence engines convert raw evidence into deployment signals.
7. **Reporting:** Leaderboards and JSON reports expose the evidence needed for model selection.

## Design Principles

- **End-to-end GenAI workflow:** The platform covers dataset readiness, model adaptation, evaluation, review, and deployment decision support.
- **Artifact-driven reproducibility:** JSON artifacts preserve experiments, evaluations, reports, and preference data without requiring a database.
- **Hardware-adaptive execution:** CUDA, MPS, and CPU detection keeps training and generation paths portable.
- **Evaluation before promotion:** Automated judging, human preferences, safety signals, and confidence scoring are first-class release gates.
- **Fast local validation:** Pure scoring and reporting modules avoid heavy ML imports, keeping CI and unit tests lightweight.
