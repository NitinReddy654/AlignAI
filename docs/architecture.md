# AlignAI Architecture

**Author:** Nitin Reddy Bommidi

## Overview

AlignAI follows a **Prepare -> Retrieve -> Train -> Evaluate -> Decide** architecture for generative AI enablement. The system separates dataset management, RAG grounding, distributed model training, evaluation, alignment scoring, human review, code-generation assessment, and reporting into clear modules backed by portable JSON and SQL artifacts.

This keeps the platform lightweight enough to run locally while still reflecting the workflow AI teams use to compare model variants and approve deployment candidates.

## System Layers

```text
+---------------------------------------------------------+
|                   Streamlit Platform                    |
| Datasets | Experiments | Training | Evaluation | Reports  |
+---------------------------+-----------------------------+
                            |
+---------------------------v-----------------------------+
|                    CLI Entrypoints                      |
|  run_finetune.py | run_evaluation.py | analyze_dataset  |
+---------------------------+-----------------------------+
                            |
+---------------------------v-----------------------------+
|                   src/alignai Package                   |
| training | rag | models | evaluation | code alignment |
+---------------------------+-----------------------------+
                            |
+---------------------------v-----------------------------+
|          JSON + SQL Artifact Store (data/artifacts/)    |
| experiments | evaluations | reports | preferences | SQLite |
+---------------------------------------------------------+
```

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | Environment configuration with validation |
| `hardware.py` | GPU, MPS, and CPU detection with dtype selection |
| `models/` | Base, full fine-tuned, LoRA, and QLoRA loading plus generation |
| `training/` | Dataset formatting, strategy configs, TRL SFTTrainer orchestration, DDP/FSDP helpers |
| `rag/` | ChromaDB retrieval, embeddings, semantic search, prompt augmentation, faithfulness scoring |
| `evaluation/` | LLM-as-a-judge scoring, metrics, confidence, and reports |
| `alignment/` | Readiness scoring and normalized deployment signals |
| `experiments/` | JSON registry, leaderboard, deployment candidate ranking, code-generation quality scoring |
| `preference/` | Human A/B comparison capture and analytics |
| `datasets_analysis/` | Dataset health reporting and quality checks |

## Data Flow

1. **Dataset analysis:** JSONL conversations are inspected for size, role balance, duplicates, token distribution, and quality issues.
2. **SQL exploration:** SQLite schema and notebook support dataset, judge-score, evaluation-run, and preference queries.
3. **RAG grounding:** ChromaDB retrieval injects relevant enterprise context into evaluation prompts.
4. **Training:** Full fine-tuning, LoRA, or QLoRA creates checkpoints and experiment metadata, with DDP/FSDP hooks for multi-GPU runs.
5. **Generation:** Candidate models produce evaluation responses with latency and token metrics.
6. **Automated judging:** The LLM-as-a-judge rubric scores response quality, safety, and alignment categories.
7. **Code alignment:** Programming LLM outputs can be scored for correctness, security, readability, and efficiency.
8. **Human review:** Blind A/B comparisons capture human preference signals.
9. **Readiness scoring:** Alignment and confidence engines convert raw evidence into deployment signals.
10. **Reporting:** Leaderboards and JSON reports expose the evidence needed for model selection.

## Design Principles

- **End-to-end GenAI workflow:** The platform covers dataset readiness, RAG grounding, model adaptation, code evaluation, review, and deployment decision support.
- **Artifact-driven reproducibility:** JSON artifacts preserve experiments, evaluations, reports, and preference data without requiring a database.
- **Hardware-adaptive execution:** CUDA, MPS, CPU, DDP, DataParallel, and FSDP paths keep training and generation portable.
- **Evaluation before promotion:** Automated judging, human preferences, safety signals, and confidence scoring are first-class release gates.
- **Fast local validation:** Pure scoring and reporting modules avoid heavy ML imports, keeping CI and unit tests lightweight.
