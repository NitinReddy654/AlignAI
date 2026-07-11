# Deployment Guide

**Author:** Nitin Reddy Bommidi

## Local Runtime

```bash
git clone <repo-url>
cd alignai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
cp .env.example .env
streamlit run app/Home.py
```

Set `OPENAI_API_KEY` in `.env` for judge-based evaluation.

## Docker Deployment

```bash
docker build -t alignai .
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  alignai
```

Access at `http://localhost:8501`.

## GPU Training

Fine-tuning runs best on a CUDA-capable GPU. LoRA on 3B models should be planned around 16GB+ VRAM.

```bash
# NVIDIA CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Apple Silicon
pip install torch

python scripts/run_finetune.py \
  --strategy lora \
  --dataset data/samples/enterprise_support_dataset.jsonl
```

## Cloud Deployment Options

### Hugging Face Spaces
Deploy the Streamlit application to HF Spaces with GPU-backed inference and evaluation workflows.

### AWS/GCP
Use the container image with GPU instances such as AWS g4dn or comparable GCP accelerator hosts for training jobs.

## Environment Variables

See `.env.example` for full configuration. Required:
- `OPENAI_API_KEY` - LLM-as-a-judge evaluation

Optional:
- `ALIGNAI_GPU_HOUR_COST` - cost estimation rate
- `HF_TOKEN` - private model access

## RAG and Vector Store Runtime

RAG features use ChromaDB for vector retrieval. Local development can use the built-in deterministic embedding fallback for offline tests. Production retrieval can use `OpenAIEmbeddingFunction` with `OPENAI_API_KEY`.

```python
from alignai.rag import ChromaContextRetriever, OpenAIEmbeddingFunction

retriever = ChromaContextRetriever(
    persist_directory="data/artifacts/chroma",
    embedding_function=OpenAIEmbeddingFunction(),
)
```

## Distributed Training Runtime

Distributed fine-tuning helpers support PyTorch DDP, DataParallel fallback, and FSDP wrapping. Launch multi-process training with `torchrun` and use the runtime environment variables that PyTorch sets:

```bash
torchrun --nproc_per_node=4 scripts/run_finetune.py \
  --strategy lora \
  --dataset data/samples/enterprise_support_dataset.jsonl
```

## Release Gates

Before a model variant is promoted, AlignAI expects:

- Secure OpenAI and Hugging Face credentials
- GPU capacity matched to the selected training strategy
- Vector store persistence configured for RAG-grounded evaluation
- Durable artifact storage for checkpoints and reports
- Completed dataset analysis, automated evaluation, and human preference review
- Alignment Readiness Score at or above the staged-deployment threshold
