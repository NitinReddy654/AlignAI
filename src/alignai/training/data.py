"""Dataset loading and formatting for fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from alignai.logging_utils import setup_logging

logger = setup_logging(__name__)


def load_jsonl_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """Load conversations from JSONL file."""
    records: List[Dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def load_hf_dataset(dataset_id: str, split: str = "train") -> Any:
    """Load dataset from Hugging Face Hub."""
    from datasets import load_dataset

    logger.info("Loading HF dataset: %s split=%s", dataset_id, split)
    ds = load_dataset(dataset_id)
    return ds[split] if split in ds else ds["train"]


def format_conversation(record: Dict[str, Any], conversation_key: str = "conversation") -> List[dict]:
    """Extract conversation messages from a record."""
    if conversation_key in record:
        return record[conversation_key]
    if "messages" in record:
        return record["messages"]
    return [{"role": "user", "content": record.get("prompt", "")}]


def build_sft_dataset(records: List[Dict[str, Any]], tokenizer: Any, max_length: int = 2048) -> Any:
    """Format records into tokenized SFT dataset using chat template."""
    from datasets import Dataset

    def _format_example(example: dict) -> dict:
        messages = format_conversation(example)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    ds = Dataset.from_list(records)
    return ds.map(_format_example, remove_columns=ds.column_names)
