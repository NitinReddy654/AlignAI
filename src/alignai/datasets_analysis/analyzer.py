"""Dataset analysis and health reporting."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from alignai.logging_utils import setup_logging
from alignai.training.data import format_conversation, load_jsonl_dataset

logger = setup_logging(__name__)


@dataclass
class DatasetHealthReport:
    """Comprehensive dataset health analysis."""

    dataset_name: str
    total_records: int
    avg_conversation_length: float
    avg_turns_per_conversation: float
    token_distribution: Dict[str, Any] = field(default_factory=dict)
    role_distribution: Dict[str, int] = field(default_factory=dict)
    quality_issues: List[str] = field(default_factory=list)
    duplicate_count: int = 0
    health_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "total_records": self.total_records,
            "avg_conversation_length": round(self.avg_conversation_length, 1),
            "avg_turns_per_conversation": round(self.avg_turns_per_conversation, 1),
            "token_distribution": self.token_distribution,
            "role_distribution": self.role_distribution,
            "quality_issues": self.quality_issues,
            "duplicate_count": self.duplicate_count,
            "health_score": round(self.health_score, 1),
        }


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (words * 1.3)."""
    return max(1, int(len(text.split()) * 1.3))


def analyze_dataset(path: str | Path) -> DatasetHealthReport:
    """Analyze a JSONL dataset and produce a health report."""
    path = Path(path)
    records = load_jsonl_dataset(path)
    dataset_name = path.stem

    if not records:
        return DatasetHealthReport(
            dataset_name=dataset_name,
            total_records=0,
            avg_conversation_length=0,
            avg_turns_per_conversation=0,
            quality_issues=["Dataset is empty."],
            health_score=0,
        )

    char_lengths: List[int] = []
    turn_counts: List[int] = []
    role_counter: Counter = Counter()
    quality_issues: List[str] = []
    content_hashes: set = set()
    duplicate_count = 0

    for i, record in enumerate(records):
        messages = format_conversation(record)
        turn_counts.append(len(messages))
        total_chars = sum(len(m.get("content", "")) for m in messages)
        char_lengths.append(total_chars)

        for msg in messages:
            role_counter[msg.get("role", "unknown")] += 1
            content = msg.get("content", "")
            if not content.strip():
                quality_issues.append(f"Record {i}: empty message content.")
            if len(content) > 10000:
                quality_issues.append(f"Record {i}: excessively long message ({len(content)} chars).")

        content_key = json.dumps(messages, sort_keys=True)
        if content_key in content_hashes:
            duplicate_count += 1
        content_hashes.add(content_key)

    token_estimates = [_estimate_tokens(json.dumps(format_conversation(r))) for r in records]
    avg_tokens = sum(token_estimates) / len(token_estimates)

    health_score = 100.0
    health_score -= min(duplicate_count * 2, 20)
    health_score -= min(len(quality_issues) * 3, 30)
    if avg_tokens < 50:
        health_score -= 10
        quality_issues.append("Average conversation length is very short.")
    if avg_tokens > 3000:
        health_score -= 10
        quality_issues.append("Average conversation length is very long.")
    health_score = max(0, health_score)

    return DatasetHealthReport(
        dataset_name=dataset_name,
        total_records=len(records),
        avg_conversation_length=sum(char_lengths) / len(char_lengths),
        avg_turns_per_conversation=sum(turn_counts) / len(turn_counts),
        token_distribution={
            "avg_tokens": round(avg_tokens, 1),
            "min_tokens": min(token_estimates),
            "max_tokens": max(token_estimates),
            "total_estimated_tokens": sum(token_estimates),
        },
        role_distribution=dict(role_counter),
        quality_issues=quality_issues[:20],
        duplicate_count=duplicate_count,
        health_score=health_score,
    )


def save_health_report(report: DatasetHealthReport, output_dir: Optional[Path] = None) -> Path:
    """Save dataset health report to JSON."""
    from alignai.config import get_config

    cfg = get_config()
    out_dir = output_dir or cfg.artifacts_dir / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report.dataset_name}_health.json"
    path.write_text(json.dumps(report.to_dict(), indent=2))
    logger.info("Saved health report: %s", path)
    return path
