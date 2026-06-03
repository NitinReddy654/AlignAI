"""Experiment registry for tracking fine-tuning jobs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from alignai.config import get_config
from alignai.logging_utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class ExperimentRecord:
    """Metadata for a single fine-tuning experiment."""

    experiment_id: str
    model_version: str
    dataset_version: str
    strategy: str
    hyperparams: Dict[str, Any]
    output_dir: str
    duration_seconds: float
    cost_estimate: Dict[str, Any]
    status: str = "completed"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    evaluation_scores: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ExperimentRegistry:
    """JSON-backed experiment store."""

    def __init__(self, registry_dir: Optional[Path] = None):
        cfg = get_config()
        self.registry_dir = registry_dir or cfg.experiments_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.registry_dir / "index.json"

    def _load_index(self) -> List[str]:
        if self.index_file.exists():
            return json.loads(self.index_file.read_text())
        return []

    def _save_index(self, ids: List[str]) -> None:
        self.index_file.write_text(json.dumps(ids, indent=2))

    def save(self, record: ExperimentRecord) -> None:
        path = self.registry_dir / f"{record.experiment_id}.json"
        path.write_text(json.dumps(record.to_dict(), indent=2))
        ids = self._load_index()
        if record.experiment_id not in ids:
            ids.append(record.experiment_id)
        self._save_index(ids)
        logger.info("Saved experiment: %s", record.experiment_id)

    def load(self, experiment_id: str) -> Optional[ExperimentRecord]:
        path = self.registry_dir / f"{experiment_id}.json"
        if not path.exists():
            return None
        return ExperimentRecord.from_dict(json.loads(path.read_text()))

    def list_all(self) -> List[ExperimentRecord]:
        records = []
        for exp_id in self._load_index():
            record = self.load(exp_id)
            if record:
                records.append(record)
        return records

    def update_evaluation_scores(self, experiment_id: str, scores: Dict[str, Any]) -> None:
        record = self.load(experiment_id)
        if record:
            record.evaluation_scores = scores
            self.save(record)

    def get_latest(self) -> Optional[ExperimentRecord]:
        records = self.list_all()
        return records[-1] if records else None
