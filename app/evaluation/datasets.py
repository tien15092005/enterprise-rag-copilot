from __future__ import annotations

import hashlib
from pathlib import Path

from app.evaluation.models import EvaluationDataset


def load_dataset(path: Path) -> EvaluationDataset:
    return EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
