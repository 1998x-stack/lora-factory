from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import yaml

from ..config import AdapterBuildConfig, TargetSpec


def load_adapter_build_config(path: str) -> AdapterBuildConfig:
    """Load YAML config file into AdapterBuildConfig."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    targets = []
    for t in raw.get("targets", []):
        targets.append(TargetSpec(
            module_names=t.get("module_names", TargetSpec().module_names),
            module_types=t.get("module_types", TargetSpec().module_types),
            rank=int(t.get("rank", 8)),
            alpha=float(t.get("alpha", 16)),
            dropout=float(t.get("dropout", 0.0)),
        ))

    return AdapterBuildConfig(
        kind=str(raw.get("kind", "lora")),
        lora_plus_lr_ratio=float(raw.get("lora_plus_lr_ratio", 16.0)),
        seed=int(raw.get("seed", 42)),
        targets=targets or [TargetSpec()],
    )
