"""Extract and load adapter-only state dicts."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


ADAPTER_KEYWORDS = (
    "lora_a.weight",
    "lora_b.weight",
    "vera_b",
    "vera_d",
    "A_prev",  # delta-lora shadows (optional)
    "B_prev",
)


def extract_adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a state_dict with **only** adapter parameters/buffers."""
    sd = model.state_dict()
    subset = {k: v for k, v in sd.items() if any(k.lower().endswith(suf) for suf in ADAPTER_KEYWORDS)}
    return subset


def load_adapter_state_dict(model: nn.Module, adapter_sd: Dict[str, torch.Tensor], strict: bool = False) -> None:
    """Load adapter-only state dict into model."""
    model.load_state_dict(adapter_sd, strict=False if not strict else True)
