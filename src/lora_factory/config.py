from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TargetSpec:
    """Which modules to adapt.

    Attributes:
        module_names: Name substrings to match (e.g., ["q_proj","k_proj","v_proj","out_proj","dense","query","key","value","q_lin"...]).
        module_types: Class names to match (default ["Linear"]).
        rank: LoRA rank r.
        alpha: Scaling α (effective scale is α/r).
        dropout: Dropout probability for the adapter path.
    """
    module_names: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"])
    module_types: List[str] = field(default_factory=lambda: ["Linear"])
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0


@dataclass
class AdapterBuildConfig:
    """Top-level build config for the factory.

    Attributes:
        kind: One of ["lora", "lora_fa", "vera", "delta_lora"].
        lora_plus_lr_ratio: For LoRA+, set lr_B = ratio * lr_A via param groups.
        seed: RNG seed for reproducibility / VeRA shared matrices.
        targets: TargetSpec list.
    """
    kind: str = "lora"
    lora_plus_lr_ratio: float = 16.0
    seed: int = 42
    targets: List[TargetSpec] = field(default_factory=lambda: [TargetSpec()])
