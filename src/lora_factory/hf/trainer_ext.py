from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from transformers import Trainer

from ..nn.inject import apply_post_step_updates


class AdapterTrainer(Trainer):
    """Trainer that applies post-step updates (for Delta-LoRA) automatically."""

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        loss = super().training_step(model, inputs)
        # When gradient_accumulation > 1, post-step happens in optimizer_step; do it there.
        return loss

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        apply_post_step_updates(self.model)
