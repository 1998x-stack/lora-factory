"""Transformers Trainer extension that applies post-step adapter updates."""
from __future__ import annotations

from transformers import Trainer

from ..nn.inject import apply_post_step_updates


class AdapterTrainer(Trainer):
    """Trainer that applies post-step updates (for Delta-LoRA) automatically.

    HF invokes ``optimizer_step`` once per gradient-accumulation cycle, so the
    Delta-LoRA merge runs exactly once per optimizer update.
    """

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        apply_post_step_updates(self.model)
