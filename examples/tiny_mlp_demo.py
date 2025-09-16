# A tiny end-to-end demo: inject adapters, build optimizer (with LoRA+ LR ratio),
# train on a toy regression, and (if Delta-LoRA) apply post-step updates.

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from loguru import logger

from lora_factory.config import AdapterBuildConfig, TargetSpec
from lora_factory.nn.inject import apply_adapters, apply_post_step_updates
from lora_factory.utils.optim import build_optimizer_with_lora_plus
from lora_factory.utils.seed import set_seed


class TinyMLP(nn.Module):
    def __init__(self, d: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def toy_data(n: int = 1024, d: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(n, d, generator=gen)
    w = torch.randn(d, 1, generator=gen) / math.sqrt(d)
    y = x @ w + 0.1 * torch.randn(n, 1, generator=gen)
    return x, y


def main() -> None:
    set_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyMLP().to(device)

    # Choose your variant: "lora", "lora_fa", "vera", "delta_lora"
    cfg = AdapterBuildConfig(
        kind="lora",
        lora_plus_lr_ratio=16.0,  # LoRA+
        targets=[TargetSpec(module_names=["net.0", "net.2"], rank=8, alpha=16, dropout=0.0)],
    )
    patched = apply_adapters(model, cfg)
    logger.info(f"Patched modules: {patched}")

    opt = build_optimizer_with_lora_plus(model, base_lr=1e-3, lora_plus_lr_ratio=cfg.lora_plus_lr_ratio)
    loss_fn = nn.MSELoss()

    x, y = toy_data(n=2048, d=128)
    x, y = x.to(device), y.to(device)

    model.train()
    for step in range(200):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        # If using Delta-LoRA, accumulate Δ(AB) into base weight after step
        apply_post_step_updates(model)
        if (step + 1) % 20 == 0:
            logger.info(f"step={step+1} loss={loss.item():.6f}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
