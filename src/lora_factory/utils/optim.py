from __future__ import annotations

from typing import List, Tuple

import torch


def split_lora_params_for_lora_plus(model: torch.nn.Module) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Return (non-LoRA, LoRA-A, LoRA-B-like) params by naming convention."""
    base, a_list, b_list = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if lname.endswith("lora_a.weight"):
            a_list.append(p)
        elif lname.endswith("lora_b.weight") or lname.endswith("vera_b") or lname.endswith("vera_d"):
            b_list.append(p)
        else:
            base.append(p)
    return base, a_list, b_list


def build_optimizer_with_lora_plus(
    model: torch.nn.Module,
    base_lr: float = 1e-4,
    lora_plus_lr_ratio: float = 16.0,
    weight_decay: float = 0.0,
    betas=(0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    base_params, a_params, b_params = split_lora_params_for_lora_plus(model)
    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr, "weight_decay": weight_decay})
    if a_params:
        groups.append({"params": a_params, "lr": base_lr, "weight_decay": weight_decay})
    if b_params:
        groups.append({"params": b_params, "lr": base_lr * lora_plus_lr_ratio, "weight_decay": weight_decay})
    return torch.optim.AdamW(groups, betas=betas, eps=eps)
