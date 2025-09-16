#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from lora_factory.utils.yaml_cfg import load_adapter_build_config
from lora_factory.utils.seed import set_seed
from lora_factory.utils.optim import build_optimizer_with_lora_plus
from lora_factory.nn.inject import apply_adapters
from lora_factory.hf.sst2_helper import load_sst2_and_tokenizer, build_compute_metrics
from lora_factory.hf.trainer_ext import AdapterTrainer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config (sst2_*.yaml)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    # Load YAML (training + adapter fields live in one file)
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    adapter_cfg = load_adapter_build_config(str(cfg_path))
    set_seed(int(train_cfg.get("seed", 42)))

    model_name = train_cfg["model_name"]
    output_dir = train_cfg["output_dir"]
    max_length = int(train_cfg.get("max_length", 128))
    fp16 = bool(train_cfg.get("fp16", False))

    logger.info(f"Loading dataset/tokenizer for SST-2 with model={model_name}")
    ds, tok, collator = load_sst2_and_tokenizer(model_name, max_length=max_length)

    logger.info("Loading HF model")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    logger.info("Applying adapters")
    patched = apply_adapters(model, adapter_cfg)
    logger.info(f"Patched {len(patched)} modules:\n{patched}")

    # Optimizer (LoRA+ param groups)
    base_lr = float(train_cfg.get("learning_rate", 5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    optimizer = build_optimizer_with_lora_plus(
        model,
        base_lr=base_lr,
        lora_plus_lr_ratio=float(adapter_cfg.lora_plus_lr_ratio),
        weight_decay=weight_decay,
    )

    # TrainingArguments
    targs = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 32)),
        learning_rate=base_lr,  # not used since we pass optimizer; still recorded
        weight_decay=weight_decay,
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.06)),
        evaluation_strategy=train_cfg.get("evaluation_strategy", "steps"),
        eval_steps=int(train_cfg.get("eval_steps", 200)),
        save_strategy=train_cfg.get("save_strategy", "no"),
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        seed=int(train_cfg.get("seed", 42)),
        fp16=fp16,
        report_to=[],
        load_best_model_at_end=False,
    )

    # Scheduler linked to Trainer's total steps
    total_train_bs = targs.per_device_train_batch_size * max(1, targs._n_gpu)
    train_data_size = len(ds["train"])
    steps_per_epoch = train_data_size // total_train_bs
    num_update_steps = int(steps_per_epoch * targs.num_train_epochs)
    num_warmup_steps = int(num_update_steps * targs.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_update_steps)

    trainer = AdapterTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=build_compute_metrics(),
        optimizers=(optimizer, scheduler),
    )

    logger.info("Starting training…")
    trainer.train()
    logger.info("Evaluating…")
    eval_out = trainer.evaluate()
    logger.info(f"Eval: {eval_out}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
