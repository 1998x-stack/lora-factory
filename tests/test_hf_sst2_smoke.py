from __future__ import annotations

import os

import torch
import pytest
from transformers import AutoModelForSequenceClassification, TrainingArguments, get_linear_schedule_with_warmup

from lora_factory.utils.seed import set_seed
from lora_factory.config import AdapterBuildConfig, TargetSpec
from lora_factory.nn.inject import apply_adapters
from lora_factory.utils.optim import build_optimizer_with_lora_plus
from lora_factory.hf.sst2_helper import load_sst2_and_tokenizer, build_compute_metrics
from lora_factory.hf.trainer_ext import AdapterTrainer


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sst2_tiny_one_epoch(tmp_path):
    set_seed(7)
    model_name = "prajjwal1/bert-tiny"
    ds, tok, collator = load_sst2_and_tokenizer(model_name, max_length=64)
    # use a very small subset for speed
    ds_small = {
        "train": ds["train"].select(range(64)),
        "validation": ds["validation"].select(range(64)),
    }

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    cfg = AdapterBuildConfig(
        kind="lora",
        lora_plus_lr_ratio=8.0,
        targets=[TargetSpec(module_names=["query","key","value","dense","q_lin","k_lin","v_lin","out_lin"], rank=4, alpha=8)]
    )
    apply_adapters(model, cfg)

    bs = 8
    lr = 5e-4
    weight_decay = 0.0
    args = TrainingArguments(
        output_dir=str(tmp_path),
        num_train_epochs=1,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        learning_rate=lr,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
        fp16=False,
    )

    opt = build_optimizer_with_lora_plus(model, base_lr=lr, lora_plus_lr_ratio=cfg.lora_plus_lr_ratio, weight_decay=weight_decay)
    steps = len(ds_small["train"]) // bs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=steps)

    trainer = AdapterTrainer(
        model=model,
        args=args,
        train_dataset=ds_small["train"],
        eval_dataset=ds_small["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=build_compute_metrics(),
        optimizers=(opt, sched),
    )

    out = trainer.train()
    # basic assertion: training ran and returned a log history
    assert out is not None
