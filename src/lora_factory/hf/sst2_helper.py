"""Load GLUE/SST-2 and build tokenizer/collator/metrics."""
from __future__ import annotations

from typing import Dict, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import evaluate


def load_sst2_and_tokenizer(model_name: str, max_length: int = 128):
    ds = load_dataset("glue", "sst2")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def _tok(batch: Dict[str, str]) -> Dict[str, list]:
        return tok(batch["sentence"], truncation=True, max_length=max_length)

    ds = ds.map(_tok, batched=True, remove_columns=["sentence", "idx"])
    collator = DataCollatorWithPadding(tok)
    return ds, tok, collator


def build_compute_metrics():
    metric = evaluate.load("glue", "sst2")
    def compute_metrics(eval_pred: Tuple):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return metric.compute(predictions=preds, references=labels)
    return compute_metrics
