# lora-factory

A minimal, from-scratch library for pluggable **LoRA-family parameter-efficient
fine-tuning** (PEFT) on PyTorch and Hugging Face `transformers` models. Built as a
transparent, educational implementation — every adapter is a small class with an
explicit forward pass, so you can read it top-to-bottom.

## Features

- **Adapters:** `LoRA`, `LoRA-FA` (frozen random A), `VeRA` (frozen shared A/B + per-layer vectors), `Delta-LoRA`.
- **LoRA+ optimizer:** higher learning rate for the B-like params via parameter groups.
- **HF integration:** ready-to-run GLUE/SST-2 training with drop-in `transformers` models.

## Installation

```bash
pip install -e .
```

If you'd rather not install the package, add `src/` to `PYTHONPATH`:

```bash
export PYTHONPATH=src
```

## Quickstart (Python API)

```python
import torch
from lora_factory.config import AdapterBuildConfig, TargetSpec
from lora_factory.nn.inject import apply_adapters

model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU())
cfg = AdapterBuildConfig(
    kind="lora",              # lora | lora_fa | vera | delta_lora
    targets=[TargetSpec(module_names=["0"], rank=8, alpha=16)],
)
patched = apply_adapters(model, cfg)   # ['0']
```

## Quickstart (SST-2 CLI)

```bash
python scripts/train_sst2.py --config configs/sst2_lora.yaml
python scripts/train_sst2.py --config configs/sst2_vera.yaml
python scripts/train_sst2.py --config configs/sst2_delta_lora.yaml
```

## Adapters at a glance

| Adapter        | Trains        | Key idea |
|----------------|---------------|----------|
| `lora`         | A, B          | `ΔW = (α/r)·B·A` (standard LoRA) |
| `lora_fa`      | B             | A frozen after init (LoRA-FA) |
| `delta_lora`   | A, B          | merges `Δ(AB)` into the base weight each optimizer step |
| `vera`         | `b`, `d` vectors | frozen shared A/B + per-layer scaling vectors |

## Configuration reference

Adapters are configured via `AdapterBuildConfig` / `TargetSpec` (Python API) or
the `kind:` – `targets:` block in a YAML file (CLI). The key fields:

| Field              | Meaning                                           |
|--------------------|---------------------------------------------------|
| `kind`             | `lora` \| `lora_fa` \| `delta_lora` \| `vera`     |
| `rank`             | adapter dimension `r`                             |
| `alpha`            | effective scale is `alpha / rank`                 |
| `dropout`          | dropout probability on the adapter path           |
| `module_names`     | substrings matching target module names           |
| `module_types`     | class-name substrings to match (default `Linear`) |
| `lora_plus_lr_ratio` | `lr_B = ratio * lr_A` (LoRA+)                   |

See `configs/*.yaml` for complete, runnable examples.

## Delta-LoRA note

`Delta-LoRA` accumulates `Δ(AB)` into the base weight after each optimizer step
(see `AdapterTrainer` / `apply_post_step_updates`). The merge happens exactly
once per optimizer update regardless of gradient-accumulation settings.

## Tests

```bash
pytest -q            # CPU-only unit + tiny SST-2 smoke test
```

The smoke test downloads `prajjwal1/bert-tiny` on first run and requires
network access to the Hugging Face Hub.

## Project structure

```
src/lora_factory/
  adapters/      LoRA, LoRA-FA, VeRA, Delta-LoRA + registry
  config.py      AdapterBuildConfig, TargetSpec
  hf/            GLUE/SST-2 helpers + Trainer extension
  nn/inject.py   module match/wrap logic (Apply adapters)
  utils/         seed, optimizer (LoRA+), state, yaml parsing
```

## Extensibility

Add a new adapter by subclassing `AdapterModule` (implement `adapter_forward`)
and registering it in `registry.py` under a `kind` string:

```python
from lora_factory.adapters.base import AdapterModule

class MyAdapter(AdapterModule):
    def adapter_forward(self, x):
        return ...   # returns the delta to add on top of the base linear

# registry.py
_REGISTRY["my_adapter"] = MyAdapter
```

## License

MIT