# lora-factory Professionalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make lora-factory actually run (fix the missing `utils/seed.py`), harden adapter/injection correctness, and modernize its docs to reference quality — without restructuring the package.

**Architecture:** A surgical pass over an existing `src/lora_factory` package: add the one missing `utils/seed.py`, tighten one error path, add regression tests that lock in the intended frozen-parameter hygiene and the LoRA+ optimizer split, verify the full test suite + demo in a fresh venv, then rewrite README/configs/docstrings.

**Tech Stack:** Python 3.12, PyTorch 2.x (CPU), transformers, datasets, evaluate, loguru, pyyaml, pytest.

## Global Constraints

- Package layout and all existing names/semantics must remain unchanged (no refactor).
- Venv: `.venv/` at repo root already created. Run tests/examples via `.venv/bin/python` and `.venv/bin/python -m pytest`. If a dependency errors on import, run `.venv/bin/python -m pip install -e ".[test]"` first.
- Required fix: `lora_factory.utils.seed` must exist and expose `set_seed(seed: int) -> None`.
- Educational repo: favor simplicity and clarity over optimization. Leave VeRA's per-forward `.to()` cache behavior as-is (approved decision).
- TDD: write the failing test first, then the implementation, then commit.
- Every task ends with a green test run and a commit.

---

### Task 1: Reproducibility module `utils/seed.py`

**Files:**
- Create: `src/lora_factory/utils/seed.py`
- Test: `tests/test_utils_seed.py`
- Depends on: venv with `torch` installed (already provisioned).

**Interfaces:**
- Produces: `lora_factory.utils.seed.set_seed(seed: int) -> None`. Callers (`inject.py`, `scripts/train_sst2.py`, examples) import `from lora_factory.utils.seed import set_seed`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_utils.py`:

```python
from __future__ import annotations

import random

import torch

from lora_factory.utils.seed import set_seed


def test_set_seed_reproducible():
    set_seed(123)
    a = torch.randn(4)
    set_seed(123)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_set_seed_also_seeds_python_and_numpy():
    import numpy as np

    set_seed(7)
    A = (random.random(), np.random.rand(2).tolist())
    set_seed(7)
    B = (random.random(), np.random.rand(2).tolist())
    assert A == B
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lora_factory.utils.seed'`

- [ ] **Step 3: Write the implementation**

Create `src/lora_factory/utils/seed.py`:

```python
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all RNGs (Python, NumPy, PyTorch CPU/CUDA) for reproducibility.

    Also enables deterministic cuDNN. Safe to call repeatedly; only affects
    the current process's random state.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_utils.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Sanity-check the package imports end-to-end**

Run: `.venv/bin/python -c "import lora_factory; print('package import OK')"`
Expected: prints `package import OK` (this failed before the fix).

- [ ] **Step 6: Commit**

```bash
git add src/lora_factory/utils/seed.py tests/test_utils.py
git commit -m "feat: add utils/seed.py for reproducible RNG seeding"
```

---

### Task 2: Harden module-path resolution error

**Files:**
- Modify: `src/lora_factory/nn/inject.py:104-108` (the `_find_parent_with_attr` function)
- Test: `tests/test_inject_robustness.py`

**Interfaces:**
- Consumes: `_find_parent_with_attr(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]`.
- Produces: unchanged signature, but now raises `ValueError` (with the offending dotted name) when any path segment (including the final attribute) is unresolvable, instead of a bare `AttributeError`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_inject_robustness.py`:

```python
from __future__ import annotations

import re

import pytest
import torch
import torch.nn as nn

from lora_factory.nn.inject import _find_parent_with_attr


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 8), nn.ReLU())


def test_find_parent_with_attr_missing_module_raises_clear_error():
    m = Toy()
    with pytest.raises(ValueError, match=re.escape("does_not_exist")):
        _find_parent_with_attr(m, "does_not_exist.foo")


def test_find_parent_with_attr_missing_leaf_raises_clear_error():
    m = Toy()
    with pytest.raises(ValueError, match=re.escape("nope")):
        _find_parent_with_attr(m, "block.nope")


def test_find_parent_with_attr_valid_name():
    m = Toy()
    parent, attr = _find_parent_with_attr(m, "block.0")
    assert parent is m.block and attr == "0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_inject_robustness.py -v`
Expected: the first two FAIL with `AttributeError` (wrong exception type).

- [ ] **Step 3: Write the implementation**

Replace the body of `_find_parent_with_attr` in `src/lora_factory/nn/inject.py`:

```python
def _find_parent_with_attr(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    if not parts or not all(parts):
        raise ValueError(f"Invalid module name: {dotted_name!r}")
    parent = root
    for p in parts[:-1]:
        if not hasattr(parent, p):
            raise ValueError(
                f"Cannot resolve module name {dotted_name!r}: no attribute {p!r} on {type(parent).__name__}"
            )
        parent = getattr(parent, p)
    attr = parts[-1]
    if not hasattr(parent, attr):
        raise ValueError(
            f"Cannot resolve module name {dotted_name!r}: no attribute {attr!r} on {type(parent).__name__}"
        )
    return parent, attr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_inject_robustness.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lora_factory/nn/inject.py tests/test_inject_robustness.py
git commit -m "feat: raise clear error when resolving missing module path"
```

---

### Task 3: Lock in frozen-param hygiene, no double-wrap, and LoRA+ split

**Files:**
- Test: `tests/test_frozen_hygiene.py`
- No source changes expected (these lock in existing correct behavior; if a test reveals a real bug, fix the offending source in that task).

**Interfaces:**
- Consumes: `AdapterBuildConfig`, `TargetSpec`, `apply_adapters`, `split_lora_params_for_lora_plus`, `build_optimizer_with_lora_plus`, `LoRA_FALinear`, `VeRALinear`, `AdapterModule`.

- [ ] **Step 1: Write the tests**

Create `tests/test_frozen_hygiene.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn

from lora_factory.adapters.base import AdapterModule
from lora_factory.adapters.lora_fa import LoRA_FALinear
from lora_factory.adapters.vera import VeRALinear
from lora_factory.config import AdapterBuildConfig, TargetSpec
from lora_factory.nn.inject import apply_adapters
from lora_factory.utils.optim import build_optimizer_with_lora_plus


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.block = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x):
        return self.block(x)


def _cfg(kind: str) -> AdapterBuildConfig:
    return AdapterBuildConfig(
        kind=kind,
        targets=[TargetSpec(module_names=["block.0", "block.2"], rank=4, alpha=8)],
    )


def test_wrapped_base_weights_frozen_for_all_kinds():
    for kind in ("lora", "lora_fa", "vera", "delta_lora"):
        m = Toy()
        apply_adapters(m, _cfg(kind))
        for name, sub in m.named_modules():
            if isinstance(sub, AdapterModule):
                for p in sub.wrapped.parameters():
                    assert not p.requires_grad, (kind, name)


def test_lora_fa_freezes_a_only():
    m = Toy()
    apply_adapters(m, _cfg("lora_fa"))
    for idx in (0, 2):
        sub = m.block[idx]
        assert isinstance(sub, LoRA_FALinear)
        assert not sub.lora_A.weight.requires_grad
        assert sub.lora_B.weight.requires_grad


def test_vera_trainable_params_are_only_band_vectors():
    m = Toy()
    apply_adapters(m, _cfg("vera"))
    for idx in (0, 2):
        sub = m.block[idx]
        assert isinstance(sub, VeRALinear)
        trainable = {n for n, p in sub.named_parameters() if p.requires_grad}
        assert trainable <= {"vera_b", "vera_d"}, trainable


def test_apply_adapters_twice_does_not_double_wrap():
    m = Toy()
    first = apply_adapters(m, _cfg("lora"))
    second = apply_adapters(m, _cfg("lora"))
    assert first == second
    n_adapters = sum(1 for sub in m.modules() if isinstance(sub, AdapterModule))
    assert n_adapters == 2


def test_optimizer_lora_plus_split_excludes_frozen_and_ratio():
    m = Toy()
    apply_adapters(m, _cfg("lora"))
    opt = build_optimizer_with_lora_plus(m, base_lr=1e-4, lora_plus_lr_ratio=16.0)
    high = 1e-4 * 16.0
    assert high in {g["lr"] for g in opt.param_groups}
    seen = [p for g in opt.param_groups for p in g["params"]]
    assert all(p.requires_grad for p in seen)
    # frozen wrapped base weights must not appear in any optimizer group
    for idx in (0, 2):
        for p in m.block[idx].wrapped.parameters():
            assert p not in seen
```

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_frozen_hygiene.py -v`
Expected: PASS (5 passed). If any fail, the existing source has a genuine bug — fix the source in-place in this task (e.g. a param that should be frozen but is not), then re-run until green.

- [ ] **Step 3: Commit**

```bash
git add tests/test_frozen_hygiene.py
git commit -m "test: lock in frozen-param hygiene, no double-wrap, LoRA+ split"
```
(Merge any source fix into this commit if a bug was found.)

---

### Task 4: End-to-end verification

**Files:**
- Run: `tests/test_build_and_inject.py`, `tests/test_hf_sst2_smoke.py`, `examples/tiny_mlp_demo.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3 plus the pre-existing HF integration (`AdapterTrainer`, `load_sst2_and_tokenizer`, `build_compute_metrics`).

- [ ] **Step 1: Run the non-HF unit test**

Run: `.venv/bin/python -m pytest tests/test_build_and_inject.py -v`
Expected: PASS.

- [ ] **Step 2: Run the offline demo end-to-end**

Run: `.venv/bin/python examples/tiny_mlp_demo.py 2>&1 | tail -5`
Expected: prints `Done.` and decreasing loss lines. Also run once with `kind = "delta_lora"` by temporarily changing `examples/tiny_mlp_demo.py`'s `AdapterBuildConfig(kind=...)` to `"delta_lora"`, confirm it reaches `Done.`, then revert to `"lora"`.

- [ ] **Step 3: Confirm Delta-LoRA trainer timing (analysis, no code change)**

In `AdapterTrainer.optimizer_step`, HF calls `optimizer_step` exactly once per gradient-accumulation cycle, so `apply_post_step_updates(self.model)` fires once per accumulated update — correct. Document this invariant in the `AdapterTrainer` docstring (one sentence) in `src/lora_factory/hf/trainer_ext.py`:

```python
    """Trainer that applies post-step updates (for Delta-LoRA) automatically.

    HF invokes ``optimizer_step`` once per gradient-accumulation cycle, so the
    Delta-LoRA merge runs exactly once per optimizer update.
    """
```

- [ ] **Step 4: Run the HF smoke test**

Run: `.venv/bin/python -m pytest tests/test_hf_sst2_smoke.py -v`
Expected: PASS (downloads `prajjwal1/bert-tiny` on first run). If network to HF Hub is blocked, note it and skip with `@pytest.mark.skip`.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lora_factory/hf/trainer_ext.py
git commit -m "docs: clarify Delta-LoRA post-step timing in trainer docstring"
```

---

### Task 5: Professional documentation

**Files:**
- Rewrite: `README.md`
- Modify: `configs/sst2_lora.yaml`, `configs/sst2_vera.yaml`, `configs/sst2_delta_lora.yaml` (add short reference comments only)
- Modify: `src/lora_factory/**/*.py` (module-level docstrings where missing/inconsistent)

**Interfaces:**
- Consumes: the final public API produced by prior tasks and the existing modules.

- [ ] **Step 1: Rewrite `README.md`**

Replace the current README with the following:

```markdown
# lora-factory

A minimal, from-scratch library for pluggable **LoRA-family parameter-efficient
fine-tuning** (PEFT) on PyTorch and Hugging Face `transformers` models. Built as a
transparent, educational implementation — every adapter is a small class with an
explicit forward pass, so you can read it top-to-bottom.

## Features

- **Adapters:** `LoRA`, `LoRA-FA` (frozen random A), `VeRA` (frozen shared A/B + per-layer vectors), `Delta-LoRA`.
- **LoRA+ optimizer:** higher learning rate for the B-like params via parameter groups.
- **HF integration:** ready-to-run GLUE/SST-2 training with drop-in `transformer` models.

## Installation

```bash
pip install -e .
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
patched = apply_adapters(model, cfg)
```

## Quickstart (SST-2 CLI)

```bash
python scripts/train_sst2.py --config configs/sst2_lora.yaml
python scripts/train_sst2.py --config configs/sst2_vera.yaml
python scripts/train_sst2.py --config configs/sst2_delta_lora.yaml
```

## Adapters at a glance

| Adapter | Trains | Key idea |
|---|---|---|
| `lora` | A, B | `ΔW = β·A` (standard LoRA) |
| `lora_fa` | B | A frozen after init (LoRA-FA) |
| `vera` | `b`, `d` vectors | frozen shared A/B + per-layer scaling vectors |
| `delta_lora` | A, B | merges `Δ(AB)` into base weight each step |

## Configuration reference

Adapters are configured via `AdapterBuildConfig`/`TargetSpec` (Python) or the
`kind:` / `targets:` block in a YAML file (CLI). See `configs/*.yaml`.

## Tests

```bash
pytest -q            # CPU-only unit + tiny SST-2 smoke test
```

## Project structure

```
src/lora_factory/
  adapters/   LoRA, LoRA-FA, VeRA, Delta-LoRA + registry
  config.py   AdapterBuildConfig, TargetSpec
  hf/         GLUE/SST-2 helpers + Trainer extension
  nn/inject.py  module match/wrap logic
  utils/      seed, optimizer (LoRA+), state, yaml parsing
```

## Extensibility

Add a new adapter by subclassing `AdapterModule` (implement `adapter_forward`)
and registering it in `registry.py` under a `kind` string.
```

- [ ] **Step 2: Add reference comments to the YAML configs**

In `configs/sst2_lora.yaml`, after the existing block, append (and mirror the shape in the other two configs):

```yaml
# Adapter build config (parsed by lora_factory.utils.yaml_build)
kind: "lora"                # lora | lora_fa | vera | delta_lora
lora_plus_lr_ratio: 16.0    # lr_B = ratio * lr_A (LoRA+)
targets:
  - module_names: ["query", "key", "value", "q_lin", "k_lin", "v_lin", "out_proj", "dense", "out_lin"]
    module_types: ["Linear"]
    rank: 8                 # adapter dimension
    alpha: 16               # effective scale is alpha / rank
    dropout: 0.0
```

- [ ] **Step 3: Standardize module docstrings**

For each source module under `src/lora_factory/`, ensure it has a one-line module docstring (or a clear existing one). Add/verify:
- `adapters/base.py`, `adapters/lora.py`, `adapters/lora_fa.py`, `adapters/vera.py`, `adapters/delta_lora.py`, `adapters/registry.py`
- `config.py`, `nn/inject.py`, `utils/optim.py`, `utils/state.py`, `utils/yaml_cfg.py`, `utils/seed.py`
- `hf/sst2_helper.py`, `hf/trainer_ext.py`

Use the pattern: a single sentence describing the module's responsibility. Do not change public names.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md configs/ src/lora_factory/
git commit -m "docs: professionalize README, config comments, and docstrings"
```

---

## Self-Review

- **Spec coverage:** 1.1 (Task 1) ✓; 1.3 robustness — double-wrap + parent lookup (Tasks 2 & 3) ✓; 1.2 frozen-param hygiene (Task 3) ✓; 1.4 trainer timing (Task 4 Step 3) ✓; 2 verification (Task 4) ✓; 3 documentation (Task 5) ✓.
- **Placeholder scan:** No TBD/TODO/"add appropriate" placeholders; every step has concrete code and commands.
- **Type consistency:** `set_seed(seed: int) -> None` used consistently; `_find_parent_with_attr` signature unchanged; adapter class names match existing imports.

## Post-Plan Checklist
- [ ] All spec sections (1.1–1.4, 2, 3) map to a task above (see Self-Review).
- [ ] Full test suite green in the venv.
- [ ] Final commit exists for each task.