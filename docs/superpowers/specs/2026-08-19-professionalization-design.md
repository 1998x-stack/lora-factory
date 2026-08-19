# lora-factory: Professionalization Design

**Date:** 2026-08-19
**Status:** Approved by user (approach B: "surgical professionalization")
**Scope:** Fix correctness and polish docs; no unrelated refactoring.

## Objective

Bring `lora-factory` to a professional-quality state: make it actually run
(correctness first), harden the adapter/injection behavior around a specific set
of concerns, and raise the documentation to a clean, reference-grade standard.
The package must remain a simple, readable educational repo — simplicity wins
over convenience everywhere there is a tradeoff.

## Current State (as discovered)

- Package layout: `src/lora_factory/{adapters,config,hf,nn,utils}`.
- Adapters: `lora`, `lora_fa`, `vera`, `delta_lora` behind a registry.
- HF integration: GLUE/SST-2 runner (`scripts/train_sst2.py`) + `AdapterTrainer`.
- **Blocker:** `src/lora_factory/nn/inject.py` and `scripts/train_sst2.py`
  import `from lora_factory.utils.seed import set_seed`, but
  `utils/seed.py` does not exist. Every top-level import of the package fails
  with `ModuleNotFoundError: No module named 'lora_factory.utils.seed'`.

## 1. Correctness

### 1.1 Create `utils/seed.py`
- Provide `set_seed(seed: int) -> None` that:
  - seeds `random.seed`, `numpy.random.seed`,
  - seeds `torch` CPU generators and `torch.cuda.manual_seed_all`,
  - sets `torch.backends.cudnn.deterministic = True` and
    `torch.backends.cudnn.benchmark = False`.
- No signature changes elsewhere; this matches the existing call sites.

### 1.2 Frozen-parameter hygiene
Preserve existing intent; add tests to lock it in:
- **LoRA-FA:** `lora_A` frozen, `lora_B` trainable.
- **VeRA:** shared A/B are non-trainable; only `vera_b` and `vera_d` trainable.
- **All adapters:** wrapped base `nn.Linear` weights/bias frozen.
- **Optimizer groups:** `split_lora_params_for_lora_plus` must skip
  `requires_grad=False` params (already implemented; verify + test).

### 1.3 Robustness
- **Double-wrap avoidance:** already in `_match_module`; add a test that calling
  `apply_adapters` twice does not re-wrap an already-adapted layer.
- **Parent lookup:** `_find_parent_with_attr` should raise a clear error
  (KeyError/ValueError with module path) if a module name cannot be resolved,
  instead of a buried `AttributeError`.
- **VeRA shared-matrix cache:** confirm shape/dtype/device correctness only.
  Per user decision, leave the per-forward `.to()` as-is (no hot-path
  optimization); clarity over speed.

### 1.4 Trainer correctness
- Confirm Delta-LoRA post-step merge in `optimizer_step` fires correctly with
  gradient accumulation (HF calls `optimizer_step` once per accumulation cycle,
  so the timing is correct). Keep the implementation.

## 2. Verification
- Run the existing tests (`test_build_and_inject`, `test_hf_sst2_smoke`) and
  `examples/tiny_mlp_demo.py` in a local venv (torch CPU).
- Add focused tests:
  - `set_seed` determinism (same input seed => same RNG stream).
  - Per-adapter frozen/trainable parameter sets match the intended hygiene.
  - Optimizer param-group membership reflects the LoRA+ split.
  - Calling `apply_adapters` twice does not double-wrap.

## 3. Documentation
- Rewrite `README.md` with a clean, professional structure:
  Overview, Features, Install, Quickstart (API + CLI), Configuration reference,
  Adapters table, Testing, Project structure, Extensibility.
- Standardize module-level docstrings while keeping existing naming and
  semantics.
- Refresh the `.yaml` config comments to read as reference documentation.

## Constraints
- No unrelated refactoring; keep the existing package layout, naming, and
  adapter semantics exactly as-is.
- Educational repo: favor simplicity and clarity.

## Out of Scope (YAGNI)
- Adapter redesign or new variants.
- Package/packaging overhaul (full refactor approach).
- Apart from the documentation rewrite, any broader build/CI introduction.