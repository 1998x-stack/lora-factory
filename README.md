# lora-factory

**What’s inside**
- Adapters: LoRA, LoRA-FA (Frozen-A), VeRA (shared A/B + layer vectors), Delta-LoRA.
- **LoRA+** via optimizer param groups: larger LR for B-like params.
- HF integration for **GLUE/SST-2** (`scripts/train_sst2.py`) with YAML configs.

**Quickstart**
```bash
pip install -e .
python examples/tiny_mlp_demo.py
python scripts/train_sst2.py --config configs/sst2_lora.yaml
```

**How to run**
```bash
# Install
pip install -e .

# Quick sanity check (CPU ok)
pytest -q

# Train SST-2 with LoRA
python scripts/train_sst2.py --config configs/sst2_lora.yaml

# Try VeRA or Delta-LoRA
python scripts/train_sst2.py --config configs/sst2_vera.yaml
python scripts/train_sst2.py --config configs/sst2_delta_lora.yaml
```
