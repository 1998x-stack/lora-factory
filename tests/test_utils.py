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