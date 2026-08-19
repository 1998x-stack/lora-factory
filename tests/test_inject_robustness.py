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