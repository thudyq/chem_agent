# -*- coding: utf-8 -*-
"""tests/conftest.py — 共享测试 fixture。

fake_rdkit：用假 rdkit 覆盖 tag_validator 的 SMILES 语义校验，
KNOWN_SMILES 中的 SMILES 视为合法，其余视为非法。
fake_renderers：把注册表渲染器替换为固定输出，并自动恢复。
"""

import types

import pytest

import core.tag_validator as tv
from renderers import registry


class _FakeMol:
    def __init__(self, n_atoms):
        self._n = n_atoms

    def GetNumAtoms(self):
        return self._n


KNOWN_SMILES = {
    "CCl": _FakeMol(2), "CO": _FakeMol(2), "[OH-]": _FakeMol(2),
    "c1ccccc1": _FakeMol(6), "CCO": _FakeMol(3), "CC[OH2+]": _FakeMol(3),
    "CC[OH+]CC": _FakeMol(5), "O": _FakeMol(1), "CCOCC": _FakeMol(4),
    "OCCO": _FakeMol(4), "C=C": _FakeMol(2), "CC=O": _FakeMol(3),
    "[H]O[H]": _FakeMol(3), "[H]": _FakeMol(1), "[H+]": _FakeMol(1),
}


@pytest.fixture
def fake_rdkit(monkeypatch):
    fake_chem = types.SimpleNamespace(
        MolFromSmiles=lambda smi: KNOWN_SMILES.get(smi))
    monkeypatch.setattr(tv, "_RDKIT_OK", True)
    monkeypatch.setattr(tv, "Chem", fake_chem)


@pytest.fixture
def fake_renderers(monkeypatch):
    original = dict(registry.RENDERER_REGISTRY)
    for key in list(registry.RENDERER_REGISTRY):
        registry.RENDERER_REGISTRY[key] = lambda *a: f"RENDERED:{a[0]}"
    yield
    registry.RENDERER_REGISTRY.clear()
    registry.RENDERER_REGISTRY.update(original)
