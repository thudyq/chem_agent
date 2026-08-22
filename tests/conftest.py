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
    "C[C@@H](O)C(=O)O": _FakeMol(6),
    # B1（20260812）自由基机理：单原子/小分子组分
    "C": _FakeMol(1), "Cl": _FakeMol(1), "ClCl": _FakeMol(2),
    "[Cl]": _FakeMol(1), "[CH3]": _FakeMol(1),
    # 20260821：显式 H 是真实原子参与编号（a#k 废弃）——甲烷显式 H 写法
    "C([H])([H])([H])[H]": _FakeMol(5), "O([H])[H]": _FakeMol(3),
    "N([H])([H])[H]": _FakeMol(4), "[H]OCCO": _FakeMol(5),
    "CC(=O)O[H]": _FakeMol(5),
    # CHAIR：取代环己烷
    "BrC1CCCCC1": _FakeMol(7), "CC1CCCCC1": _FakeMol(7),
    # 20260822 corpus：苯磺化（EAS 机理），真实 RDKit 可解析的凯库勒苯/SO3/σ 络合物
    "C1=CC=CC=C1": _FakeMol(6), "O=S(=O)=O": _FakeMol(4),
    "O=S([O-])(=O)C([H])1C=CC=C[CH+]1": _FakeMol(10),
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


@pytest.fixture(autouse=True)
def no_model_route(monkeypatch):
    """默认禁用"flash 首跑 + 失败升级"路由——测试不依赖 .env 的
    UPGRADE_MODEL_NAME/UPGRADE_KEYWORDS 配置（用户本地 .env 可能已启用路由）。

    路由行为测试（tests/test_correction.py 的 test_route_*）用各自
    monkeypatch 重新设置 app.settings 显式启用。
    """
    import types
    monkeypatch.setattr(
        "app.settings",
        types.SimpleNamespace(
            llm=types.SimpleNamespace(upgrade_model_name="")))
