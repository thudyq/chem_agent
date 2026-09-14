# -*- coding: utf-8 -*-
"""tests/conftest.py — 共享测试 fixture。

fake_rdkit：用假 rdkit 覆盖 tag_validator 的 SMILES 语义校验，
KNOWN_SMILES 中的 SMILES 视为合法，其余视为非法。
fake_renderers：把注册表渲染器替换为固定输出，并自动恢复。
workspace_tmp_path：受限环境下替代 pytest 内置 tmp_path（系统 temp 不可写时）。
"""

import os
import shutil
import stat
import types
import uuid
from pathlib import Path

import pytest

import core.tag_validator as tv
from renderers import registry

# ---------------------------------------------------------------- 临时目录
# 背景：pytest 内置 tmp_path 走系统 temp。受限环境（容器 / CI / DSH 沙箱）
# 禁止写系统 temp，用例会以 WinError 5 / PermissionError 失败，看起来像
# "代码坏了"。下面的 fixture 同名覆盖内置 tmp_path：**先探测系统 temp 是否
# 可写**，可写就完全沿用 pytest 原语义（基目录仍由 pytest 管理），不可写才
# 回退到工作区内的 data/pytest_tmp（data/ 已被 .gitignore 覆盖）。

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def workspace_tmp_path():
    """返回一个可写临时目录（优先 TMPDIR 风格环境变量，其次工作区内）。"""
    explicit = os.environ.get("CHEM_AGENT_TEST_TMP", "").strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(_PROJECT_ROOT / "data" / "pytest_tmp")
    for root in candidates:
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return root
        except OSError:
            continue
    raise RuntimeError("找不到可写的测试临时目录（可设 CHEM_AGENT_TEST_TMP 指定）")


def _rmtree_force(path: Path) -> None:
    """删目录树（Windows 上先清只读属性再删，避免 PermissionError）。"""
    def _onerr(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_onerr)


@pytest.fixture(scope="session")
def _tmp_base(tmp_path_factory):
    """会话级临时基目录：系统 temp 可写则用 pytest 的，否则用工作区内。

    只在 **yield 之前** 探测（不能把 yield 包进 try/except OSError——
    测试异常会被抛回生成器，被 except 吞掉后二次 yield 变成 RuntimeError）。
    """
    try:
        return tmp_path_factory.mktemp("base")
    except OSError:
        return workspace_tmp_path()


@pytest.fixture
def tmp_path(_tmp_base):
    """可写临时目录：系统 temp 可用时等价于 pytest 内置 tmp_path。"""
    root = _tmp_base / uuid.uuid4().hex[:12]
    root.mkdir(parents=True, exist_ok=True)
    yield root
    _rmtree_force(root)


def ensure_workspace_tmp() -> Path:
    """测试用的工作区临时目录（沙箱/容器里系统 temp 不可写时的回退）。"""
    return workspace_tmp_path()


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
    # 自由基机理：单原子/小分子组分
    "C": _FakeMol(1), "Cl": _FakeMol(1), "ClCl": _FakeMol(2),
    "[Cl]": _FakeMol(1), "[CH3]": _FakeMol(1),
    # 显式 H 是真实原子参与编号（a#k 废弃）——甲烷显式 H 写法
    "C([H])([H])([H])[H]": _FakeMol(5), "O([H])[H]": _FakeMol(3),
    "N([H])([H])[H]": _FakeMol(4), "[H]OCCO": _FakeMol(5),
    "CC(=O)O[H]": _FakeMol(5),
    # CHAIR：取代环己烷
    "BrC1CCCCC1": _FakeMol(7), "CC1CCCCC1": _FakeMol(7),
    # corpus：苯磺化（EAS 机理），真实 RDKit 可解析的凯库勒苯/SO3/σ 络合物
    "C1=CC=CC=C1": _FakeMol(6), "O=S(=O)=O": _FakeMol(4),
    "O=S([O-])(=O)C([H])1C=CC=C[CH+]1": _FakeMol(10),
    # corpus：环氧误报病例（1,2-环氧丁烷/甲醇/2-甲氧基-1-丁醇）
    "CCC1CO1": _FakeMol(5), "OCC(OC)CC": _FakeMol(7),
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


@pytest.fixture
def no_aux_calls(monkeypatch):
    """屏蔽辅助 LLM 调用（手术式箭头/结构重写），让修正闭环用例聚焦主流程。

    背景：`app` 的辅助调用按"文本返回"使用 `ask_llm` 的返回值，与主生成
    的 `return_result=True` 形态不同；多数修正用例并不关心辅助路径，
    直接让它们返回 None（=重写失败 → 回退常规修正）即可。
    """
    monkeypatch.setattr("app._rewrite_struct_smiles",
                        lambda *a, **k: None)
    monkeypatch.setattr("app._rewrite_composite_arrows",
                        lambda *a, **k: None)


@pytest.fixture(autouse=True)
def isolate_credential_state():
    """每个测试前后清理凭证层状态（基准配置 / 信号量 / 能力表 / 档位提示）。

    背景：配置与凭证已统一到 `core.credentials`，测试若注入替身而忘记恢复，
    会污染后续用例；能力表（`core.capabilities`）是跨请求的进程级缓存，
    同样必须在测试间隔离。**模型路由已删除**（单模型），因此不再需要
    旧的 `no_model_route` 替身。
    """
    from core import capabilities, credentials

    def _clean():
        credentials.set_base_settings_for_tests(None)
        credentials._reset_semaphores_for_tests()
        # `apply_credentials()` 只设不重置（生成器场景需要），pytest 却全程
        # 共用同一个 Context，不清理会泄漏到后续用例
        credentials.reset_for_tests()
        capabilities.reset_for_tests()
        try:
            import app
            app._EFFORT_NOTICE_SHOWN.clear()
        except Exception:
            pass

    _clean()
    yield
    _clean()
