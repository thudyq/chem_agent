# -*- coding: utf-8 -*-
"""utils/rxn_balancer 单元测试：SynRBL 配平兜底。

synrbl 首次 import ~23s、MCS 搜索最坏 ~50s，测试全部用 monkeypatch 模拟
（不实例化真实 Balancer）：门槛 / 输出复校验 / 超时降级 / 修正 prompt 注入。
"""

import threading
import time

import pytest

from core.tag_parser import RenderTag


@pytest.fixture
def no_synrbl(monkeypatch):
    """模拟 synrbl 未安装：balance_reaction 直接返回 None。"""
    import utils.rxn_balancer as rb
    monkeypatch.setattr(rb, "_load_synrbl", lambda: False)
    monkeypatch.setattr(rb, "_stuck", False)


@pytest.fixture
def fake_balancer(monkeypatch):
    """把 _run_with_timeout 替换为直接返回给定的配平结果（跳过真实 SynRBL）。"""
    import utils.rxn_balancer as rb
    holder = {"result": None}

    def fake_run(fn, timeout):
        return holder["result"]

    monkeypatch.setattr(rb, "_run_with_timeout", fake_run)
    monkeypatch.setattr(rb, "_load_synrbl", lambda: True)
    monkeypatch.setattr(rb, "_stuck", False)
    return holder


def test_not_available_returns_none(no_synrbl):
    from utils.rxn_balancer import balance_reaction
    assert balance_reaction("CCO", "CC=O") is None
    assert balance_reaction("", "") is None


def test_empty_or_unparseable_returns_none(fake_balancer):
    from utils.rxn_balancer import balance_reaction
    assert balance_reaction("", "CC=O") is None
    assert balance_reaction("CCO", "") is None
    # 非法系数（1/3）组分被丢弃后为空 → None
    assert balance_reaction("1/3CCO", "CC=O") is None


def test_formula_species_gate(fake_balancer):
    """教科文化学式（KMnO4/H2SO4 等非 SMILES）不喂 SynRBL。"""
    from utils.rxn_balancer import balance_reaction
    assert balance_reaction("KMnO4", "CC=O") is None
    assert balance_reaction("CCO;H2O", "CC=O") is None


def test_imputation_hint_used(fake_balancer):
    """SynRBL 补全物种且通过守恒复核 → 返回提示文本（LLM 照抄）。"""
    fake_balancer["result"] = ["CC(=O)O.CCO>>CC(=O)OCC.O"]
    from utils.rxn_balancer import balance_reaction
    hint = balance_reaction("CC(=O)O;CCO", "CC(=O)OCC")
    assert hint is not None
    assert "SynRBL 自动配平参考" in hint
    assert "CC(=O)O" in hint and "CCO" in hint
    assert "O" in hint.split("生成物")[1]


def test_noop_result_rejected(fake_balancer):
    """SynRBL 原样返回（系数错误类，无补全信息）→ None。"""
    fake_balancer["result"] = ["CCO>>CCOCC"]
    from utils.rxn_balancer import balance_reaction
    assert balance_reaction("CCO", "CCOCC") is None


def test_unbalanced_output_rejected(fake_balancer):
    """SynRBL 输出仍不守恒 → 不采用（复校验兜底）。"""
    fake_balancer["result"] = ["CCO>>CC=O.O"]
    from utils.rxn_balancer import balance_reaction
    # C2H6O vs C2H4O+O=C2H4O2：H 差 2，不守恒
    assert balance_reaction("CCO", "CC=O") is None


def test_timeout_sets_stuck(monkeypatch):
    """超时 → 返回 None 且 _stuck 置位；后台结束后复位。"""
    import utils.rxn_balancer as rb
    monkeypatch.setattr(rb, "_SYNRBL_OK", True)
    monkeypatch.setattr(rb, "_stuck", False)
    monkeypatch.setattr(rb, "_TIMEOUT", 0.05)
    release = threading.Event()

    def slow_fn():
        release.wait(5)
        return ["CCO>>CC=O"]

    t0 = time.time()
    res = rb._run_with_timeout(slow_fn, rb._TIMEOUT)
    assert res is None
    assert time.time() - t0 < 1.0  # 未等满 slow_fn 的 5s
    assert rb._stuck is True
    # stuck 期间新调用直接跳过（不再起线程）
    assert rb._run_with_timeout(lambda: ["x"], rb._TIMEOUT) is None
    release.set()
    # reaper 线程等 slow_fn 结束后复位 _stuck
    deadline = time.time() + 3
    while rb._stuck and time.time() < deadline:
        time.sleep(0.05)
    assert rb._stuck is False


def test_balancer_available_flag(no_synrbl):
    from utils.rxn_balancer import balancer_available
    assert balancer_available() is False


def _reaction_tag(raw, args):
    """构造 REACTION RenderTag（直接传 args，绕过解析器）。"""
    return RenderTag(type="REACTION", args=args, raw=raw, start_pos=0, end_pos=0)


def test_fetch_synrbl_balance_hint(monkeypatch):
    """化学校验失败的 REACTION 标记 → 修正 prompt 注入 SynRBL 配平参考。"""
    from app import _fetch_synrbl_balance
    monkeypatch.setattr(
        "utils.rxn_balancer.balance_reaction",
        lambda l, r: "SynRBL 自动配平参考（已通过守恒校验）：反应物 A，生成物 B",
    )
    tag = _reaction_tag("[REACTION:CCO|CC=O|Cu, Δ]", ["CCO", "CC=O", "Cu, Δ"])
    out = _fetch_synrbl_balance([(tag, "化学校验：方程式两侧原子不守恒")])
    assert "SynRBL 自动配平参考" in out
    assert "CC=O" in out  # 原始标记片段


def test_fetch_synrbl_balance_skips_non_chem_failure(monkeypatch):
    """非化学校验失败（无效 SMILES 等）不触发 SynRBL。"""
    from app import _fetch_synrbl_balance
    called = []
    monkeypatch.setattr(
        "utils.rxn_balancer.balance_reaction",
        lambda l, r: called.append((l, r)) or "hint",
    )
    tag = _reaction_tag("[REACTION:XYZ|CC=O|Cu, Δ]", ["XYZ", "CC=O", "Cu, Δ"])
    assert _fetch_synrbl_balance([(tag, "无效 SMILES「XYZ」")]) == ""
    assert called == []


def test_fetch_synrbl_balance_skips_non_reaction(monkeypatch):
    """非 REACTION 标记（STRUCT 等）不触发 SynRBL。"""
    from app import _fetch_synrbl_balance
    called = []
    monkeypatch.setattr(
        "utils.rxn_balancer.balance_reaction",
        lambda l, r: called.append((l, r)) or "hint",
    )
    tag = RenderTag(type="STRUCT", args=["CCO", None], raw="[STRUCT:CCO]",
                    start_pos=0, end_pos=0)
    assert _fetch_synrbl_balance([(tag, "化学校验：label 与 SMILES 不一致")]) == ""
    assert called == []


def test_correction_prompt_includes_synrbl_reference(monkeypatch):
    """P2 修正 prompt 端到端：化学校验失败 → 含 SynRBL 配平参考。"""
    from app import _build_correction_prompt
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "utils.rxn_balancer.balance_reaction",
        lambda l, r: "SynRBL 自动配平参考（已通过守恒校验）：反应物"
                     "CC(=O)O;CCO，生成物 CC(=O)OCC;O",
    )
    tag = _reaction_tag(
        "[REACTION:CC(=O)O;CCO|CC(=O)OCC|浓H2SO4, Δ]",
        ["CC(=O)O;CCO", "CC(=O)OCC", "浓H2SO4, Δ"],
    )
    prompt = _build_correction_prompt(
        "写酯化反应", "[REACTION:CC(=O)O;CCO|CC(=O)OCC|浓H2SO4, Δ]",
        [(tag, "化学校验：方程式两侧原子不守恒（C2H4O2+C2H6O vs C4H8O2，需配平或补全物种；辅助试剂请写入箭头条件而非省略主物种）")],
    )
    assert "SynRBL 自动配平参考" in prompt
    assert "CC(=O)OCC;O" in prompt
