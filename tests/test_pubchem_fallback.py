# -*- coding: utf-8 -*-
"""tests/test_pubchem_fallback.py — app.py 的 PubChem 兜底辅助函数测试（校验失败时注入权威 SMILES 参考）。

mock 掉 LLM 翻译与 PubChem 网络（不依赖真实服务）：
- _translate_name_zh2en：monkeypatch app.ask_llm
- _fetch_pubchem_references：monkeypatch app.name_to_smiles
"""

import pytest

import app
from core.tag_parser import parse_tags


@pytest.fixture(autouse=True)
def _clear_pubchem_fail_cache():
    """每个用例前清空 PubChem 兜底失败缓存（模块级状态，避免污染）。"""
    app._PUBCHEM_FAIL_CACHE.clear()
    yield
    app._PUBCHEM_FAIL_CACHE.clear()


def _failure(raw: str, reason: str):
    tag = parse_tags(raw)[0]
    return [(tag, reason)]


# ---------------------------------------------------------------- _extract_chem_labels

def test_extract_chem_labels_picks_smiles_failures():
    """SMILES 相关失败（无效/化学校验/越界）提取 label。"""
    fails = _failure("[STRUCT:XYZbad,label=阿司匹林]", "无效 SMILES「XYZbad」")
    labels = app._extract_chem_labels(fails)
    assert labels == [("阿司匹林", "无效 SMILES「XYZbad」")]


def test_extract_chem_labels_filters_role_words():
    """角色/流程 label（底物/中间体等）不作为化合物名。"""
    fails = _failure("[STRUCT:CCO,label=底物]", "无效 SMILES")
    assert app._extract_chem_labels(fails) == []


def test_extract_chem_labels_filters_non_smiles_failures():
    """非 SMILES 失败（label 过长）不触发 PubChem。"""
    fails = _failure("[STRUCT:CCO,label=阿司匹林]", "label 过长")
    assert app._extract_chem_labels(fails) == []


def test_extract_chem_labels_requires_label():
    """无 label 的失败标记跳过。"""
    fails = _failure("[STRUCT:XYZbad]", "无效 SMILES")
    assert app._extract_chem_labels(fails) == []


# ---------------------------------------------------------------- _translate_name_zh2en

def test_translate_name_zh2en_english_passthrough():
    """英文名直通（无需 LLM）。"""
    assert app._translate_name_zh2en("aspirin") == "aspirin"


def test_translate_name_zh2en_calls_llm(monkeypatch):
    """中文名调 LLM 翻译。"""
    monkeypatch.setattr(app, "ask_llm", lambda *a, **k: "Aspirin")
    assert app._translate_name_zh2en("阿司匹林") == "Aspirin"


def test_translate_name_zh2en_llm_failure_returns_none(monkeypatch):
    """LLM 翻译失败 → None（不抛异常）。"""
    monkeypatch.setattr(app, "ask_llm", lambda *a, **k: None)
    assert app._translate_name_zh2en("阿司匹林") is None


# ---------------------------------------------------------------- _fetch_pubchem_references

def test_fetch_pubchem_references_full_flow(monkeypatch):
    """完整链路：提取 → 翻译 → PubChem → 参考文本。"""
    monkeypatch.setattr(app, "ask_llm", lambda *a, **k: "Aspirin")
    monkeypatch.setattr(
        "utils.name_resolver.name_to_smiles",
        lambda name: "CC(=O)OC1=CC=CC=C1C(=O)O" if name == "Aspirin" else None,
    )
    fails = _failure("[STRUCT:XYZbad,label=阿司匹林]", "无效 SMILES「XYZbad」")
    ref = app._fetch_pubchem_references(fails)
    assert "阿司匹林" in ref
    assert "CC(=O)OC1=CC=CC=C1C(=O)O" in ref


def test_fetch_pubchem_references_skips_when_pubchem_miss(monkeypatch):
    """PubChem 查不到 → 无参考（静默）。"""
    monkeypatch.setattr(app, "ask_llm", lambda *a, **k: "UnknownXYZ")
    monkeypatch.setattr(
        "utils.name_resolver.name_to_smiles", lambda name: None)
    fails = _failure("[STRUCT:XYZbad,label=未知物]", "无效 SMILES")
    assert app._fetch_pubchem_references(fails) == ""


def test_fetch_pubchem_references_limits(monkeypatch):
    """最多处理 limit 个失败标记。"""
    monkeypatch.setattr(app, "ask_llm", lambda *a, **k: "Aspirin")
    monkeypatch.setattr(
        "utils.name_resolver.name_to_smiles",
        lambda name: "CC(=O)OC1=CC=CC=C1C(=O)O",
    )
    fails = (_failure("[STRUCT:A,label=阿司匹林]", "无效 SMILES")
             + _failure("[STRUCT:B,label=乙醇]", "无效 SMILES")
             + _failure("[STRUCT:C,label=苯]", "无效 SMILES"))
    ref = app._fetch_pubchem_references(fails)
    assert ref.count("的 PubChem 标准 SMILES") == 2  # limit=2


def test_fetch_pubchem_references_role_labels_skipped(monkeypatch):
    """角色 label 全部过滤 → 无参考。"""
    monkeypatch.setattr(app, "ask_llm", lambda *a, **k: "X")
    monkeypatch.setattr(
        "utils.name_resolver.name_to_smiles", lambda name: "CCO")
    fails = _failure("[STRUCT:CCO,label=底物]", "无效 SMILES")
    assert app._fetch_pubchem_references(fails) == ""


def test_fetch_pubchem_references_caches_miss(monkeypatch):
    """PubChem 查不到的 label 缓存失败：第二次不再翻译（不调 LLM）。"""
    calls = {"llm": 0}

    def fake_ask_llm(*a, **k):
        calls["llm"] += 1
        return "UnknownXYZ"

    monkeypatch.setattr(app, "ask_llm", fake_ask_llm)
    monkeypatch.setattr(
        "utils.name_resolver.name_to_smiles", lambda name: None)
    fails = _failure("[STRUCT:XYZbad,label=未知物]", "无效 SMILES")
    assert app._fetch_pubchem_references(fails) == ""
    assert app._fetch_pubchem_references(fails) == ""
    assert calls["llm"] == 1  # 第二次命中失败缓存，不再翻译


def test_fetch_pubchem_references_caches_translate_failure(monkeypatch):
    """翻译失败同样缓存：第二次不再调 LLM。"""
    calls = {"llm": 0}

    def fake_ask_llm(*a, **k):
        calls["llm"] += 1
        return None

    monkeypatch.setattr(app, "ask_llm", fake_ask_llm)
    monkeypatch.setattr(
        "utils.name_resolver.name_to_smiles", lambda name: "CCO")
    fails = _failure("[STRUCT:XYZbad,label=某中文物]", "无效 SMILES")
    assert app._fetch_pubchem_references(fails) == ""
    assert app._fetch_pubchem_references(fails) == ""
    assert calls["llm"] == 1


def test_fetch_pubchem_references_success_not_cached(monkeypatch):
    """成功查询不缓存：再次调用照常翻译 + 查询。"""
    calls = {"llm": 0}

    def fake_ask_llm(*a, **k):
        calls["llm"] += 1
        return "Aspirin"

    monkeypatch.setattr(app, "ask_llm", fake_ask_llm)
    monkeypatch.setattr(
        "utils.name_resolver.name_to_smiles",
        lambda name: "CC(=O)OC1=CC=CC=C1C(=O)O")
    fails = _failure("[STRUCT:XYZbad,label=阿司匹林]", "无效 SMILES")
    assert app._fetch_pubchem_references(fails) != ""
    assert app._fetch_pubchem_references(fails) != ""
    assert calls["llm"] == 2  # 成功不缓存，照常重查


def test_fetch_pubchem_references_caches_by_label(monkeypatch):
    """失败缓存按 label 隔离：不同 label 互不影响。"""
    calls = {"llm": 0}

    def fake_ask_llm(*a, **k):
        calls["llm"] += 1
        return "Known" if a[0] == "已知物" else "UnknownXYZ"

    monkeypatch.setattr(app, "ask_llm", fake_ask_llm)
    monkeypatch.setattr(
        "utils.name_resolver.name_to_smiles",
        lambda name: "CCO" if name == "Known" else None)
    fail_unknown = _failure("[STRUCT:A,label=未知物]", "无效 SMILES")
    fail_known = _failure("[STRUCT:B,label=已知物]", "无效 SMILES")
    assert app._fetch_pubchem_references(fail_unknown) == ""
    assert app._fetch_pubchem_references(fail_known) != ""   # 已知物不受影响
    assert app._fetch_pubchem_references(fail_unknown) == ""  # 命中缓存
    assert calls["llm"] == 2  # 未知物 1 次 + 已知物 1 次


def test_build_correction_prompt_includes_pubchem_ref(monkeypatch):
    """修正 prompt 含 PubChem 参考。"""
    monkeypatch.setattr(app, "ask_llm", lambda *a, **k: "Aspirin")
    monkeypatch.setattr(
        "utils.name_resolver.name_to_smiles",
        lambda name: "CC(=O)OC1=CC=CC=C1C(=O)O",
    )
    fails = _failure("[STRUCT:XYZbad,label=阿司匹林]", "无效 SMILES「XYZbad」")
    prompt = app._build_correction_prompt("问题", "原回答", fails)
    assert "PubChem 参考" in prompt
    assert "CC(=O)OC1=CC=CC=C1C(=O)O" in prompt
