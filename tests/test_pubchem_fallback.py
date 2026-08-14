# -*- coding: utf-8 -*-
"""app.py 的 PubChem 兜底辅助函数测试（校验失败时注入权威 SMILES 参考）。

mock 掉 LLM 翻译与 PubChem 网络（不依赖真实服务）：
- _translate_name_zh2en：monkeypatch app.ask_llm
- _fetch_pubchem_references：monkeypatch app.name_to_smiles
"""

import pytest

import app
from core.tag_parser import parse_tags


def _failure(raw: str, reason: str):
    tag = parse_tags(raw)[0]
    return [(tag, reason)]


# ---------------- _extract_chem_labels ----------------

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


# ---------------- _translate_name_zh2en ----------------

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


# ---------------- _fetch_pubchem_references ----------------

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
