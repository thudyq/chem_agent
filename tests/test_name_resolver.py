# -*- coding: utf-8 -*-
"""utils/name_resolver 单元测试：PubChem 名称→SMILES。

网络请求用 monkeypatch 模拟（不依赖真实 PubChem 可达性）：
- http_get 返回构造的响应对象（status_code / text）。
- 中文名映射英文后查询。
"""

import pytest

from utils.name_resolver import (
    COMMON_CN_EN, _is_valid_smiles, name_to_smiles,
)


class _FakeResp:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text
        self.headers = {}


def test_common_cn_en_mapping():
    """20 个常用化合物的中文名映射英文。"""
    assert COMMON_CN_EN["阿司匹林"] == "aspirin"
    assert COMMON_CN_EN["乙醇"] == "ethanol"
    assert COMMON_CN_EN["苯"] == "benzene"
    assert len(COMMON_CN_EN) == 20


def test_name_to_smiles_pubchem_hit(monkeypatch):
    """PubChem 返回合法 SMILES → 命中。"""
    monkeypatch.setattr(
        "utils.name_resolver.http_get",
        lambda url, timeout=10, retries_503=3: _FakeResp(200, "CCO\n"),
    )
    assert name_to_smiles("ethanol") == "CCO"


def test_name_to_smiles_cn_maps_to_en(monkeypatch):
    """中文名映射英文后查询（URL 含英文名）。"""
    captured = {}

    def fake_get(url, timeout=10, retries_503=3):
        captured["url"] = url
        return _FakeResp(200, "CCO\n")

    monkeypatch.setattr("utils.name_resolver.http_get", fake_get)
    assert name_to_smiles("乙醇") == "CCO"
    assert "ethanol" in captured["url"]


def test_name_to_smiles_multi_line(monkeypatch):
    """PubChem 返回多行（同名多记录）→ 取首个合法 SMILES。"""
    monkeypatch.setattr(
        "utils.name_resolver.http_get",
        lambda url, timeout=10, retries_503=3: _FakeResp(200, "bad\nCCO\n"),
    )
    assert name_to_smiles("ethanol") == "CCO"


def test_name_to_smiles_invalid_smiles_rejected(monkeypatch):
    """PubChem 返回非法 SMILES → 返回 None。"""
    monkeypatch.setattr(
        "utils.name_resolver.http_get",
        lambda url, timeout=10, retries_503=3: _FakeResp(200, "not-a-smiles\n"),
    )
    assert name_to_smiles("xyz") is None


def test_name_to_smiles_404(monkeypatch):
    """PubChem 404（未收录）→ None。"""
    monkeypatch.setattr(
        "utils.name_resolver.http_get",
        lambda url, timeout=10, retries_503=3: _FakeResp(404),
    )
    assert name_to_smiles("not-a-real-compound") is None


def test_name_to_smiles_network_error(monkeypatch):
    """网络异常 → None（不抛异常）。"""
    def boom(url, timeout=10, retries_503=3):
        raise OSError("network down")

    monkeypatch.setattr("utils.name_resolver.http_get", boom)
    assert name_to_smiles("ethanol") is None


def test_name_to_smiles_empty():
    """空/None 输入 → None。"""
    assert name_to_smiles("") is None
    assert name_to_smiles(None) is None
    assert name_to_smiles("   ") is None


def test_is_valid_smiles():
    """RDKit 校验 SMILES。"""
    assert _is_valid_smiles("CCO")
    assert not _is_valid_smiles("")
    assert not _is_valid_smiles("not-smiles")
