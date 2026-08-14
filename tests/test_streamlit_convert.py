# -*- coding: utf-8 -*-
"""streamlit_app 的 mhchem \\ce{...} → KaTeX 转换（_convert_ce_math）单元测试。

覆盖双轨制之外、LLM 直写 \\ce{...} 时的排版规则：电荷上标（含方括号配离子）、
数字下标、沉淀符号（mhchem 的孤立 v / ↓ → \\downarrow）。
"""

import pytest

from streamlit_app import _convert_ce_math


def test_ce_basic_subscript_and_arrow():
    """基础：数字下标 + 反应箭头。"""
    out = _convert_ce_math(r"\ce{C6H6 + HNO3 ->[H2SO4, \triangle] C6H5NO2 + H2O}")
    assert "C_6H_6" in out
    assert "HNO_3" in out
    assert r"\xrightarrow{H_2SO_4, \triangle}" in out
    assert "C_6H_5NO_2" in out
    assert "H_2O" in out


def test_ce_charge_superscript_bracket_complex():
    """方括号配离子电荷上标：[Ag(NH3)2]+ → [Ag(NH_3)_2]^{+}。"""
    out = _convert_ce_math(r"\ce{2[Ag(NH3)2]+ + 2OH-}")
    assert "[Ag(NH_3)_2]^{+}" in out
    assert "2OH^{-}" in out


def test_ce_charge_superscript_simple():
    """简单离子电荷：Fe2+ → Fe^{2+}、Cl- → Cl^{-}。"""
    out = _convert_ce_math(r"\ce{Fe2+ + Cl-}")
    assert "Fe^{2+}" in out
    assert "Cl^{-}" in out


def test_ce_precipitation_symbol_v():
    """mhchem 沉淀符号：孤立 v → \\downarrow（2Ag v → 2Ag↓）。"""
    out = _convert_ce_math(r"\ce{CH3CHO + 2[Ag(NH3)2]+ + 2OH- -> CH3COONH4 + 2Ag v + 3NH3 + H2O}")
    assert "2Ag \\downarrow" in out
    assert "CH_3COONH_4" in out


def test_ce_precipitation_unicode_downarrow():
    """Unicode ↓ 沉淀符号 → \\downarrow。"""
    out = _convert_ce_math(r"\ce{BaSO4 ↓}")
    assert "BaSO_4 \\downarrow" in out


def test_ce_v_not_element_symbol():
    """孤立 v 判定：不误伤元素符号中的 v（V2O5 大写 V、NaVO3 的 v 后跟字母）。"""
    assert _convert_ce_math(r"\ce{V2O5}") == "V_2O_5"
    assert _convert_ce_math(r"\ce{NaVO3}") == "NaVO_3"
    assert _convert_ce_math(r"\ce{2v}") != "2v"  # 无空格孤立 v 仍按沉淀处理
