# -*- coding: utf-8 -*-
"""tests/test_readback.py — core/readback.py 回读摘要与身份保持测试。"""

import pytest

pytest.importorskip("rdkit", reason="rdkit 未安装，跳过回读摘要测试")

from core.readback import (identity_drift, is_high_risk, summarize_species,
                           summarize_tag)
from core.tag_parser import parse_tags


def _tag(text):
    return parse_tags(text)[0]


def test_summarize_species_ring_and_charge():
    """溴鎓四元环病例：摘要须给出环尺寸与带电原子，
    使"声称三元环、实为四元环"在回读中可见。"""
    s = summarize_species("[Br+]1CC(C)C1C", label="环状溴鎓离子", cid="s0")
    assert "s0" in s and "环状溴鎓离子" in s
    assert "4 元环" in s and "Br" in s
    assert "净电荷 +1" in s
    # 正确三元环版本的摘要对照
    s3 = summarize_species("[Br+]1CC1")
    assert "3 元环" in s3


def test_summarize_species_ion_pair():
    """离子对：净电荷为零但带电原子明细须暴露 Na+ 与 C- 两个组分。"""
    s = summarize_species("[Na+].[C-]#C", label="乙炔钠")
    assert "净电荷 0" in s
    assert "Na" in s and "+1" in s and "-1" in s


def test_summarize_species_chiral_and_radical():
    """手性中心计数与 R/S 分布（集合级，不带原子序——原子序会诱使模型
    与 label 的 IUPAC 位次混淆，把对的结构翻成对映体）；
    自由基单电子总数。"""
    s = summarize_species("CC[C@H](Cl)[C@@H](C)Cl", label="(2R,3S)-2,3-二氯戊烷")
    assert "手性中心 2（R×1、S×1）" in s
    s2 = summarize_species("[CH](C)c1ccccc1", label="苄基自由基")
    assert "自由基单电子 1" in s2


def test_summarize_species_coeff_and_formula_text():
    """系数前缀剥除（2CC=O → 乙醛）；化学式文本组件产出说明而非伪事实。"""
    s = summarize_species("2CC=O")
    assert "C2H4O" in s and "重原子 3" in s
    s2 = summarize_species("KMnO4")
    assert "无原子结构" in s2 or "无法解析" in s2


def test_summarize_tag_composite_with_block():
    """COMPOSITE（含 BLOCK 嵌套）逐组件一行摘要，cid 齐全。"""
    t = _tag(
        "[COMPOSITE:reaction][BLOCK]"
        "[STRUCT:[CH](C)c1ccccc1,id=a,label=苄位自由基]"
        "[ARROW:type=resonance]"
        "[STRUCT:CC=C1[CH]C=CC=C1,id=b,label=邻位共振式]"
        "[/BLOCK][/COMPOSITE]")
    out = summarize_tag(t)
    assert "a" in out and "b" in out
    assert "自由基单电子 1" in out
    # 无 STRUCT 的标记（ENERGY/REASONING）→ 空摘要
    assert summarize_tag(_tag("[ENERGY:0,108,-20]")) == ""


def test_identity_drift_keeps_skeleton_level_fixes():
    """身份保持：补漏写的端位原子（σ 络合物补 Cl）、质子化/电荷修正不改骨架
    → 不报漂移。"""
    # σ 络合物补上漏写的 Cl（端位取代，BM 骨架不变）
    assert identity_drift(
        "[STRUCT:COC1([H])C=CC(=[N+]([O-])[O-])C=C1,id=s,label=σ络合物]",
        "[STRUCT:COC1(Cl)C=CC(=[N+]([O-])[O-])C=C1,id=s,label=σ络合物]",
    ) == []
    # 质子化修正（无环物种：拓扑骨架不变，电荷抹平）
    assert identity_drift(
        "[STRUCT:CCOCC,id=e,label=乙醚]",
        "[STRUCT:CC[OH+]CC,id=e,label=质子化乙醚]",
    ) == []
    # 原样 → 无漂移
    assert identity_drift("[STRUCT:c1ccccc1,id=b]", "[STRUCT:c1ccccc1,id=b]") == []


def test_identity_drift_flags_skeleton_changes():
    """漂移判定：环尺寸改变（溴鎓四元环→三元环）、碳链改变、
    物种数变化 → 报漂移。"""
    # 环尺寸变化：四元环 → 三元环（BM 骨架变化）
    drift = identity_drift(
        "[STRUCT:[Br+]1CC(C)C1C,id=x,label=环状溴鎓离子]",
        "[STRUCT:CC1[Br+]C1C,id=x,label=环状溴鎓离子]")
    assert drift and "x" in drift[0]
    # 乙醚 → 丙醚（无环物种拓扑骨架变化）
    assert identity_drift("[STRUCT:CCO,id=a]", "[STRUCT:CCCO,id=a]")
    # 物种数变化
    assert identity_drift(
        "[COMPOSITE:reaction][STRUCT:CCO,id=a][PLUS][STRUCT:O,id=b]"
        "[/COMPOSITE]",
        "[COMPOSITE:reaction][STRUCT:CCO,id=a][/COMPOSITE]")


def test_is_high_risk():
    """高风险界定：机理 COMPOSITE / 盐 / 电荷自由基 / 环系 / 立体 → True；
    平面简单小分子、REASONING、ENERGY → False。"""
    assert not is_high_risk(_tag("[STRUCT:C,label=甲烷]"))
    assert not is_high_risk(_tag("[STRUCT:CCO,label=乙醇]"))
    assert is_high_risk(_tag("[STRUCT:c1ccccc1,label=苯]"))           # 环系
    assert is_high_risk(_tag("[STRUCT:[Na+].[C-]#C,label=乙炔钠]"))    # 盐/电荷
    assert is_high_risk(_tag("[STRUCT:[CH3],label=甲基自由基]"))       # 自由基
    assert is_high_risk(_tag("[STRUCT:C/C=C/C,label=反-2-丁烯]"))      # 立体
    assert is_high_risk(_tag("[STRUCT:CC1CCCCC1C,mode=chair]"))        # 画法
    assert is_high_risk(_tag("[STRUCT:CCO,label=某种环状溴鎓]"))       # label 声明
    assert is_high_risk(_tag(
        "[COMPOSITE:reaction][STRUCT:CCO,id=a][ARROW:type=single]"
        "[STRUCT:CC=O,id=b][MECHARROW:a:0-1>b:0][/COMPOSITE]"))
    assert not is_high_risk(_tag("[REASONING]随便想想[/REASONING]"))
    assert not is_high_risk(_tag("[ENERGY:0,108,-20]"))
