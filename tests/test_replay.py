# -*- coding: utf-8 -*-
"""tests/test_replay.py — 离线重放工具（core/replay.py）测试。

运行: python -m pytest tests/test_replay.py -v
"""

import pytest

from core.replay import replay

# Q17 病例：σ 络合物脱质子箭头写错（文字/产物对、箭头乱写）
_Q17_BAD = ("[COMPOSITE:reaction]"
            "[STRUCT:BrC([H])1C=CC=C[CH+]1,label=σ 络合物,id=sigma]"
            "[ARROW:type=single]"
            "[STRUCT:BrC1=CC=CC=C1,label=溴苯][PLUS][STRUCT:[H+],label=H+]"
            "[MECHARROW:sigma:1-2>sigma:1][MECHARROW:sigma:7-1>sigma:7]"
            "[/COMPOSITE]")
_Q17_OK = _Q17_BAD.replace("sigma:1-2>sigma:1]", "sigma:1-2>sigma:1-7]") \
                  .replace("[MECHARROW:sigma:7-1>sigma:7]", "")


def test_replay_plain_text(capsys):
    assert replay("没有任何标记的纯文本。") == 0
    assert "未解析到任何标记" in capsys.readouterr().out


def test_replay_pass_valid_composite(capsys):
    pytest.importorskip("rdkit")
    assert replay(_Q17_OK) == 0
    out = capsys.readouterr().out
    assert "判定: ✓ 通过" in out
    assert "EAS σ 脱质子方向  PASS" in out
    assert "reaction 守恒  PASS" in out
    assert "渲染: ✓ 成功" in out


def test_replay_trace_pinpoints_failing_rule(capsys):
    """拦截案例：trace 面板精确指出哪条规则开枪（EAS/消除 FAIL、
    配对/SN2/守恒 PASS）——归因 Case B 的核心能力。"""
    pytest.importorskip("rdkit")
    assert replay(_Q17_BAD) == 1
    out = capsys.readouterr().out
    assert "判定: ✗ 拦截" in out
    assert "EAS σ 脱质子方向  FAIL" in out
    assert "消除成 π 键方向    FAIL" in out
    assert "质子转移配对       PASS" in out
    assert "reaction 守恒  PASS" in out
    assert "校验未过，管线中不会进渲染器" in out


def test_replay_struct_label_trace(capsys):
    """STRUCT 标签 trace：仲/叔丁基拓扑错误由中文名检查一项指出。"""
    pytest.importorskip("rdkit")
    assert replay("[STRUCT:C[C+](C)C,label=2-丁基碳正离子]") == 1
    out = capsys.readouterr().out
    assert "中文名一致性（含拓扑）  FAIL" in out
    assert "直链骨架" in out
    assert "质子化一致性       PASS" in out
    assert "自由基一致性       PASS" in out


def test_replay_struct_pass(capsys):
    pytest.importorskip("rdkit")
    assert replay("[STRUCT:CC[CH+]C,label=仲丁基碳正离子]") == 0
    out = capsys.readouterr().out
    assert "判定: ✓ 通过" in out
    assert "中文名一致性（含拓扑）  PASS" in out


def test_replay_mixed_counts(capsys):
    """多标记混合：汇总计数正确，退出码反映任一拦截。"""
    pytest.importorskip("rdkit")
    text = ("[STRUCT:C[C+](C)C,label=2-丁基碳正离子] 和 "
            "[STRUCT:CC[CH+]C,label=仲丁基碳正离子]")
    assert replay(text) == 1
    out = capsys.readouterr().out
    assert "通过 1，拦截 1" in out
