# -*- coding: utf-8 -*-
"""tests/test_energy_roles.py — 能量点角色标注单元测试（Drawbacks 第 2 条规范）。

规范：两步反应需标注 反应中间体 与 第二步反应的过渡态。
运行: python -m pytest tests/test_energy_roles.py -v
"""

import re

import pytest

from renderers.energy import render_energy
from renderers.layout import energy_point_roles


def test_roles_two_step():
    """5 点两步反应：反应物/过渡态×2/反应中间体/产物。"""
    roles = energy_point_roles([0, 80, -10, 60, -30])
    assert roles[0] == "反应物"
    assert roles[1] == "过渡态"
    assert roles[2] == "反应中间体"
    assert roles[3] == "过渡态"
    assert roles[4] == "产物"


def test_roles_single_step():
    """3 点单步反应：反应物/过渡态/产物，无中间体。"""
    roles = energy_point_roles([0, 108, -20])
    assert roles == {0: "反应物", 1: "过渡态", 2: "产物"}


def test_roles_plateau_unlabeled():
    """平台段（与相邻等值）不标注。"""
    roles = energy_point_roles([0, 50, 50, 20])
    assert 1 not in roles and 2 not in roles


def test_energy_two_step_labels():
    """render_energy 两步反应：中间体与第二个过渡态均有能量标签。"""
    out = render_energy("0,80,-10,60,-30")
    assert "反应中间体 (-10)" in out
    assert out.count("过渡态 (+80)") == 1
    assert "过渡态 (+60)" in out
    assert "反应物 (+0)" in out
    assert "产物 (-30)" in out


def test_energy_two_step_label_positions():
    """过渡态标签在点上方（+0.35），中间体/反应物/产物在下方（-0.3）。"""
    out = render_energy("0,80,-10,60,-30")
    above = re.findall(r"at \(0,0.35\) \{([^}]+)\}", out)
    below = re.findall(r"at \(0,-0.30\) \{([^}]+)\}", out)
    assert len(above) == 2                          # 两个过渡态
    assert all("过渡态" in t for t in above)
    assert any("反应中间体" in t for t in below)
    assert len(below) == 3                          # 反应物/中间体/产物


def test_energy_single_step_unchanged():
    """单步反应标签行为不变。"""
    out = render_energy("0,108,-20")
    assert "反应物 (+0)" in out
    assert "过渡态 (+108)" in out
    assert "产物 (-20)" in out
    assert "中间体" not in out
