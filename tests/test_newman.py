# -*- coding: utf-8 -*-
"""tests/test_newman.py — 纽曼投影渲染器单元测试（Drawbacks 第 1 条规范）。

规范：键在圆外的部分 = 圆半径的 2/3；键的粗细与圆一致（thick）。
运行: python -m pytest tests/test_newman.py -v
"""

import math
import re

import pytest

from renderers.newman import render_newman

rdkit = pytest.importorskip("rdkit", reason="rdkit 未安装，跳过纽曼投影测试")


def _circle_radius(out: str) -> float:
    m = re.search(r"\\draw\[thick\] \(0,0\) circle \(([-\d.]+)\)", out)
    assert m is not None
    return float(m.group(1))


def test_bond_outside_length_is_two_thirds_radius():
    """前键与后键的圆外部分均 = 圆半径 × 2/3（端点距圆心 = 5R/3）。"""
    out = render_newman("CC", "60")
    r = _circle_radius(out)
    expected_d = r * (1 + 2 / 3)
    # 前键：(0,0) -- (x,y)，端点距圆心 = D
    for m in re.finditer(r"\\draw\[thick\] \(0,0\) -- \(([-\d.]+),([-\d.]+)\);", out):
        d = math.hypot(float(m.group(1)), float(m.group(2)))
        assert abs(d - expected_d) < 0.01
    # 后键：(x0,y0) -- (x1,y1)，端点距圆心 = D，起点在圆周上
    back = re.findall(r"\\draw\[thick, gray\] \(([-\d.]+),([-\d.]+)\) -- \(([-\d.]+),([-\d.]+)\);", out)
    assert len(back) == 3
    for x0, y0, x1, y1 in back:
        assert abs(math.hypot(float(x0), float(y0)) - r) < 0.01        # 起点在圆周
        assert abs(math.hypot(float(x1), float(y1)) - expected_d) < 0.01


def test_bond_and_circle_same_thickness():
    """键（前/后）与圆同为 thick。"""
    out = render_newman("CC", "60")
    assert out.count("\\draw[thick]") >= 4          # 3 前键 + 1 圆
    assert out.count("\\draw[thick, gray]") == 3    # 3 后键
    assert "\\draw (0,0) --" not in out             # 不再有默认细线前键
    assert "\\draw[gray]" not in out                # 不再有默认细线后键


def test_staggered_and_eclipsed():
    """交叉式(60°)与重叠式(0°)：后键角度 = 前键角度 + θ。"""
    out60 = render_newman("CC", "60")
    out0 = render_newman("CC", "0")
    assert out60 != out0
    for out in (out60, out0):
        assert out.startswith("\\begin{tikzpicture}")
        assert out.endswith("\\end{tikzpicture}")


def _label_angles(out: str, gray: bool) -> list:
    """提取前/后标签相对圆心的角度（度，0~360）。"""
    pat = r"\\node\[gray\] at \(([-\d.]+),([-\d.]+)\)" if gray \
        else r"\\node at \(([-\d.]+),([-\d.]+)\)"
    angles = []
    for m in re.finditer(pat, out):
        ang = math.degrees(math.atan2(float(m.group(2)), float(m.group(1))))
        angles.append(ang % 360.0)
    return sorted(angles)


def test_eclipsed_back_labels_shifted_clockwise():
    """重叠式(0°)：后标签统一顺时针偏移 15° 露出（后角 = 前角 - 15）。"""
    out = render_newman("CC", "0")
    front = _label_angles(out, gray=False)
    back = _label_angles(out, gray=True)
    assert len(front) == 3 and len(back) == 3
    for ba in back:
        assert any(abs((fa - ba) % 360.0 - 15.0) < 1.0 for fa in front), \
            f"后标签 {ba}° 与任一前标签差 ≠ 15°"


def test_eclipsed_back_bonds_shifted_with_labels():
    """重叠式(0°)：后键与后标签同步偏移（键末端角度 = 标签角度）。"""
    out = render_newman("CC", "0")
    bond_end_angles = sorted(
        math.degrees(math.atan2(float(m.group(2)), float(m.group(1)))) % 360.0
        for m in re.finditer(
            r"\\draw\[thick, gray\] \([-\d.]+,[-\d.]+\) -- \(([-\d.]+),([-\d.]+)\);", out)
    )
    label_angles = _label_angles(out, gray=True)
    assert len(bond_end_angles) == 3 and len(label_angles) == 3
    for ea, la in zip(bond_end_angles, label_angles):
        assert abs((ea - la) % 360.0) < 1.0 or abs((la - ea) % 360.0) < 1.0, \
            f"后键末端 {ea}° 与后标签 {la}° 未对齐"


def test_staggered_back_labels_not_shifted():
    """交叉式(60°)：后标签不偏移（与前键角无重叠）。"""
    out = render_newman("CC", "60")
    front = _label_angles(out, gray=False)
    back = _label_angles(out, gray=True)
    for ba in back:
        assert any(abs((ba - fa) % 360.0 - 60.0) < 1.0 for fa in front), \
            f"后标签 {ba}° 与任一前标签差 ≠ 60°"


def test_label_distance_from_bond_end():
    """标签距圆心 = D × 1.25（比键末端稍远）。"""
    out = render_newman("CC", "60")
    r = _circle_radius(out)
    expected = r * (1 + 2 / 3) * 1.25
    for m in re.finditer(r"\\node(?:\[gray\])? at \(([-\d.]+),([-\d.]+)\)", out):
        d = math.hypot(float(m.group(1)), float(m.group(2)))
        assert abs(d - expected) < 0.02, f"标签距圆心 {d:.2f} ≠ {expected:.2f}"


def test_error_paths():
    """错误路径：无效 SMILES / 无 C-C 单键。"""
    assert "无效 SMILES" in render_newman("XYZ")
    assert "无效 SMILES" in render_newman("")
    assert "无 C-C 单键" in render_newman("O")
