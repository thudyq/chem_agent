# -*- coding: utf-8 -*-
"""P3 布局升级测试：字符宽度表与矩形相交（纯函数，无 rdkit 依赖）。

- label_visual_width：区分全角/半角，西文宽度与旧估算一致、中文修正；
- _rects_intersect：energy 布局 at= 冲突消解的基础。
"""

from renderers.composite import _rects_intersect
from renderers.mol_primitives import _is_wide_char, label_visual_width


def test_west_char_width_unchanged():
    assert abs(label_visual_width("CH3Cl") - 1.30) < 1e-6


def test_cjk_char_width_doubled():
    assert abs(label_visual_width("质子化的乙醇") - 3.12) < 1e-6


def test_mixed_width():
    assert abs(label_visual_width("乙醇 OH") - 1.82) < 1e-6


def test_wide_char_detection():
    assert _is_wide_char("乙")
    assert _is_wide_char("，")
    assert not _is_wide_char("C")
    assert not _is_wide_char("3")


def test_label_ignores_latex_markup():
    assert abs(label_visual_width("CH$_{3}$")
               - label_visual_width("CH3")) < 1e-6


def test_rect_intersect():
    a = (0, 0, 1, 1)
    assert _rects_intersect(a, (0.5, 0.5, 1.5, 1.5))
    assert not _rects_intersect(a, (2, 0, 3, 1))


def test_rect_intersect_with_pad():
    a = (0, 0, 1, 1)
    assert _rects_intersect(a, (0.9, 0.9, 2, 2), pad=0.15)


def test_rect_no_intersect_beyond_pad():
    a = (0, 0, 1, 1)
    assert not _rects_intersect(a, (1.3, 0, 2.3, 1))
