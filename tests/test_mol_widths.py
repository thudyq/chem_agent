# -*- coding: utf-8 -*-
"""P3 布局升级测试：字符宽度表、标签自动换行与矩形相交（纯函数，无 rdkit 依赖）。

- label_visual_width：区分全角/半角，西文宽度与旧估算一致、中文修正；
- wrap_label_lines / wrap_format_text / label_wrapped_size：C2 标签自动换行；
- _rects_intersect：energy 布局 at= 冲突消解的基础。
"""

from renderers.composite import _rects_intersect
from renderers.mol_primitives import (
    _is_wide_char, label_visual_width, label_wrapped_size, wrap_format_text,
    wrap_label_lines,
)


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


# ---------------------------------------------------------------------------
# C2 标签自动换行
# ---------------------------------------------------------------------------

def test_wrap_short_label_single_line():
    """短标签（≤ 默认 3.5 宽）原样单行返回。"""
    assert wrap_label_lines("CH3Cl") == ["CH3Cl"]
    assert wrap_label_lines("质子化乙醇") == ["质子化乙醇"]   # 5 字 = 2.6 宽


def test_wrap_long_cjk_breaks_by_width():
    """长中文标签按可视宽度（每字 0.52）断行。"""
    lines = wrap_label_lines("质子化乙醇的反应中间体")   # 10 字 = 5.2 宽
    assert len(lines) >= 2
    assert "".join(lines) == "质子化乙醇的反应中间体"    # 内容完整无丢失
    for ln in lines:
        assert label_visual_width(ln) <= 3.5 + 0.6      # 每行不超宽（单字宽允许 0.52）


def test_wrap_breaks_at_space_preferred():
    """断行优先落在空格处（保持单词完整）。"""
    lines = wrap_label_lines("浓H2SO4, 加热至回流")     # 含空格
    assert all(" " not in ln.strip() or ln.strip().endswith(",") for ln in lines)


def test_wrap_format_text_multiline_joins():
    """多行标签以 \\\\ 连接；每行独立化学排版（下标/加热符号）。"""
    out = wrap_format_text("质子化乙醇的反应中间体")
    assert "\\\\" in out
    assert out.count("\\\\") == len(wrap_label_lines("质子化乙醇的反应中间体")) - 1


def test_wrap_format_text_short_unchanged():
    """短文本等价 format_chem_text（无 \\\\）。"""
    assert wrap_format_text("H2SO4") == "H$_2$SO$_4$"
    assert "\\\\" not in wrap_format_text("CuO, △")


def test_label_wrapped_size_returns_width_height():
    """换行后 (总宽, 总高)：高 = 行数 × 行高 0.35。"""
    w, h = label_wrapped_size("质子化乙醇的反应中间体")
    assert w > 0 and h > 0
    assert abs(h - 0.35 * len(wrap_label_lines("质子化乙醇的反应中间体"))) < 1e-9


def test_wrap_format_text_name_locant_no_superscript():
    """Drawbacks bug：中文化学名含位次号（3-溴-1-甲基环己烯）被换行成
    "3-溴-1-" 行尾时，位次号 "1-" 不得被误判为电荷上标（$^{1-}$）。"""
    out = wrap_format_text("3-溴-1-甲基环己烯")
    assert "$^{1-}$" not in out
    assert "$^{-}$" not in out          # 不应引入任何电荷上标
    assert "3-溴-1-" in out and "甲基环己烯" in out   # 名称内容完整保留
