# -*- coding: utf-8 -*-
"""renderers/base.py — 渲染器注册表。

每种 [TAG] 标记对应一个渲染函数，在此注册供主流程分派。
后续 Day 按需新增 ARROW / NEWMAN / ENERGY 渲染器。
"""

from renderers.structure import render_structure

RENDERER_REGISTRY = {
    "STRUCT": render_structure,
    # "ARROW": render_arrow,    # Day 10-11 实现
    # "NEWMAN": render_newman,  # Day 15-16 实现（可选）
    # "ENERGY": render_energy,  # 后续实现
}


def get_renderer(tag_type: str):
    """按标记类型取渲染函数；未注册返回 None。"""
    return RENDERER_REGISTRY.get(tag_type)
