# -*- coding: utf-8 -*-
"""renderers/registry.py — 渲染器注册表。

每种 [TAG] 标记对应一个渲染函数，在此注册供主流程分派。
后续 Day 按需新增 ARROW / NEWMAN / ENERGY 渲染器。
"""

from .structure import render_structure
from .arrow import render_arrow
from .reaction import render_reaction
from .composite import render_composite
from .newman import render_newman
from .lewis import render_lewis
from .energy import render_energy
from .stereo import render_stereo
from .charge import render_charge
from .retro import render_retro
from .xh_bond import render_bond, render_xh
from .chair import render_chair

RENDERER_REGISTRY = {
    "STRUCT": render_structure,
    "ARROW": render_arrow,
    "REACTION": render_reaction,
    "COMPOSITE": render_composite,
    "NEWMAN": render_newman,
    "LEWIS": render_lewis,
    "ENERGY": render_energy,
    "STEREO": render_stereo,
    "CHARGE": render_charge,
    "RETRO": render_retro,
    "XH": render_xh,
    "BOND": render_bond,
    "CHAIR": render_chair,
}


def get_renderer(tag_type: str):
    """按标记类型取渲染函数；未注册返回 None。"""
    return RENDERER_REGISTRY.get(tag_type)
