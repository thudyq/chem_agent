# -*- coding: utf-8 -*-
"""renderers/base.py — 渲染器注册表。

每种 [TAG] 标记对应一个渲染函数，在此注册供主流程分派。
后续 Day 按需新增 ARROW / NEWMAN / ENERGY 渲染器。
"""

from renderers.structure import render_structure
from renderers.arrow import render_arrow
from renderers.newman import render_newman
from renderers.lewis import render_lewis
from renderers.energy import render_energy
from renderers.stereo import render_stereo
from renderers.mechanism import render_mechanism
from renderers.charge import render_charge
from renderers.resonance import render_resonance
from renderers.hbond import render_hbond

RENDERER_REGISTRY = {
    "STRUCT": render_structure,
    "ARROW": render_arrow,
    "NEWMAN": render_newman,
    "LEWIS": render_lewis,
    "ENERGY": render_energy,
    "STEREO": render_stereo,
    "MECH": render_mechanism,
    "CHARGE": render_charge,
    "RESONANCE": render_resonance,
    "HBOND": render_hbond,
}


def get_renderer(tag_type: str):
    """按标记类型取渲染函数；未注册返回 None。"""
    return RENDERER_REGISTRY.get(tag_type)
