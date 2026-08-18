# -*- coding: utf-8 -*-
"""renderers/registry.py — 渲染器注册表。

每种 [TAG] 标记对应一个渲染函数，在此注册供主流程分派。
分子家族重构（20260818）：LEWIS/STEREO/CHAIR/NEWMAN 由解析层归一化为
STRUCT+mode，统一走 render_structure 分派；旧注册保留兼容直接调用。
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


def render_tag(tag) -> str:
    """按标记渲染（统一入口，替代各处 `renderer(*tag.args)`）。

    STRUCT 是分子家族统一入口：mode/subs/bond/angle 等参数在 tag.attrs
    （解析层归一化后 LEWIS/STEREO/CHAIR/NEWMAN 也变成 STRUCT+mode），
    按位置传给渲染器（兼容 registry mock 的 lambda *a 签名）。
    其余标记按 args 透传。返回 None 表示无渲染器/REASONING。
    """
    renderer = RENDERER_REGISTRY.get(tag.type)
    if renderer is None or tag.type == "REASONING":
        return None
    if tag.type == "STRUCT":
        return renderer(
            tag.args[0] if tag.args else "",
            tag.args[1] if len(tag.args) > 1 else None,
            tag.attrs.get("mode", "skeleton"),
            tag.attrs.get("subs", ""),
            tag.attrs.get("bond", ""),
            tag.attrs.get("angle", ""),
            tag.attrs.get("charge", ""),
        )
    return renderer(*tag.args)
