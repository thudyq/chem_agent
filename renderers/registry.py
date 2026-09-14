# -*- coding: utf-8 -*-
"""renderers/registry.py — 渲染器注册表。

每种 [TAG] 标记对应一个渲染函数，在此注册供主流程分派。
分子家族（lewis/stereo/chair/newman）统一为 STRUCT+mode 由
render_structure 分派（旧标记解析入口已于 B3 清理移除）。
"""

from .structure import render_structure
from .composite import render_composite
from .energy import render_energy

RENDERER_REGISTRY = {
    "STRUCT": render_structure,
    "COMPOSITE": render_composite,
    "ENERGY": render_energy,
}


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
