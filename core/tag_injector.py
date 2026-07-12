# -*- coding: utf-8 -*-
"""core/tag_injector.py — 标记注入器。

把 LLM 原始响应中的渲染标记替换为已渲染的 TikZ 代码（或错误提示）。
按 TRANSITION 要求从后往前（start_pos 降序）替换，保证位置偏移不因前序替换而错乱。
"""

from typing import List

from core.tag_parser import RenderTag


def inject_tags_into_text(original_text: str, tags: List[RenderTag], rendered_map: dict) -> str:
    """将标记替换为渲染结果，返回最终图文文本。

    参数:
        original_text: LLM 原始响应（含标记）。
        tags: parse_tags 解析出的标记列表。
        rendered_map: {tag.raw: 渲染输出} 字典。

    返回:
        标记被替换后的文本。REASONING 去标记保留思考内容；未渲染的标记原样保留。
    """
    result = original_text
    # 从后往前替换：每次替换只影响该位置之后（已处理或无标记），不影响前序标记的偏移
    for tag in sorted(tags, key=lambda t: t.start_pos, reverse=True):
        if tag.type == "REASONING":
            # REASONING 无渲染器：去掉 [REASONING]/[/REASONING] 包裹，保留思考内容
            replacement = tag.args[0] if tag.args else ""
        else:
            replacement = rendered_map.get(tag.raw, tag.raw)
        result = result[:tag.start_pos] + replacement + result[tag.end_pos:]
    return result
