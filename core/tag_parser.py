# -*- coding: utf-8 -*-
"""core/tag_parser.py — 标记解析器。

解析 LLM 输出中的渲染标记，提取为 RenderTag 列表，供渲染器分派。

支持的标记（TRANSITION Section 2）：
    [STRUCT:SMILES] 或 [STRUCT:SMILES,label=名称]
    [ARROW:反应物,产物,类型]
    [NEWMAN:SMILES,角度]
    [ENERGY:点序列]
    [REASONING]...[/REASONING]   （配对标记）
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class RenderTag:
    """单个渲染标记的解析结果。"""
    type: str        # STRUCT / ARROW / NEWMAN / ENERGY / REASONING
    args: list       # 捕获组（按标记格式排列；可选组未命中为 None）
    raw: str         # 原始匹配文本，如 "[STRUCT:c1ccccc1]"
    start_pos: int   # 在原文中的起始下标
    end_pos: int     # 结束下标（ exclusive）


# 各标记的编译正则。STRUCT 的 SMILES 组用 [^,\]] 排除逗号与 ]，
# 避免贪婪吞掉 ,label= 部分（TRANSITION 原始正则的 bug 修正）。
_PATTERNS = [
    ("STRUCT", re.compile(r"\[STRUCT:([^,\]]+)(?:,label=([^\]]+))?\]")),
    ("ARROW", re.compile(r"\[ARROW:([^,]+),([^,]+),([^\]]+)\]")),
    ("NEWMAN", re.compile(r"\[NEWMAN:([^,]+),([^\]]+)\]")),
    ("ENERGY", re.compile(r"\[ENERGY:([^\]]+)\]")),
    # REASONING 配对，DOTALL 允许内容跨行
    ("REASONING", re.compile(r"\[REASONING\](.*?)\[/REASONING\]", re.DOTALL)),
]


def parse_tags(text: str) -> List[RenderTag]:
    """提取文本中所有渲染标记，按 start_pos 升序返回。

    参数:
        text: LLM 输出的原始文本。

    返回:
        List[RenderTag]：无标记时返回空列表。
    """
    if not text:
        return []
    tags = []
    for tag_type, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            # 捕获组转为 list，未命中的可选组保留为 None
            args = [g if g is not None else None for g in m.groups()]
            tags.append(RenderTag(
                type=tag_type,
                args=args,
                raw=m.group(0),
                start_pos=m.start(),
                end_pos=m.end(),
            ))
    tags.sort(key=lambda t: t.start_pos)
    return tags


if __name__ == "__main__":
    # 快速自测：python core/tag_parser.py
    demo = "[STRUCT:c1ccccc1] 和 [ARROW:c1ccccc1,c1ccccc1N,amination]"
    tags = parse_tags(demo)
    print(f"输入: {demo}")
    print(f"解析到 {len(tags)} 个标记:")
    for t in tags:
        print(f"  {t.type}: args={t.args} raw={t.raw!r} pos=[{t.start_pos},{t.end_pos})")
