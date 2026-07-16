# -*- coding: utf-8 -*-
"""core/tag_parser.py — 标记解析器。

解析 LLM 输出中的渲染标记，提取为 RenderTag 列表，供渲染器分派。

支持的标记（TRANSITION Section 2）：
    [STRUCT:SMILES] 或 [STRUCT:SMILES,label=名称]
    [ARROW:反应物,产物,类型]
    [NEWMAN:SMILES,角度]
    [ENERGY:点序列]
    [RETRO:目标,前体,转化名]   （逆合成空心箭头 ⇒）
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


# 单标记类型的开启串
_OPENERS = {
    "STRUCT": "[STRUCT:",
    "ARROW": "[ARROW:",
    "NEWMAN": "[NEWMAN:",
    "ENERGY": "[ENERGY:",
    "LEWIS": "[LEWIS:",
    "STEREO": "[STEREO:",
    "MECH": "[MECH:",
    "CHARGE": "[CHARGE:",
    "RESONANCE": "[RESONANCE:",
    "HBOND": "[HBOND:",
    "RETRO": "[RETRO:",
}

# REASONING 配对正则（内容不与括号冲突，可用正则）
_REASONING_RE = re.compile(r"\[REASONING\](.*?)\[/REASONING\]", re.DOTALL)


def _find_tag_end(text: str, start: int) -> int:
    """从 start（指向 opener 的 [）用括号深度平衡扫描，返回配对 ] 下标；未闭合返回 -1。

    兼容 SMILES 括号原子 [N+]/[O-]/[C@@H]：其内部 ] 不会提前闭合标记。
    """
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _parse_content(tag_type: str, content: str) -> list:
    """把标记内部内容拆为参数列表。SMILES 不含逗号，按逗号切分安全。"""
    if tag_type == "STRUCT":
        if ",label=" in content:
            smi, _, label = content.partition(",label=")
            return [smi, label]
        return [content, None]
    if tag_type == "MECH":
        if "|" in content:
            smi, _, arrows = content.partition("|")
            return [smi.strip(), arrows.strip()]
        return [content.strip(), ""]
    if tag_type == "CHARGE":
        if "|" in content:
            smi, _, charges = content.partition("|")
            return [smi.strip(), charges.strip()]
        return [content.strip(), ""]
    if tag_type == "HBOND":
        if "|" in content:
            smi, _, pairs = content.partition("|")
            return [smi.strip(), pairs.strip()]
        return [content.strip(), ""]
    if tag_type == "ARROW":
        parts = content.split(",", 2)
        while len(parts) < 3:
            parts.append("")
        return parts
    if tag_type == "RETRO":
        parts = content.split(",", 2)
        while len(parts) < 3:
            parts.append("")
        return parts
    if tag_type == "NEWMAN":
        parts = content.split(",", 1)
        if len(parts) == 1:
            parts.append("")
        return parts
    return [content]  # ENERGY / 其他


def parse_tags(text: str) -> List[RenderTag]:
    """提取文本中所有渲染标记，按 start_pos 升序返回。无标记返回空列表。"""
    if not text:
        return []
    tags = []
    for tag_type, opener in _OPENERS.items():
        search_from = 0
        while True:
            idx = text.find(opener, search_from)
            if idx == -1:
                break
            end = _find_tag_end(text, idx)
            if end == -1:
                search_from = idx + len(opener)
                continue
            raw = text[idx:end + 1]
            content = raw[len(opener):-1]  # 去掉 "[TAG:" 与 "]"
            tags.append(RenderTag(
                type=tag_type,
                args=_parse_content(tag_type, content),
                raw=raw,
                start_pos=idx,
                end_pos=end + 1,
            ))
            search_from = end + 1
    for m in _REASONING_RE.finditer(text):
        tags.append(RenderTag(
            type="REASONING",
            args=[m.group(1)],
            raw=m.group(0),
            start_pos=m.start(),
            end_pos=m.end(),
        ))
    tags.sort(key=lambda t: t.start_pos)
    return tags


if __name__ == "__main__":
    demo = "[STRUCT:c1ccccc1] 和 [STRUCT:O=[N+]([O-])c1ccccc1] 然后 [ARROW:c1ccccc1,c1ccccc1N,amination]"
    tags = parse_tags(demo)
    print(f"解析到 {len(tags)} 个标记:")
    for t in tags:
        print(f"  {t.type}: args={t.args}")
