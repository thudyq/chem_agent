# -*- coding: utf-8 -*-
"""core/tag_parser.py — 标记解析器。

解析 LLM 输出中的渲染标记，提取为 RenderTag 列表，供渲染器分派。

支持的标记（TRANSITION Section 2）：
    [STRUCT:SMILES] 或 [STRUCT:SMILES,label=名称] （复合容器内可再加 ,id=引用名）
    [ARROW:反应物,产物,类型]
    [REACTION:反应物1;反应物2;...|产物1;产物2;...|反应条件]
    [REACTIONMECH:反应物1;...|产物1;...|条件|机理箭头]（可追加第五段 numbering 显示原子序号）
    [NEWMAN:SMILES,角度]
    [ENERGY:点序列]
    [RETRO:目标,前体,转化名]   （逆合成空心箭头 ⇒）
    [REASONING]...[/REASONING]   （配对标记）
    [COMPOSITE:布局]...[/COMPOSITE]   （容器式复合标记，见下方说明）

COMPOSITE 容器内允许的子标记：
    [STRUCT:SMILES,label=名称,id=r0]   结构组件；id 省略时按出现顺序自动编号 r0/r1/...
    [PLUS]                             加号连接符
    [RXNARROW] 或 [RXNARROW:条件]      主反应箭头（兼作反应物/产物分界）
    [CONDITION:文本]                   主反应箭头上方的条件文本
    [MECHARROW:src:atom>dst:atom]      机理弯箭头；>> 为鱼钩箭头
    [CHARGE:ref|idx:δ+,...]            组件 ref 上的部分电荷标注（R-2）
    [HBOND:ref|from-to,...]            组件 ref 内的氢键虚线（R-2）

COMPOSITE 解析结果为单个 RenderTag：type="COMPOSITE"，
args=[布局名, 子标记 RenderTag 列表]，raw 覆盖整个配对块。
容器内部的标记不会作为顶层标记重复出现。
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
    "REACTION": "[REACTION:",
    "REACTIONMECH": "[REACTIONMECH:",
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

# COMPOSITE 容器配对串
_COMPOSITE_OPEN = "[COMPOSITE:"
_COMPOSITE_CLOSE = "[/COMPOSITE]"

# COMPOSITE 容器内允许的带子标记 opener（冒号形式）
_INNER_OPENERS = {
    "STRUCT": "[STRUCT:",
    "MECHARROW": "[MECHARROW:",
    "CONDITION": "[CONDITION:",
    "RXNARROW": "[RXNARROW:",
    "CHARGE": "[CHARGE:",
    "HBOND": "[HBOND:",
    "ENERGY": "[ENERGY:",
}

# COMPOSITE 容器内允许的无参子标记（整串匹配）
_INNER_TOKENS = {
    "PLUS": "[PLUS]",
    "RXNARROW": "[RXNARROW]",
    "RESARROW": "[RESARROW]",
    "NEWLINE": "[NEWLINE]",
}


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
        smi, label = content, None
        li = content.find(",label=")
        ii = content.find(",id=")
        ai = content.find(",at=")
        pi = content.find(",pos=")
        cut = [p for p in (li, ii, ai, pi) if p != -1]
        if cut:
            smi = content[: min(cut)]
        if li != -1:
            lab_start = li + len(",label=")
            after = [p for p in (ii, ai, pi) if p != -1 and p > li]
            label = content[lab_start:min(after)] if after else content[lab_start:]
        return [smi, label]
    if tag_type == "MECH":
        parts = content.split("|", 2)
        while len(parts) < 3:
            parts.append("")
        return [p.strip() for p in parts]
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
    if tag_type == "REACTION":
        parts = content.split("|", 2)
        while len(parts) < 3:
            parts.append("")
        return [p.strip() for p in parts]
    if tag_type == "REACTIONMECH":
        parts = content.split("|", 4)
        while len(parts) < 5:
            parts.append("")
        return [p.strip() for p in parts]
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
    if tag_type in ("MECHARROW", "CONDITION", "RXNARROW"):
        return [content.strip()]
    return [content]  # ENERGY / 其他


def _in_regions(pos: int, regions: list) -> bool:
    """pos 是否落在任一 (start, end) 区域内。"""
    return any(start <= pos < end for start, end in regions)


def _parse_inner_tags(inner: str, base: int) -> List[RenderTag]:
    """解析 COMPOSITE 容器内部的子标记。

    参数:
        inner: 容器内文本（不含 [COMPOSITE:...] 与 [/COMPOSITE]）。
        base: inner 在原文中的起始下标，用于把子标记位置换算回原文坐标。
    """
    children = []
    for tag_type, opener in _INNER_OPENERS.items():
        search_from = 0
        while True:
            idx = inner.find(opener, search_from)
            if idx == -1:
                break
            end = _find_tag_end(inner, idx)
            if end == -1:
                search_from = idx + len(opener)
                continue
            raw = inner[idx:end + 1]
            content = raw[len(opener):-1]
            children.append(RenderTag(
                type=tag_type,
                args=_parse_content(tag_type, content),
                raw=raw,
                start_pos=base + idx,
                end_pos=base + end + 1,
            ))
            search_from = end + 1
    for tag_type, token in _INNER_TOKENS.items():
        search_from = 0
        while True:
            idx = inner.find(token, search_from)
            if idx == -1:
                break
            children.append(RenderTag(
                type=tag_type,
                args=[],
                raw=token,
                start_pos=base + idx,
                end_pos=base + idx + len(token),
            ))
            search_from = idx + len(token)
    children.sort(key=lambda t: t.start_pos)
    return children


def parse_tags(text: str) -> List[RenderTag]:
    """提取文本中所有渲染标记，按 start_pos 升序返回。无标记返回空列表。

    COMPOSITE 配对块解析为单个 COMPOSITE 标记（args=[布局, 子标记列表]），
    容器区域内的标记不再作为顶层标记返回。
    """
    if not text:
        return []
    tags = []
    composite_regions = []

    search_from = 0
    while True:
        idx = text.find(_COMPOSITE_OPEN, search_from)
        if idx == -1:
            break
        open_end = _find_tag_end(text, idx)
        if open_end == -1:
            search_from = idx + len(_COMPOSITE_OPEN)
            continue
        close_idx = text.find(_COMPOSITE_CLOSE, open_end + 1)
        if close_idx == -1:
            search_from = open_end + 1
            continue
        layout = text[idx + len(_COMPOSITE_OPEN):open_end].strip()
        inner_start = open_end + 1
        children = _parse_inner_tags(text[inner_start:close_idx], inner_start)
        end = close_idx + len(_COMPOSITE_CLOSE)
        tags.append(RenderTag(
            type="COMPOSITE",
            args=[layout, children],
            raw=text[idx:end],
            start_pos=idx,
            end_pos=end,
        ))
        composite_regions.append((idx, end))
        search_from = end

    for tag_type, opener in _OPENERS.items():
        search_from = 0
        while True:
            idx = text.find(opener, search_from)
            if idx == -1:
                break
            if _in_regions(idx, composite_regions):
                search_from = idx + len(opener)
                continue
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
        if _in_regions(m.start(), composite_regions):
            continue
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

    print()
    composite_demo = (
        "SN2 机理：\n"
        "[COMPOSITE:reaction_mech]\n"
        "[STRUCT:CCl,label=CH3Cl]\n"
        "[PLUS]\n"
        "[STRUCT:[OH-],id=nu,label=OH-]\n"
        "[RXNARROW]\n"
        "[STRUCT:CO,label=CH3OH]\n"
        "[PLUS]\n"
        "[STRUCT:[Cl-],label=Cl-]\n"
        "[MECHARROW:nu:0>r0:0]\n"
        "[MECHARROW:r0:0-1>r0:1]\n"
        "[CONDITION:SN2]\n"
        "[/COMPOSITE]\n"
        "后续文字 [STRUCT:C]"
    )
    tags = parse_tags(composite_demo)
    print(f"复合标记解析到 {len(tags)} 个顶层标记:")
    for t in tags:
        print(f"  {t.type}: layout/args={t.args[0] if t.type != 'COMPOSITE' else t.args[0]}")
        if t.type == "COMPOSITE":
            for c in t.args[1]:
                print(f"    子标记 {c.type}: args={c.args}")
