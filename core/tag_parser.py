# -*- coding: utf-8 -*-
"""core/tag_parser.py — 标记解析器。

解析 LLM 输出中的渲染标记，提取为 RenderTag 列表，供渲染器分派。

支持的标记（TRANSITION Section 2）：
    [STRUCT:SMILES] 或 [STRUCT:SMILES,label=名称] （复合容器内可再加 ,id=引用名）
    [ARROW:反应物,产物,类型]
    [REACTION:反应物1;反应物2;...|产物1;产物2;...|反应条件]
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
from dataclasses import dataclass, field
from typing import List


@dataclass
class RenderTag:
    """单个渲染标记的解析结果。"""
    type: str        # STRUCT / ARROW / ENERGY / REASONING 等标记类型
    args: list       # 捕获组（按标记格式排列；可选组未命中为 None）
    raw: str         # 原始匹配文本，如 "[STRUCT:c1ccccc1]"
    start_pos: int   # 在原文中的起始下标
    end_pos: int     # 结束下标（ exclusive）
    # STRUCT 的结构化属性（id/at/pos）——由解析器一次性提取，
    # composite/tag_validator 直接读取，避免从 raw 二次正则解析
    attrs: dict = field(default_factory=dict)


# 单标记类型的开启串
_OPENERS = {
    "STRUCT": "[STRUCT:",
    "ARROW": "[ARROW:",
    "REACTION": "[REACTION:",
    "ENERGY": "[ENERGY:",
    "CHARGE": "[CHARGE:",
    "HBOND": "[HBOND:",
    "RETRO": "[RETRO:",
    "XH": "[XH:",
    "BOND": "[BOND:",
}

# REASONING 配对正则（内容不与括号冲突，可用正则）
_REASONING_RE = re.compile(r"\[REASONING\](.*?)\[/REASONING\]", re.DOTALL)

# COMPOSITE 容器配对串
_COMPOSITE_OPEN = "[COMPOSITE:"
_COMPOSITE_CLOSE = "[/COMPOSITE]"


def _only_child_tags_between(text: str, lo: int, hi: int) -> bool:
    """text[lo:hi] 是否只含容器子标记与空白（夹有正文文字则返回 False）。

    用于区分"[COMPOSITE: 正文提及"与"真嵌套"：前者两个开头之间是中文
    说明文字，后者之间只有 [STRUCT:...] 等子标记。
    """
    i = lo
    while i < hi:
        if text[i].isspace():
            i += 1
            continue
        matched = False
        for opener in _INNER_OPENERS.values():
            if text.startswith(opener, i):
                end = _find_tag_end(text, i)
                if end == -1 or end >= hi:
                    return False
                i = end + 1
                matched = True
                break
        if not matched:
            for token in _INNER_TOKENS.values():
                if text.startswith(token, i):
                    i += len(token)
                    matched = True
                    break
        if not matched:
            return False
    return True

# COMPOSITE 容器内允许的带子标记 opener（冒号形式）
# ARROW 为大一统架构的新箭头标记（[ARROW:type=...,sup=...,条件]）；
# RXNARROW/RESARROW/CONDITION 为旧箭头标记（兼容保留，新架构不用）
_INNER_OPENERS = {
    "STRUCT": "[STRUCT:",
    "ARROW": "[ARROW:",
    "MECHARROW": "[MECHARROW:",
    "CONDITION": "[CONDITION:",
    "RXNARROW": "[RXNARROW:",
    "CHARGE": "[CHARGE:",
    "HBOND": "[HBOND:",
    "XH": "[XH:",
    "BOND": "[BOND:",
    "ENERGY": "[ENERGY:",
}

# COMPOSITE 容器内允许的无参子标记（整串匹配）
_INNER_TOKENS = {
    "PLUS": "[PLUS]",
    "RXNARROW": "[RXNARROW]",
    "RESARROW": "[RESARROW]",
    "NEWLINE": "[NEWLINE]",
}

# 共振块定界（20260819 大一统架构）：[BLOCK]...[/BLOCK]——块内为
# STRUCT + 共振箭头序列，作为复合结构参与外层反应序列。
_BLOCK_OPEN = "[BLOCK]"
_BLOCK_CLOSE = "[/BLOCK]"


# STRUCT 支持的绘制模式（skeleton=默认键线式/结构简式）
_STRUCT_MODES = ("skeleton", "lewis", "stereo", "chair", "newman")


def _normalize_render_tag(tag_type: str, content: str, raw: str,
                          start_pos: int, end_pos: int) -> RenderTag:
    """构建 RenderTag；STRUCT 提取结构化属性（id/at/pos/mode/subs/
    bond/angle/charge/arrow），其余标记只有 args。"""
    attrs = _parse_struct_attrs(content) if tag_type == "STRUCT" else {}
    return RenderTag(type=tag_type, args=_parse_content(tag_type, content),
                     raw=raw, start_pos=start_pos, end_pos=end_pos, attrs=attrs)


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


def _find_kv(content: str, key: str) -> tuple:
    """定位 ",key=" 或 ", key="（容忍逗号后空格），返回 (逗号位置, 值起点)。

    逗号位置用于裁剪 SMILES（截止到该参数）；值起点用于提取参数值。
    未找到返回 (-1, -1)。
    """
    m = re.search(r",\s*" + re.escape(key) + r"=", content or "")
    return (m.start(), m.end()) if m else (-1, -1)


def _strip_fake_label(s: str) -> str:
    """剥离非 STRUCT 标记 SMILES 字段里误写的 ,label= 尾随（问题 C-2）。

    LLM 高频错误：以为所有标记都支持 label，在 [XH:O,label=苯酚|0]、
    [CHARGE:O,label=苯酚|0:δ-] 等标记里误写 label。按 `|` 分割后 SMILES
    字段会是 "O,label=苯酚"，直接进 RDKit 报 SMILES Parse Error。
    这里把 ,label= 及其后的内容剥掉（SMILES 本身不含逗号，安全）。
    """
    s = s.strip()
    li = s.find(",label=")
    if li != -1:
        s = s[:li]
    return s.strip().rstrip(",")


def _parse_content(tag_type: str, content: str) -> list:
    """把标记内部内容拆为参数列表。SMILES 不含逗号，按逗号切分安全。"""
    if tag_type == "STRUCT":
        smi, label = content, None
        li, li_end = _find_kv(content, "label")
        ii, ii_end = _find_kv(content, "id")
        ai, ai_end = _find_kv(content, "at")
        pi, pi_end = _find_kv(content, "pos")
        mi, mi_end = _find_kv(content, "mode")
        si, si_end = _find_kv(content, "subs")
        bi, bi_end = _find_kv(content, "bond")
        gi, gi_end = _find_kv(content, "angle")
        ci, ci_end = _find_kv(content, "charge")
        # arrow 裸令牌（箭头上附件标记）也是参数分隔符
        atok = re.search(r",\s*arrow(?=,|\s*$)", content or "")
        ap = atok.start() if atok else -1
        cut = [p for p in (li, ii, ai, pi, mi, si, bi, gi, ci, ap) if p != -1]
        if cut:
            smi = content[: min(cut)]
        if li != -1:
            # 值截止到下一个参数的逗号位置（after 用 start）
            after = [p for p in (ii, ai, pi, mi, si, bi, gi, ci, ap)
                     if p != -1 and p > li]
            label = content[li_end:min(after)] if after else content[li_end:]
        # LLM 偶发写出尾逗号（如 [STRUCT:CC[OH2+],]），归一化去掉
        return [smi.strip().rstrip(","), label]
    if tag_type == "CHARGE":
        if "|" in content:
            smi, _, charges = content.partition("|")
            return [_strip_fake_label(smi), charges.strip()]
        return [_strip_fake_label(content), ""]
    if tag_type == "HBOND":
        if "|" in content:
            smi, _, pairs = content.partition("|")
            return [_strip_fake_label(smi), pairs.strip()]
        if ">" in content:
            # 跨组件紧凑写法 HBOND:idA:from>idB:to（无 |）→ [idA, "from>idB:to"]
            ida, _, rest = content.partition(":")
            return [ida.strip(), rest.strip()]
        return [_strip_fake_label(content), ""]
    if tag_type == "ARROW":
        if (content or "").strip().startswith("type="):
            # 容器内新语法（20260819 大一统架构）：[ARROW:type=..., sup=..., 条件]
            # 返回 [type, sup附件列表, 条件]。sup 附件 = "+id"（副反应物，箭头上方）
            # / "-id"（副产物，箭头下方），逗号分隔（与用户示例一致：sup=+E,-F）。
            # 附件列表吸收以 +/- 开头且形如 id 的后续逗号项，消除与条件文本歧义。
            content = content.strip()
            type_, rest = "single", content
            m = re.match(r"^type=(\w+)", content)
            if m:
                type_, rest = m.group(1), content[m.end():].strip()
            sup_items, rest2 = [], []
            tokens = [t.strip() for t in rest.split(",") if t.strip()]
            in_sup = False
            for t in tokens:
                if t.startswith("sup="):
                    in_sup = True
                    # sup 值内附件可再以 `;` 分隔（与 REACTION 物种分隔一致）
                    for s in t[len("sup="):].split(";"):
                        if s.strip():
                            sup_items.append(s.strip())
                    continue
                if in_sup and re.fullmatch(r"[+-][A-Za-z0-9_]+", t):
                    sup_items.append(t)
                    continue
                in_sup = False
                rest2.append(t)
            return [type_, sup_items, ", ".join(rest2)]
        # 顶层旧语法：[ARROW:反应物,产物,类型]
        parts = content.split(",", 2)
        while len(parts) < 3:
            parts.append("")
        return parts
    if tag_type == "REACTION":
        parts = content.split("|", 2)
        while len(parts) < 3:
            parts.append("")
        return [p.strip() for p in parts]
    if tag_type == "RETRO":
        parts = content.split(",", 2)
        while len(parts) < 3:
            parts.append("")
        return parts
    if tag_type in ("MECHARROW", "CONDITION", "RXNARROW"):
        return [content.strip()]
    if tag_type in ("XH", "BOND"):
        if "|" in content:
            ref, _, spec = content.partition("|")
            return [_strip_fake_label(ref), spec.strip()]
        return [_strip_fake_label(content), ""]
    return [content]  # ENERGY / 其他


def _parse_struct_attrs(content: str) -> dict:
    """提取 STRUCT 内容中的结构化属性 id/at/pos/mode/subs/bond/angle/charge。

    与 _parse_content 的 STRUCT 分支共用查找逻辑，但额外返回属性值，
    供 composite / tag_validator / 渲染分派直接读取（改进 3：避免从
    raw 二次正则解析）。mode 缺省 "skeleton"（键线式）。
    """
    attrs = {}
    li, li_end = _find_kv(content, "label")
    ii, ii_end = _find_kv(content, "id")
    ai, ai_end = _find_kv(content, "at")
    pi, pi_end = _find_kv(content, "pos")
    mi, mi_end = _find_kv(content, "mode")
    si, si_end = _find_kv(content, "subs")
    bi, bi_end = _find_kv(content, "bond")
    gi, gi_end = _find_kv(content, "angle")
    ci, ci_end = _find_kv(content, "charge")
    # 值截止用下一参数的逗号位置（start）；提取起点用本参数值起点（end）
    atok = re.search(r",\s*arrow(?=,|\s*$)", content or "")
    ap = atok.start() if atok else -1
    others_start = [p for p in (li, ii, ai, pi, mi, si, bi, gi, ci, ap)
                    if p != -1]
    if ii != -1:
        after = [p for p in others_start if p > ii]
        val = content[ii_end:min(after) if after else len(content)]
        attrs["id"] = val.strip()
    if ai != -1:
        after = [p for p in others_start if p > ai]
        val = content[ai_end:min(after) if after else len(content)].strip()
        if val.isdigit():
            attrs["at"] = int(val)
    if pi != -1:
        after = [p for p in others_start if p > pi]
        val = content[pi_end:min(after) if after else len(content)].strip()
        if val in ("above", "below"):
            attrs["pos"] = val
    if mi != -1:
        after = [p for p in others_start if p > mi]
        val = content[mi_end:min(after) if after else len(content)].strip()
        attrs["mode"] = val
    else:
        attrs["mode"] = "skeleton"
    if si != -1:
        after = [p for p in others_start if p > si]
        val = content[si_end:min(after) if after else len(content)].strip()
        attrs["subs"] = val
    if bi != -1:
        after = [p for p in others_start if p > bi]
        val = content[bi_end:min(after) if after else len(content)].strip()
        attrs["bond"] = val
    if gi != -1:
        after = [p for p in others_start if p > gi]
        val = content[gi_end:min(after) if after else len(content)].strip()
        attrs["angle"] = val
    if ci != -1:
        after = [p for p in others_start if p > ci]
        val = content[ci_end:min(after) if after else len(content)].strip()
        attrs["charge"] = val
    # arrow 裸令牌（大一统架构：该 STRUCT 是箭头上附件——副反应物/副产物）
    if re.search(r",\s*arrow(?=,|\s*$)", content or ""):
        attrs["arrow"] = True
    return attrs


def _in_regions(pos: int, regions: list) -> bool:
    """pos 是否落在任一 (start, end) 区域内。"""
    return any(start <= pos < end for start, end in regions)


def _parse_inner_tags(inner: str, base: int) -> List[RenderTag]:
    """解析 COMPOSITE 容器内部的子标记。

    参数:
        inner: 容器内文本（不含 [COMPOSITE:...] 与 [/COMPOSITE]）。
        base: inner 在原文中的起始下标，用于把子标记位置换算回原文坐标。

    先配对 [BLOCK]...[/BLOCK] 共振块（块内子标记递归解析，不允许嵌套），
    其余子标记照常解析并跳过块区域。
    """
    children = []
    block_regions = []
    bi = 0
    while True:
        bo = inner.find(_BLOCK_OPEN, bi)
        if bo == -1:
            break
        bc = inner.find(_BLOCK_CLOSE, bo + len(_BLOCK_OPEN))
        if bc == -1:
            bi = bo + len(_BLOCK_OPEN)
            continue
        block_inner = inner[bo + len(_BLOCK_OPEN):bc]
        sub = _parse_inner_tags(block_inner, base + bo + len(_BLOCK_OPEN))
        raw = inner[bo:bc + len(_BLOCK_CLOSE)]
        children.append(RenderTag(
            type="BLOCK", args=[sub], raw=raw,
            start_pos=base + bo, end_pos=base + bc + len(_BLOCK_CLOSE),
        ))
        block_regions.append((bo, bc + len(_BLOCK_CLOSE)))
        bi = bc + len(_BLOCK_CLOSE)

    def _in_block(pos: int) -> bool:
        return any(s <= pos < e for s, e in block_regions)

    for tag_type, opener in _INNER_OPENERS.items():
        search_from = 0
        while True:
            idx = inner.find(opener, search_from)
            if idx == -1:
                break
            if _in_block(idx):
                search_from = idx + len(opener)
                continue
            end = _find_tag_end(inner, idx)
            if end == -1:
                search_from = idx + len(opener)
                continue
            raw = inner[idx:end + 1]
            content = raw[len(opener):-1]
            children.append(_normalize_render_tag(
                tag_type, content, raw, base + idx, base + end + 1))
            search_from = end + 1
    for tag_type, token in _INNER_TOKENS.items():
        search_from = 0
        while True:
            idx = inner.find(token, search_from)
            if idx == -1:
                break
            if _in_block(idx):
                search_from = idx + len(token)
                continue
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
        inner_open = text.find(_COMPOSITE_OPEN, open_end + 1, close_idx)
        if inner_open != -1 and not _only_child_tags_between(
                text, open_end + 1, inner_open):
            # 外层开头与内层开头之间夹有正文文字 → 外层是正文中的文字提及
            # （如"用 [COMPOSITE:reaction_mech] 展示："），跳过它，让内层
            # 真容器与闭合配对；之间只有子标记则是真嵌套（不支持），
            # 维持外层优先配对（兼容既有行为）
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
            tags.append(_normalize_render_tag(
                tag_type, content, raw, idx, end + 1))
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
