# -*- coding: utf-8 -*-
"""core/tag_validator.py — 标记契约校验层（P1）。

在 parse_tags 之后、渲染之前统一校验标记参数，坏参数不再进入渲染器：

- 格式校验（不依赖 rdkit）：必需字段非空、数值参数可解析（NEWMAN 角度 /
  ENERGY 点序列）、COMPOSITE 布局合法、容器内组件引用（MECHARROW / CHARGE /
  HBOND / XH / BOND 的组件 id）存在、energy 布局 at= 不越界；
- 语义校验（rdkit 可用时）：SMILES 可解析、原子编号在组件原子数范围内；
- label 长度硬上限（默认 24 字符），拦截端到端中最常见的标签重叠诱因。

调用方（app.process_question）对校验失败的标记注入友好降级提示，
而不是让渲染器输出格式各异的错误串或带着坏参数崩溃。

设计原则：校验层只拦截"必然出错"的输入（非法 SMILES、越界引用、格式错误），
宽松阈值避免误杀合法标记；渲染器内部的二次校验保留为兜底防线。
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from .tag_parser import RenderTag

# rdkit 可用性探测：缺失时跳过 SMILES / 原子数语义校验（渲染器内部会兜底）
try:
    from rdkit import Chem  # noqa: F401
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False

# label 长度硬上限（字符数）。prompt 建议 ≤10（中文 ≤6），此处为兜底硬拦截
LABEL_MAX_LEN = 24

# COMPOSITE 支持的布局（与 renderers/composite.py 保持一致）
COMPOSITE_LAYOUTS = ("reaction_mech", "row", "energy", "resonance")

# 与 renderers/composite.py 相同的引用/端点提取正则
_MECH_ARROW_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*:\s*(\d+(?:-\d+)?)\s*(>>|>)\s*"
    r"([A-Za-z0-9_]+)\s*:\s*(\d+(?:-\d+)?)\s*$"
)
_STRUCT_ID_RE = re.compile(r",id=([A-Za-z0-9_]+)")
_STRUCT_AT_RE = re.compile(r",at=(\d+)")

# 标记类型 → 中文名（降级提示用）
_TAG_NAMES = {
    "STRUCT": "结构式",
    "ARROW": "反应箭头",
    "REACTION": "反应方程式",
    "REACTIONMECH": "反应机理图",
    "COMPOSITE": "复合图",
    "NEWMAN": "纽曼投影",
    "LEWIS": "Lewis 结构式",
    "ENERGY": "势能面",
    "STEREO": "楔形式",
    "MECH": "机理箭头",
    "CHARGE": "电荷标注",
    "RESONANCE": "共振结构式",
    "HBOND": "氢键标注",
    "RETRO": "逆合成箭头",
}

# SMILES 字段提取器：输入 RenderTag，返回需要校验的 SMILES 字符串列表。
# 返回 None 表示该标记类型不携带 SMILES（如 ENERGY / PLUS）。
_SMILES_FIELDS = {
    "STRUCT": lambda a: [a[0]] if a and a[0] else [],
    "ARROW": lambda a: [a[i] for i in (0, 1) if i < len(a) and a[i]],
    "REACTION": lambda a: _split_multi(a[0]) + _split_multi(a[1]) if len(a) >= 2 else [],
    "REACTIONMECH": lambda a: _split_multi(a[0]) + _split_multi(a[1]) if len(a) >= 2 else [],
    "NEWMAN": lambda a: [a[0]] if a and a[0] else [],
    "LEWIS": lambda a: [a[0]] if a and a[0] else [],
    "STEREO": lambda a: [a[0]] if a and a[0] else [],
    "MECH": lambda a: [a[0]] if a and a[0] else [],
    "CHARGE": lambda a: [a[0]] if a and a[0] else [],
    "HBOND": lambda a: [a[0]] if a and a[0] else [],
    "RESONANCE": lambda a: [s.strip() for s in (a[0].split("~") if a and a[0] else [])],
    "RETRO": lambda a: [a[i] for i in (0, 1) if i < len(a) and a[i]],
}


def _split_multi(seg: str) -> list:
    """把分号分隔的多组分段拆为 SMILES 列表（过滤空串）。"""
    return [s.strip() for s in seg.split(";") if s.strip()] if seg else []


@dataclass
class ValidationResult:
    """单个标记的校验结果。"""

    tag: RenderTag
    ok: bool
    reason: str = ""


def tag_name(tag_type: str) -> str:
    """标记类型的中文名（降级提示用）。"""
    return _TAG_NAMES.get(tag_type, tag_type)


def _smiles_ok(smiles: str) -> bool:
    """SMILES 语义校验；rdkit 不可用时放行（格式校验已做）。"""
    if not _RDKIT_OK:
        return True
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _label_ok(label) -> Tuple[bool, str]:
    """label 长度硬校验；None（无 label）视为通过。"""
    if not label:
        return True, ""
    if len(str(label)) > LABEL_MAX_LEN:
        return False, f"label 过长（{len(str(label))} > {LABEL_MAX_LEN} 字符）"
    return True, ""


def _validate_struct_args(args: list) -> Tuple[bool, str]:
    """校验单个 STRUCT 参数（顶层或容器内）：SMILES 非空 + label 长度。"""
    if not args or not args[0] or not args[0].strip():
        return False, "SMILES 为空"
    ok, reason = _label_ok(args[1] if len(args) > 1 else None)
    if not ok:
        return False, reason
    if not _smiles_ok(args[0].strip()):
        return False, f"无效 SMILES「{args[0].strip()}」"
    return True, ""


def _validate_newman(args: list) -> Tuple[bool, str]:
    if not args or not args[0]:
        return False, "SMILES 为空"
    angle = args[1] if len(args) > 1 else ""
    try:
        a = float(angle)
    except (TypeError, ValueError):
        return False, f"角度「{angle}」不是数字"
    if not 0 <= a <= 360:
        return False, f"角度 {a:g} 超出 0~360"
    if not _smiles_ok(args[0].strip()):
        return False, f"无效 SMILES「{args[0].strip()}」"
    return True, ""


def _validate_energy(args: list) -> Tuple[bool, str]:
    if not args or not args[0]:
        return False, "点序列为空"
    values = [v.strip() for v in args[0].split(",") if v.strip()]
    for v in values:
        try:
            float(v)
        except ValueError:
            return False, f"能量值「{v}」不是数字"
    if len(values) < 2:
        return False, "至少需要 2 个能量点"
    return True, ""


def _validate_mech_arrow_pt(pt: str, n_atoms: int) -> bool:
    """端点（原子序号或 a-b 键）是否在原子数范围内。"""
    if "-" in pt:
        a, b = pt.split("-")
        try:
            return 0 <= int(a) < n_atoms and 0 <= int(b) < n_atoms
        except ValueError:
            return False
    try:
        return 0 <= int(pt) < n_atoms
    except ValueError:
        return False


def _validate_composite(layout: str, children: list) -> Tuple[bool, str]:
    """COMPOSITE 容器校验：布局合法 + 子标记递归校验 + 引用存在性 + at= 越界。"""
    header = [p.strip() for p in (layout or "").split(",")]
    layout_name = header[0]
    if layout_name not in COMPOSITE_LAYOUTS:
        return False, f"未知布局「{layout_name}」，支持 {'/'.join(COMPOSITE_LAYOUTS)}"

    # 收集组件：id → {smiles, at}
    comps = {}
    for child in children:
        if child.type != "STRUCT":
            continue
        m = _STRUCT_ID_RE.search(child.raw)
        cid = m.group(1) if m else f"r{len(comps)}"
        at_m = _STRUCT_AT_RE.search(child.raw)
        comps[cid] = {
            "smiles": child.args[0].strip() if child.args and child.args[0] else "",
            "at": int(at_m.group(1)) if at_m else None,
        }
        # STRUCT 子标记本身递归校验
        ok, reason = _validate_struct_args(child.args)
        if not ok:
            return False, f"组件 {cid}: {reason}"

    if not comps:
        return False, "容器内缺少 [STRUCT] 组件"

    # energy 布局：每个 STRUCT 必须有 at= 且不越界
    if layout_name == "energy":
        energy_child = next((c for c in children if c.type == "ENERGY"), None)
        if energy_child is None or not energy_child.args or not energy_child.args[0]:
            return False, "energy 布局需要 [ENERGY:点序列] 组件"
        n_points = len([v for v in energy_child.args[0].split(",") if v.strip()])
        for cid, info in comps.items():
            if info["at"] is None:
                return False, f"energy 布局中组件 {cid} 需要 at=点序号"
            if not 0 <= info["at"] < n_points:
                return False, f"组件 {cid} 的 at={info['at']} 超出能量点范围 0~{n_points - 1}"

    # 引用校验：需要原子数时先解析所有组件 SMILES
    atom_counts = {}
    if _RDKIT_OK:
        for cid, info in comps.items():
            if info["smiles"]:
                try:
                    mol = Chem.MolFromSmiles(info["smiles"])
                    atom_counts[cid] = mol.GetNumAtoms() if mol else 0
                except Exception:
                    atom_counts[cid] = 0

    for child in children:
        ctype = child.type
        if ctype == "MECHARROW":
            if not child.args or not child.args[0]:
                return False, "MECHARROW 为空"
            for spec in child.args[0].split(","):
                m = _MECH_ARROW_RE.match(spec)
                if not m:
                    return False, f"MECHARROW 格式错误「{spec}」"
                src_id, src_pt, _, dst_id, dst_pt = m.groups()
                if src_id not in comps or dst_id not in comps:
                    return False, f"MECHARROW 引用未知组件「{src_id}→{dst_id}」"
                if _RDKIT_OK:
                    if not _validate_mech_arrow_pt(src_pt, atom_counts.get(src_id, 0)):
                        return False, f"MECHARROW 源端点「{src_id}:{src_pt}」超出原子范围"
                    if not _validate_mech_arrow_pt(dst_pt, atom_counts.get(dst_id, 0)):
                        return False, f"MECHARROW 目标端点「{dst_id}:{dst_pt}」超出原子范围"
        elif ctype in ("CHARGE", "HBOND") and len(child.args) >= 2:
            ref = child.args[0].strip()
            if ref not in comps:
                return False, f"{ctype} 引用未知组件「{ref}」"
            pairs = (child.args[1] or "").strip()
            if ctype == "CHARGE":
                idxs = [int(x) for x in re.findall(r"(\d+):", pairs)]
                if not idxs:
                    return False, f"CHARGE 标注格式错误「{pairs}」（应为 原子:δ± 列表）"
            else:
                hb_pairs = re.findall(r"(\d+)-(\d+)", pairs)
                if not hb_pairs:
                    return False, f"HBOND 标注格式错误「{pairs}」（应为 from-to 列表）"
                idxs = [int(x) for t in hb_pairs for x in t]
            if _RDKIT_OK:
                n = atom_counts.get(ref, 0)
                for i in idxs:
                    if not 0 <= i < n:
                        return False, f"{ctype} 原子编号 {i} 超出组件 {ref} 范围 0~{n - 1}"
        elif ctype == "XH" and len(child.args) >= 2:
            ref = child.args[0].strip()
            if ref not in comps:
                return False, f"XH 引用未知组件「{ref}」"
            if _RDKIT_OK:
                try:
                    i = int(child.args[1])
                except ValueError:
                    return False, f"XH 原子编号「{child.args[1]}」不是数字"
                n = atom_counts.get(ref, 0)
                if not 0 <= i < n:
                    return False, f"XH 原子编号 {i} 超出组件 {ref} 范围 0~{n - 1}"
        elif ctype == "BOND" and len(child.args) >= 2:
            ref = child.args[0].strip()
            if ref not in comps:
                return False, f"BOND 引用未知组件「{ref}」"
            if _RDKIT_OK:
                m = re.fullmatch(r"(\d+)-(\d+)", child.args[1].strip())
                if not m:
                    return False, f"BOND 键引用格式错误「{child.args[1]}」"
                n = atom_counts.get(ref, 0)
                a, b = int(m.group(1)), int(m.group(2))
                if not (0 <= a < n and 0 <= b < n):
                    return False, f"BOND 键 {a}-{b} 超出组件 {ref} 范围 0~{n - 1}"
    return True, ""


def validate_tag(tag: RenderTag) -> ValidationResult:
    """校验单个标记，返回 ValidationResult。"""
    ttype, args = tag.type, tag.args
    # REASONING 为配对文本，不校验
    if ttype == "REASONING":
        return ValidationResult(tag, True)
    if ttype == "COMPOSITE":
        ok, reason = _validate_composite(args[0], args[1]) if len(args) >= 2 \
            else (False, "COMPOSITE 缺少容器参数")
        return ValidationResult(tag, ok, reason)
    if ttype == "ENERGY":
        ok, reason = _validate_energy(args)
        return ValidationResult(tag, ok, reason)
    if ttype == "NEWMAN":
        ok, reason = _validate_newman(args)
        return ValidationResult(tag, ok, reason)
    if ttype == "STRUCT":
        ok, reason = _validate_struct_args(args)
        return ValidationResult(tag, ok, reason)

    # 通用带 SMILES 字段的标记：字段非空 + SMILES 合法
    fields = _SMILES_FIELDS.get(ttype)
    if fields is not None:
        smi_list = fields(args)
        if not smi_list:
            return ValidationResult(tag, False, "缺少 SMILES 字段")
        for smi in smi_list:
            if not smi:
                return ValidationResult(tag, False, "SMILES 为空")
            if not _smiles_ok(smi):
                return ValidationResult(tag, False, f"无效 SMILES「{smi}」")
        return ValidationResult(tag, True)
    return ValidationResult(tag, True)  # 未知类型放行（注入时保留原文）


def validate_tags(tags: List[RenderTag]) -> Tuple[List[RenderTag], List[ValidationResult]]:
    """校验标记列表。

    返回:
        (valid_tags, invalid_results)：
        valid_tags — 通过校验的标记（顺序保持）；
        invalid_results — 校验失败的 ValidationResult 列表（含失败原因）。
    """
    valid, invalid = [], []
    for tag in tags:
        result = validate_tag(tag)
        if result.ok:
            valid.append(tag)
        else:
            invalid.append(result)
    return valid, invalid


def degrade_text(tag: RenderTag, reason: str) -> str:
    """校验失败标记的降级提示文本。"""
    return f"（{tag_name(tag.type)}图示无法渲染：{reason}，已省略）"


if __name__ == "__main__":
    # 冒烟测试（不依赖 rdkit）
    demo = (
        "[STRUCT:c1ccccc1] "
        "[STRUCT:XYZXYZ,label=无效结构] "
        "[ENERGY:0,108,-20] "
        "[ENERGY:abc] "
        "[NEWMAN:CC,60] [NEWMAN:CC,xyz] "
        "[COMPOSITE:energy][ENERGY:0,108,-20]"
        "[STRUCT:CCl,label=A,at=0][STRUCT:CO,label=B,at=9]"
        "[/COMPOSITE]"
    )
    from .tag_parser import parse_tags
    for t in parse_tags(demo):
        r = validate_tag(t)
        print(("✓" if r.ok else "✗"), t.type, t.raw[:60], "" if r.ok else f"— {r.reason}")
