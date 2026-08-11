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
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple

from .tag_parser import RenderTag

# rdkit 可用性探测：缺失时跳过 SMILES / 原子数语义校验（渲染器内部会兜底）
try:
    from rdkit import Chem  # noqa: F401
    from utils.rdkit_utils import FREE_H_COMPONENT_RE, mute_rdkit_warnings
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False
    FREE_H_COMPONENT_RE = None
    mute_rdkit_warnings = None


def _parse_mol(smiles: str):
    """Chem.MolFromSmiles 局部包装：游离氢组分（合法）的无害警告静默。
    保持经本模块命名空间调用 Chem（fake_rdkit fixture 可替换）。"""
    if (mute_rdkit_warnings is not None
            and FREE_H_COMPONENT_RE.search(smiles or "")):
        with mute_rdkit_warnings():
            return Chem.MolFromSmiles(smiles)
    return Chem.MolFromSmiles(smiles)

# label 长度硬上限（字符数）。prompt 建议 ≤10（中文 ≤6），此处为兜底硬拦截
LABEL_MAX_LEN = 24

# COMPOSITE 支持的布局（与 renderers/composite.py 保持一致）
COMPOSITE_LAYOUTS = ("reaction_mech", "row", "energy", "resonance")

# 与 renderers/composite.py 相同的引用/端点提取正则
# （目标端可选 "+id:原子" 后缀：成键空白位，两原子间中点）
_MECH_ARROW_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*:\s*(\d+(?:-\d+)?)\s*(>>|>)\s*"
    r"([A-Za-z0-9_]+)\s*:\s*(\d+(?:-\d+)?)"
    r"(?:\s*\+\s*([A-Za-z0-9_]+)\s*:\s*(\d+))?\s*$"
)

# 标记类型 → 中文名（降级提示用）
_TAG_NAMES = {
    "STRUCT": "结构式",
    "ARROW": "反应箭头",
    "REACTION": "反应方程式",
    "COMPOSITE": "复合图",
    "NEWMAN": "纽曼投影",
    "LEWIS": "Lewis 结构式",
    "ENERGY": "势能面",
    "STEREO": "楔形式",
    "CHARGE": "电荷标注",
    "HBOND": "氢键标注",
    "RETRO": "逆合成箭头",
    "XH": "显式氢标注",
    "BOND": "键突出标注",
}

# SMILES 字段提取器：输入 RenderTag，返回需要校验的 SMILES 字符串列表。
# 返回 None 表示该标记类型不携带 SMILES（如 ENERGY / PLUS）。
_SMILES_FIELDS = {
    "STRUCT": lambda a: [a[0]] if a and a[0] else [],
    "ARROW": lambda a: [a[i] for i in (0, 1) if i < len(a) and a[i]],
    "REACTION": lambda a: _split_multi(a[0]) + _split_multi(a[1]) if len(a) >= 2 else [],
    "NEWMAN": lambda a: [a[0]] if a and a[0] else [],
    "LEWIS": lambda a: [a[0]] if a and a[0] else [],
    "STEREO": lambda a: [a[0]] if a and a[0] else [],
    "CHARGE": lambda a: [a[0]] if a and a[0] else [],
    "HBOND": lambda a: [a[0]] if a and a[0] else [],
    "RETRO": lambda a: [a[i] for i in (0, 1) if i < len(a) and a[i]],
}


def _split_multi(seg: str) -> list:
    """把多组分段拆为 SMILES 列表（过滤空串）。

    契约分隔符为分号；LLM 偶发用逗号分隔或写出尾逗号（SMILES 不含逗号），
    按 [;,] 拆分归一化（与 renderers/reaction.py 保持一致）。
    """
    return [s.strip() for s in re.split(r"[;,]", seg or "") if s.strip()]


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
        return _parse_mol(smiles) is not None
    except Exception:
        return False


def _label_ok(label) -> Tuple[bool, str]:
    """label 长度硬校验；None（无 label）视为通过。"""
    if not label:
        return True, ""
    if len(str(label)) > LABEL_MAX_LEN:
        return False, f"label 过长（{len(str(label))} > {LABEL_MAX_LEN} 字符）"
    return True, ""


# ---------------------------------------------------------------------------
# 化学校验（T2）：label 化学式一致性 + 原子守恒。
# 失败原因统一以「化学校验：」前缀，metrics 据此统计化学正确率维度。
# 均为 best-effort：元素计数无法计算（rdkit 缺失/fake mol）时跳过不放行误判。
# ---------------------------------------------------------------------------

_CHEM_PREFIX = "化学校验："

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

# 可识别的真实元素（有机/常见无机）。刻意不含 Ar（氩）、Ac（锕）等——
# 它们是 prompt 允许的通用基团缩写（Ar=芳基、Ac=乙酰基，见
# Instruction-for-Structure.md），误判为化学式会把缩写 label 打回。
_REAL_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I",
    "Li", "Na", "K", "Mg", "Ca", "Al", "Fe", "Cu", "Zn", "Ag", "Au",
    "Hg", "Pb", "Sn", "Se", "Te",
}


def _parse_plain_formula(text: str):
    """把纯化学式 label（CH3Cl / OH- / NO2+ / H3O+ 等）解析为候选
    (元素计数 dict, 净电荷) 列表；非纯化学式返回 []——含中文/空格/结构括号、
    通用基团缩写（R/Ar/X/Ph 等）、或含不可识别"元素"（如占位字母 A）一律跳过。

    尾电荷的数字归属有歧义（NO2+ 是 N1O2 带 +1，Ca2+ 是 Ca 带 +2），
    两种解读都给出候选，由 SMILES 比对定夺。
    """
    s = (text or "").strip()
    if not s:
        return []
    bodies = []
    m = re.search(r"(\d+)?([+-])$", s)
    if m:
        # 尾部数字串归属有歧义：NO2+ 的 2 归元素（N1O2 带 +1）、Ca2+ 的 2
        # 归电荷（Ca 带 +2）、SO42- 的 4 归元素而 2 归电荷（S1O4 带 -2）。
        # 按数字串的每个切分点各给一个候选，由 SMILES 比对定夺。
        sign = 1 if m.group(2) == "+" else -1
        digits = m.group(1) or ""
        base = s[: m.start()]
        for j in range(len(digits) + 1):
            q = int(digits[j:]) if digits[j:] else 1
            bodies.append((base + digits[:j], q * sign))
    else:
        bodies.append((s, 0))
    cands = []
    for body, charge in bodies:
        if not body or not re.fullmatch(r"([A-Z][a-z]?\d*)+", body):
            continue
        counts = {}
        ok = True
        for sym, num in _FORMULA_TOKEN_RE.findall(body):
            if sym not in _REAL_ELEMENTS:
                ok = False
                break
            counts[sym] = counts.get(sym, 0) + (int(num) if num else 1)
        if ok and (counts, charge) not in cands:
            cands.append((counts, charge))
    return cands


def _mol_counts(mol):
    """RDKit Mol → (元素计数 dict（重原子 + 隐式 H）, 净电荷)；失败返回 None。"""
    try:
        counts, charge = {}, 0
        for a in mol.GetAtoms():
            sym = a.GetSymbol()
            counts[sym] = counts.get(sym, 0) + 1
            h = a.GetTotalNumHs()
            if h:
                counts["H"] = counts.get("H", 0) + h
            charge += a.GetFormalCharge()
        return counts, charge
    except Exception:
        return None


def _hill_str(counts: dict) -> str:
    """元素计数 → Hill 化学式串（错误提示用；不含电荷）。"""
    parts = []
    for sym in sorted(counts, key=lambda s: (s != "C", s != "H", s)):
        n = counts[sym]
        parts.append(sym + (str(n) if n > 1 else ""))
    return "".join(parts)


def _check_label_formula(smiles: str, label: str) -> str:
    """T2-2：label 为纯化学式时与 SMILES 元素计数/电荷比对；不一致返回原因。"""
    cands = _parse_plain_formula(label)
    if not cands:
        return ""
    mc = _mol_counts(_parse_mol(smiles))
    if mc is None:
        return ""
    if mc not in cands:
        return (f"{_CHEM_PREFIX}label「{label}」与 SMILES「{smiles}」化学式不一致"
                f"（label={_hill_str(cands[0][0])}，SMILES={_hill_str(mc[0])}，"
                f"请使 label 与结构指向同一物质）")
    return ""


def _sum_species(smiles_list: list):
    """一组 SMILES 的元素计数加总（含净电荷，供比对）；任一无法解析返回 None。"""
    total_c, total_q = {}, 0
    for smi in smiles_list:
        mc = _mol_counts(_parse_mol(smi))
        if mc is None:
            return None
        for sym, n in mc[0].items():
            total_c[sym] = total_c.get(sym, 0) + n
        total_q += mc[1]
    return total_c, total_q


def _balance_reason(left, right, strict_h: bool, step: str) -> str:
    """两侧元素计数比对：非 H 元素必须相等；strict_h 时 H 也必须相等。
    （reaction_mech 容忍 H±差——质子转移/去质子副产 H⁺ 常按惯例不画出；
    净电荷不比对：旁观离子（如 HSO4⁻）省略是方程式惯例。）"""
    if left is None or right is None:
        return ""
    lc, rc = dict(left[0]), dict(right[0])
    if not strict_h:
        lc.pop("H", None)
        rc.pop("H", None)
    if lc != rc:
        detail = f"{_hill_str(left[0])} vs {_hill_str(right[0])}"
        return (f"{_CHEM_PREFIX}{step}两侧原子不守恒（{detail}，"
                f"需配平或补全物种；辅助试剂请写入箭头条件而非省略主物种）")
    return ""


def _check_reaction_balance(args: list) -> str:
    """T2-3：REACTION 两侧全元素（含 H）配平（REACTION 为完整方程式契约）。"""
    if len(args) < 2:
        return ""
    left = _sum_species(_split_multi(args[0]))
    right = _sum_species(_split_multi(args[1]))
    return _balance_reason(left, right, strict_h=True, step="方程式")


def _check_composite_balance(children: list, layout_name: str) -> str:
    """T2-3：COMPOSITE 的 reaction_mech 布局按 RXNARROW 分步、逐步比对
    非 H 元素守恒（容忍 H±差）；row（多步合成序列允许省略辅助试剂）、
    resonance / energy 跳过。"""
    if layout_name != "reaction_mech":
        return ""
    segments, cur, has_arrow = [], [], False
    for child in children:
        if child.type == "STRUCT" and child.args:
            cur.append(child.args[0].strip())
        elif child.type == "RXNARROW":
            has_arrow = True
            segments.append(cur)
            cur = []
    segments.append(cur)
    if not has_arrow:
        return ""
    for i in range(len(segments) - 1):
        reason = _balance_reason(_sum_species(segments[i]),
                                 _sum_species(segments[i + 1]),
                                 strict_h=False, step=f"第 {i + 1} 步")
        if reason:
            return reason
    return ""


def _validate_struct_args(args: list) -> Tuple[bool, str]:
    """校验单个 STRUCT 参数（顶层或容器内）：SMILES 非空 + label 长度
    + label 化学式一致性（化学校验 T2-2）。"""
    if not args or not args[0] or not args[0].strip():
        return False, "SMILES 为空"
    ok, reason = _label_ok(args[1] if len(args) > 1 else None)
    if not ok:
        return False, reason
    smi = args[0].strip()
    if not _smiles_ok(smi):
        return False, f"无效 SMILES「{smi}」"
    if _RDKIT_OK and len(args) > 1 and args[1]:
        reason = _check_label_formula(smi, str(args[1]))
        if reason:
            return False, reason
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


def _check_xh_h_usage(mol, idxs: list, what: str) -> str:
    """A2：XH 叠加次数不得超过原子可用隐含 H 数（防"幽灵 H"——
    渲染端纯几何放置，不看 GetTotalNumHs）。fake mol（无 GetAtomWithIdx）
    时跳过。返回错误原因或 ""。"""
    gai = getattr(mol, "GetAtomWithIdx", None)
    if gai is None or not idxs:
        return ""
    for a, cnt in Counter(idxs).items():
        avail = gai(a).GetTotalNumHs()
        if cnt > avail:
            return (f"{what}原子 {a} 可用隐含 H 为 {avail} 个，"
                    f"不足以画出 {cnt} 个")
    return ""


def _validate_composite(layout: str, children: list) -> Tuple[bool, str]:
    """COMPOSITE 容器校验：布局合法 + 子标记递归校验 + 引用存在性 + at= 越界。"""
    header = [p.strip() for p in (layout or "").split(",")]
    layout_name = header[0]
    if layout_name not in COMPOSITE_LAYOUTS:
        return False, f"未知布局「{layout_name}」，支持 {'/'.join(COMPOSITE_LAYOUTS)}"

    # 收集组件：id → {smiles, at}（attrs 由 tag_parser 结构化提取，不再从 raw 二次解析）
    comps = {}
    for child in children:
        if child.type != "STRUCT":
            continue
        cid = child.attrs.get("id") or f"r{len(comps)}"
        comps[cid] = {
            "smiles": child.args[0].strip() if child.args and child.args[0] else "",
            "at": child.attrs.get("at"),
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
    comp_mols = {}
    if _RDKIT_OK:
        for cid, info in comps.items():
            if info["smiles"]:
                try:
                    mol = _parse_mol(info["smiles"])
                except Exception:
                    mol = None
                atom_counts[cid] = mol.GetNumAtoms() if mol else 0
                comp_mols[cid] = mol

    xh_usage = {}
    for child in children:
        ctype = child.type
        if ctype == "MECHARROW":
            if not child.args or not child.args[0]:
                return False, "MECHARROW 为空"
            for spec in child.args[0].split(","):
                m = _MECH_ARROW_RE.match(spec)
                if not m:
                    return False, f"MECHARROW 格式错误「{spec}」"
                src_id, src_pt, _, dst_id, dst_pt, dst2_id, dst2_pt = m.groups()
                if src_id not in comps or dst_id not in comps:
                    return False, f"MECHARROW 引用未知组件「{src_id}→{dst_id}」"
                if dst2_id is not None:
                    if dst2_id not in comps:
                        return False, f"MECHARROW 引用未知组件「{dst2_id}」"
                    if "-" in dst_pt:
                        return False, f"MECHARROW 成键空白位端点格式错误「{spec}」"
                if _RDKIT_OK:
                    if not _validate_mech_arrow_pt(src_pt, atom_counts.get(src_id, 0)):
                        return False, f"MECHARROW 源端点「{src_id}:{src_pt}」超出原子范围"
                    if not _validate_mech_arrow_pt(dst_pt, atom_counts.get(dst_id, 0)):
                        return False, f"MECHARROW 目标端点「{dst_id}:{dst_pt}」超出原子范围"
                    if dst2_id is not None and not _validate_mech_arrow_pt(
                            dst2_pt, atom_counts.get(dst2_id, 0)):
                        return False, f"MECHARROW 目标端点「{dst2_id}:{dst2_pt}」超出原子范围"
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
                xh_usage.setdefault(ref, []).append(i)
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
                # A1：a-b 必须真实成键（渲染端对不存在的键静默跳过）
                gba = getattr(comp_mols.get(ref), "GetBondBetweenAtoms", None)
                if gba is not None and gba(a, b) is None:
                    return False, (f"BOND 键 {a}-{b} 在组件 {ref} 中不存在"
                                   f"（原子 {a} 与 {b} 之间没有化学键）")

    if _RDKIT_OK:
        for ref, idxs in xh_usage.items():
            reason = _check_xh_h_usage(comp_mols.get(ref), idxs, f"组件 {ref} ")
            if reason:
                return False, reason
        reason = _check_composite_balance(children, layout_name)
        if reason:
            return False, reason
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
    if ttype == "REACTION":
        # 通用 SMILES 字段检查 + 原子守恒（化学校验 T2-3）
        smi_list = _SMILES_FIELDS["REACTION"](args)
        if not smi_list:
            return ValidationResult(tag, False, "缺少 SMILES 字段")
        for smi in smi_list:
            if not smi:
                return ValidationResult(tag, False, "SMILES 为空")
            if not _smiles_ok(smi):
                return ValidationResult(tag, False, f"无效 SMILES「{smi}」")
        if _RDKIT_OK:
            reason = _check_reaction_balance(args)
            if reason:
                return ValidationResult(tag, False, reason)
        return ValidationResult(tag, True)
    if ttype in ("XH", "BOND"):
        # 顶层形式（[XH:SMILES|序号,...] / [BOND:SMILES|a-b,...]）；
        # 容器内子标记形式（id 引用）由 _validate_composite 处理
        if not args or not args[0] or not args[0].strip():
            return ValidationResult(tag, False, "SMILES 为空")
        smi = args[0].strip()
        if not _smiles_ok(smi):
            return ValidationResult(tag, False, f"无效 SMILES「{smi}」")
        spec = (args[1] if len(args) > 1 else "").strip()
        if not spec:
            return ValidationResult(tag, False, f"{ttype} 缺少标注参数")
        if _RDKIT_OK:
            mol = _parse_mol(smi)
            n = mol.GetNumAtoms() if mol else 0
            xh_idxs = []
            for tok in spec.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if ttype == "XH":
                    try:
                        i = int(tok)
                    except ValueError:
                        return ValidationResult(
                            tag, False, f"XH 原子编号「{tok}」不是数字")
                    if not 0 <= i < n:
                        return ValidationResult(
                            tag, False, f"XH 原子编号 {i} 超出范围 0~{n - 1}")
                    xh_idxs.append(i)
                else:
                    m = re.fullmatch(r"(\d+)-(\d+)", tok)
                    if not m:
                        return ValidationResult(
                            tag, False, f"BOND 键引用格式错误「{tok}」")
                    a, b = int(m.group(1)), int(m.group(2))
                    if not (0 <= a < n and 0 <= b < n):
                        return ValidationResult(
                            tag, False, f"BOND 键 {a}-{b} 超出范围 0~{n - 1}")
                    # A1：a-b 必须真实成键（渲染端对不存在的键静默跳过）
                    gba = getattr(mol, "GetBondBetweenAtoms", None)
                    if gba is not None and gba(a, b) is None:
                        return ValidationResult(
                            tag, False,
                            f"BOND 键 {a}-{b} 不存在（原子 {a} 与 {b} 之间没有化学键）")
            # A2：XH 叠加不超过原子可用隐含 H 数
            reason = _check_xh_h_usage(mol, xh_idxs, "XH ")
            if reason:
                return ValidationResult(tag, False, reason)
        return ValidationResult(tag, True)

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
    """校验失败标记的降级提示文本。

    用户可见版本：去掉内部校验类别前缀（化学校验：）与括号内详情/修正指导，
    只留主因（如"方程式两侧原子不守恒"）——完整原因（含元素计数明细）仍
    通过 P2 修正 prompt 与 metrics 详情供内部使用。
    """
    msg = reason
    if msg.startswith(_CHEM_PREFIX):
        msg = msg[len(_CHEM_PREFIX):].split("（", 1)[0].strip()
    return f"（{tag_name(tag.type)}图示无法渲染：{msg}，已省略）"


if __name__ == "__main__":
    # 冒烟测试（不依赖 rdkit）
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # 含故意错误示例：无效 SMILES / 越界 at=（演示校验拦截）
    demo = (
        "[STRUCT:c1ccccc1] "
        "[STRUCT:XYZXYZ,label=无效结构] "
        "[ENERGY:0,108,-20] "
        "[ENERGY:abc] "
        "[NEWMAN:CC,60] [NEWMAN:CC,xyz] "
        # 故意错误示例：at=9 越界（演示校验拦截）
        "[COMPOSITE:energy][ENERGY:0,108,-20]"
        "[STRUCT:CCl,label=反应物,at=0][STRUCT:CO,label=产物,at=9]"
        "[/COMPOSITE]"
    )
    from .tag_parser import parse_tags
    for t in parse_tags(demo):
        r = validate_tag(t)
        print(("✓" if r.ok else "✗"), t.type, t.raw[:60], "" if r.ok else f"— {r.reason}")
