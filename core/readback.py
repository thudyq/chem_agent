# -*- coding: utf-8 -*-
"""core/readback.py — 生成-回读协议的确定性摘要与身份保持检查。

回读摘要：RDKit 从模型自己写的 SMILES 算出的确定性事实清单（分子式/重原子/
电荷/自由基/环系/手性中心）——独立通道，回传模型对照自身声称，
把"找错"从模型手里拿走（Tyen et al.：模型不会找错但会改错）。
摘要解析口径与校验层一致（tag_validator._parse_mol，显式 H 保留）——
摘要里的原子编号与 MECHARROW/校验报错同口径，可直接对照。

身份保持：修正前后逐物种比对骨架——环系物种取 Bemis–Murcko 骨架
canonical SMILES；无环物种取拓扑骨架（键级/电荷/自由基抹平的
canonical SMILES——质子化/电荷修正正是回读要修的对象，不算漂移；
碳链/环系改变才算）。以"合法"为目标的修正循环会把分子改成"合法的
另一个分子"（SmiSelf/AMREC），故漂移必须作为独立信号记录
（本模块只判定记录，拒绝与否由调用方决定）。
"""

from collections import Counter

from core.tag_validator import _parse_coeff, _parse_mol, iter_struct_components

try:
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit.Chem.Scaffolds import MurckoScaffold
    _RDKIT_OK = True
except ImportError:          # 与 tag_validator 同策略：无 RDKit 时降级为空操作
    Chem = None
    CalcMolFormula = None
    MurckoScaffold = None
    _RDKIT_OK = False


# ---------------------------------------------------------------- 摘要


def summarize_species(smiles: str, label: str = "", cid: str = "") -> str:
    """单一物种的 RDKit 确定性摘要（单行紧凑结构化文本）。

    化学式文本组件（双轨制，无原子结构）与无法解析的 SMILES 产出说明性
    短句——摘要是确定性事实通道，没有事实时明说，不产出伪事实。
    """
    head = cid or ""
    if label:
        head += f"「{label}」"
    head = head or "（未标注物种）"
    parsed = _parse_coeff(smiles.strip()) if smiles else None
    if parsed is None:
        return f"{head}：SMILES 为空或系数非法，无法生成摘要"
    bare = parsed[1]
    if not _RDKIT_OK:
        return f"{head}：{bare}（无 RDKit，摘要不可用）"
    mol = _parse_mol(bare)
    if mol is None:
        return f"{head}：{bare}——化学式文本组件/无法解析，无原子结构，不参与摘要"
    parts = [f"{head}：{bare}",
             f"分子式 {CalcMolFormula(mol)}",
             f"重原子 {mol.GetNumHeavyAtoms()}"]
    charged = [(a.GetIdx(), a.GetSymbol(), a.GetFormalCharge())
               for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
    q = sum(c for _, _, c in charged)
    parts.append(f"净电荷 {q:+d}" if q != 0 else "净电荷 0")
    if charged:
        parts.append("带电原子 " + "、".join(f"{i}:{s}({c:+d})"
                                             for i, s, c in charged))
    rads = [(a.GetIdx(), a.GetSymbol(), a.GetNumRadicalElectrons())
            for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    if rads:
        parts.append(f"自由基单电子 {sum(n for _, _, n in rads)}（"
                     + "、".join(f"{i}:{s}" for i, s, _ in rads) + "）")
    rings = []
    for ring in mol.GetRingInfo().AtomRings():
        hetero = [mol.GetAtomWithIdx(i).GetSymbol() for i in ring
                  if mol.GetAtomWithIdx(i).GetAtomicNum() != 6]
        rings.append(f"{len(ring)} 元环"
                     + (f"（含{'/'.join(hetero)}）" if hetero else ""))
    if rings:
        parts.append("环系 " + "、".join(rings))
    # 手性中心（含未指派）：_parse_mol 走 sanitize=False，立体化学需补赋值
    # （与 _check_cistrans_label 同口径）。只报计数与 R/S 分布（集合级），
    # 不带 SMILES 原子序——label 用的是 IUPAC 位次（如 (2R,3S)），括号里写
    # 原子序会诱使模型混淆两种编号、把对的结构翻成对映体（校验层的
    # CIP 检查同样刻意集合级）
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    if centers:
        dist = Counter(d for _, d in centers)
        parts.append(f"手性中心 {len(centers)}（"
                     + "、".join(f"{k}×{v}" for k, v in sorted(dist.items()))
                     + "）")
    return "；".join(parts)


def summarize_tag(tag) -> str:
    """标记内全部 STRUCT 组件（含 COMPOSITE/BLOCK 嵌套）的逐行摘要；
    无 STRUCT 组件（ENERGY/REASONING 等）返回 ""。"""
    if tag.type == "STRUCT":
        comps = [(tag.attrs.get("id") or "s0", tag)]
    elif tag.type == "COMPOSITE" and len(tag.args) >= 2:
        comps = list(iter_struct_components(tag.args[1]))
    else:
        return ""
    lines = []
    for cid, child in comps:
        smi = child.args[0] if child.args else ""
        label = child.args[1] if len(child.args) > 1 else ""
        lines.append(summarize_species(smi or "", label=label or "", cid=cid))
    return "\n".join(lines)


# ---------------------------------------------------------------- 身份保持


def _identity_key(smiles: str) -> str | None:
    """物种身份键（模块 docstring 所述骨架口径）；解析失败返回 None。"""
    if not _RDKIT_OK:
        return None
    parsed = _parse_coeff(smiles.strip()) if smiles else None
    if parsed is None:
        return None
    # 骨架判定用默认 sanitize 解析（折叠显式 H；无需与校验同口径的原子编号）
    from utils.rdkit_utils import parse_smiles
    mol = parse_smiles(parsed[1])
    if mol is None:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return "scaf:" + Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    cp = Chem.RWMol(mol)
    for bond in cp.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in cp.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetIsAromatic(False)
        atom.SetIsotope(0)
        # 隐式/显式 H 一并抹平——否则质子化写法（CC[OH+]CC 的 O 带一个
        # 隐式 H）会与中性写法（CCOCC）得到不同拓扑键，把合法的电荷修正
        # 误判为漂移
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    return "topo:" + Chem.MolToSmiles(cp.GetMol())


def _species_keys(tag) -> list:
    """标记的 [(cid, 身份键)] 列表（无 STRUCT 返回 []）。"""
    if tag.type == "STRUCT":
        comps = [(tag.attrs.get("id") or "s0", tag)]
    elif tag.type == "COMPOSITE" and len(tag.args) >= 2:
        comps = list(iter_struct_components(tag.args[1]))
    else:
        return []
    return [(cid, _identity_key(ch.args[0] if ch.args else ""))
            for cid, ch in comps]


def identity_drift(old_raw: str, new_raw: str) -> list:
    """修正前后两个标记原文的骨架漂移清单（[] = 身份保持）。

    逐物种比对身份键：优先按 cid 匹配，无 id 时按出现顺序。物种数不同、
    键不同均计漂移；任一侧无法解析（None 键）跳过该对（宁漏勿拦）。
    调用方分别是同类型标记（修正闭环已按类型+身份令牌对齐）。
    """
    from core.tag_parser import parse_tags
    old_tags = parse_tags(old_raw)
    new_tags = parse_tags(new_raw)
    if not old_tags or not new_tags:
        return []
    old = _species_keys(old_tags[0])
    new = _species_keys(new_tags[0])
    drift = []
    if len(old) != len(new):
        drift.append(f"物种数变化（{len(old)} → {len(new)}）")
    new_by_cid = {}
    for cid, key in new:
        new_by_cid.setdefault(cid, key)
    for pos, (cid, old_key) in enumerate(old):
        if cid in new_by_cid:
            new_key = new_by_cid[cid]
        elif pos < len(new):
            new_key = new[pos][1]
        else:
            continue  # 物种缺失已计入物种数变化
        if old_key is None or new_key is None:
            continue  # 无法解析——跳过（宁漏勿拦）
        if old_key != new_key:
            drift.append(f"组件 {cid}：骨架/拓扑变化（{old_key} → {new_key}）")
    return drift


# ---------------------------------------------------------------- 高风险界定

_HIGH_RISK_MODES = ("stereo", "chair", "newman")
# label 环系/立体声明（结构检查之外的补网：声称了但 SMILES 没写立体/环时）
_HIGH_RISK_LABEL_RE = None


def is_high_risk(tag) -> bool:
    """高风险标记界定（回读协议只对这些启用）：机理 COMPOSITE（含
    MECHARROW/BLOCK）、多组分/盐、电荷/自由基、环系、立体（@ / 方向键 /
    stereo/chair/newman 画法），或 label 带环系/立体声明——简单平面小分子
    RDKit 已完全裁决，加回读只会引入"把对的改错"。
    REASONING/ENERGY → False。"""
    global _HIGH_RISK_LABEL_RE
    if _HIGH_RISK_LABEL_RE is None:
        import re
        _HIGH_RISK_LABEL_RE = re.compile(r"鎓|endo|exo|顺|反|内型|外型")
    if tag.type == "STRUCT":
        comps = [(tag.attrs.get("id") or "s0", tag)]
        children = []
    elif tag.type == "COMPOSITE" and len(tag.args) >= 2:
        children = tag.args[1]
        comps = list(iter_struct_components(children))
    else:
        return False
    for c in children if isinstance(children, list) else []:
        if c.type in ("MECHARROW", "BLOCK"):
            return True
    for _cid, child in comps:
        if child.attrs.get("mode") in _HIGH_RISK_MODES:
            return True
        smi = (child.args[0] or "").strip() if child.args else ""
        parsed = _parse_coeff(smi) if smi else None
        bare = parsed[1] if parsed else ""
        if any(k in bare for k in (".", "@", "/", "\\")):
            return True
        label = child.args[1] if len(child.args) > 1 else ""
        if label and _HIGH_RISK_LABEL_RE.search(str(label)):
            return True
        if not _RDKIT_OK or not bare:
            continue
        mol = _parse_mol(bare)
        if mol is None:
            continue
        if any(a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() > 0
               for a in mol.GetAtoms()):
            return True
        if mol.GetRingInfo().NumRings() > 0:
            return True
    return False
