# -*- coding: utf-8 -*-
"""utils/rdkit_utils.py — RDKit 验证与分子式工具（转型后保留）。"""

import contextlib
import re

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

# 游离氢组分（[H+]/[H]/[H-] 孤立 H，前后是 . 或边界）：合法组分
# （质子/氢自由基/氢负离子），但 RDKit 解析（AdjustHs）与坐标计算
# （RemoveHs）会对孤立 H 打 "not removing hydrogen atom without neighbors"
# 警告（无害，C4）。
FREE_H_COMPONENT_RE = re.compile(r"(?:^|(?<=\.))\[H[+-]?\](?=\.|$)")

# H 数字前缀的化学式习惯写法（非法 SMILES）：[H3O+] / [H2O] / [H3N+]。
# 方括号内以 H 开头、后跟数字 n 与元素符号 X 及可选的电荷（+/-/数字）——
# LLM 把化学式 H₃O⁺ 误写成 SMILES 时，H3 前缀在 SMILES 语法中非法
# （H 原子不能带隐式 H），RDKit 解析失败；规范化重写为 [XHn...]（[OH3+]）。
# 不匹配 H 原子/离子写法（[H]/[H+]/[H-]，无数字）、同位素（[2H]，
# 数字在 H 前）与元素开头写法（[OH3+]/[CH3]/[NH4+]）→ 原样返回。
_H_PREFIX_SMILES_RE = re.compile(r"\[H(\d+)([A-Z][a-z]?)([+-]?\d*)\]")


def normalize_h_prefix_smiles(smiles: str) -> str:
    """规范化"H 数字前缀"SMILES 写法：[H3O+] → [OH3+]、[H2O] → [OH2]、
    [H3N+] → [NH3+]）。校验层与渲染层共用（tag_validator._parse_mol /
    renderers.mol_primitives.prepare_mol 解析前调用）；未匹配原样返回。
    """
    if not smiles or not isinstance(smiles, str):
        return smiles
    return _H_PREFIX_SMILES_RE.sub(r"[\2H\1\3]", smiles)


# 通用基团缩写（Chemical-Notation.md §二：R 烷基、Ar 芳基、X 卤素、M 金属、
# Me/Et/Pr/Bu 烷基、Ph/Bn 芳基、Ac/Bz/Ts/Tf/Ms 酰基/磺酰基、Boc/Cbz/TBDMS/TMS
# 保护基；补充 n-Pr/i-Pr/n-Bu/i-Bu/s-Bu/t-Bu 带前缀写法）。
# 这些缩写不是合法 SMILES 元素（RDKit 无法解析），LLM 写通用结构时
# （[STRUCT:R-Br]、Ph-OH、AcOH、RC(=O)OEt）应放行——预处理替换为 RDKit
# dummy 原子 [*:n]（atom map），渲染端按 map 显示缩写文本。不误伤真实
# 元素：后边界拒绝小写/数字（Mg/Mn、R1 环标记；允许后接大写元素如
# EtBr 的 Br、MeOH 的 O）；前边界只拒绝小写（OEt/NEt/PhEt 的缩写前是
# 大写元素/括号/边界时允许，Cl 的 l、Ra 的 a 等小写前缀拒绝）。
_GROUP_ABBREVS = (
    "TBDMS", "Cbz", "Boc", "TMS",            # 保护基（长优先）
    "n-Pr", "i-Pr", "t-Bu", "s-Bu", "i-Bu", "n-Bu",  # 带前缀烷基
    "Tf", "Ts", "Bn", "Bz", "Ac", "Ph", "Ar",        # 双字母
    "Bu", "Pr", "Et", "Me", "Ms",                   # 双字母烷基/磺酰
    "R'", "R", "X", "M",                            # 单字母占位（R' 优先于 R）
)
_ABBR_ALT = "|".join(_GROUP_ABBREVS)
# 方括号包裹的缩写（[Et]、[R]、[Ph]）：整体替换为 [*:n]，不残留外层括号
# （否则产生 [[*:n]] 双括号非法）；捕获组取括号内文本作显示缩写。
_ABBR_BRACKET_RE = re.compile(r"\[(" + _ABBR_ALT + r")\]")
# R/X 后允许编号（R1/R2/X1 等，化学上表示不同的 R 基团，渲染为下标 R₁）；
# 在 _ABBR_ATOM_RE 之前处理（R 的裸匹配后边界拒绝数字，互不冲突）。
_ABBR_NUM_RE = re.compile(r"(?<![a-z])([RX])(\d+)(?![a-z])")
# 裸缩写：前边界非小写（允许 OEt/NEt/PhEt 等大写前缀），后边界非小写/数字
_ABBR_ATOM_RE = re.compile(
    r"(?<![a-z])(" + _ABBR_ALT + r")(?![a-z0-9])")

# 化学式后缀（非法 SMILES 写法 → 标准原子）：缩写替换后处理，顺序敏感。
# COOH → C(=O)O（羧基）；-CHO → C=O（醛基）；-NH2 → N（氨基）；
# OH → O（羟基，-OH 的 H 并入 O 隐式氢：Ph-OH → Ph-O、C-OH → CO）。
# 前边界排除 `[`：方括号内是合法 SMILES 原子（[OH-]、[OH2+]、[NH2]），
# 不得替换（否则 [OH-] 的 H 被吞）。
_SUFFIX_REPLACEMENTS = [
    (re.compile(r"(?<!\[)(?<![A-Za-z])COOH(?![A-Za-z0-9])"), "C(=O)O"),
    (re.compile(r"(?<!\[)(?<![A-Za-z])CHO(?![A-Za-z0-9])"), "C=O"),
    (re.compile(r"(?<!\[)(?<![A-Za-z])NH2(?![A-Za-z0-9])"), "N"),
    (re.compile(r"(?<!\[)(?<![A-Za-z])OH(?![A-Za-z0-9])"), "O"),
]

# dummy 原子连裸金属（有机金属试剂 BuLi / PhMgBr 等）在 RDKit 中解析失败
# （[*]Li 语法错误），金属需方括号形式（[*][Li]、[*][Mg]Br）。卤素/主族
# 非金属（[*]Br、[*]O、[*]C）不受影响，不加括号。
_METAL_WRAP_RE = re.compile(
    r"(\[\*:\d+\])(Li|Na|K|Rb|Cs|Mg|Ca|Sr|Ba|Al|Ga|In|Tl|Fe|Co|Ni|"
    r"Cu|Zn|Ag|Cd|Au|Hg|Sn|Pb)")


def expand_group_abbrevs(smiles: str):
    """通用基团缩写 → RDKit dummy 原子（[*:n]，atom map）。

    返回 (规范 SMILES, {map 号: 缩写文本})：
        R-Br  → ([*:1]Br, {1: "R"})
        Ph-OH → ([*:1]O,  {1: "Ph"})   （-OH 一并规范化为 O）
        AcOH  → ([*:1]O,  {1: "Ac"})   （MeOH/EtOH/PhOH 同）
        BuLi  → ([*:1][Li], {1: "Bu"}) （金属加方括号）
        PhCOOH→ ([*:1]C(=O)O, {1: "Ph"})
    校验层与渲染层共用；未匹配返回 (原串, {})。
    """
    if not smiles or not isinstance(smiles, str):
        return smiles, {}
    mapping = {}

    def _sub(m):
        idx = len(mapping) + 1
        mapping[idx] = m.group(1) if m.lastindex else m.group(0)
        return f"[*:{idx}]"

    def _sub_num(m):
        # R/X 编号（R1/X2）：记录完整文本（含数字），渲染端转下标
        idx = len(mapping) + 1
        mapping[idx] = m.group(0)
        return f"[*:{idx}]"

    out = _ABBR_BRACKET_RE.sub(_sub, smiles)
    out = _ABBR_NUM_RE.sub(_sub_num, out)
    out = _ABBR_ATOM_RE.sub(_sub, out)
    for pat, rep in _SUFFIX_REPLACEMENTS:
        out = pat.sub(rep, out)
    out = _METAL_WRAP_RE.sub(r"\1[\2]", out)
    return out, mapping


@contextlib.contextmanager
def mute_rdkit_warnings(include_error: bool = False):
    """局部屏蔽 RDKit 日志（进程级开关，仅影响日志输出）。

    include_error=False（默认）：仅屏蔽 rdApp.warning（无害警告，如孤立氢）。
    include_error=True：同时屏蔽 rdApp.error——用于"探测性"解析（如条件 token
    试解析 SMILES，失败是常态），此时解析错误属噪音。
    并发调用时最坏情况是漏掉一条无关日志，可接受。
    """
    RDLogger.DisableLog("rdApp.warning")
    if include_error:
        RDLogger.DisableLog("rdApp.error")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.warning")
        if include_error:
            RDLogger.EnableLog("rdApp.error")


def parse_smiles(smiles: str):
    """Chem.MolFromSmiles 包装：含游离氢组分（合法）时局部屏蔽其无害警告。
    返回 Mol；非法/空返回 None。
    """
    if not smiles or not isinstance(smiles, str):
        return None
    cm = (mute_rdkit_warnings() if FREE_H_COMPONENT_RE.search(smiles)
          else contextlib.nullcontext())
    with cm:
        return Chem.MolFromSmiles(smiles)


def validate_smiles(smiles: str) -> bool:
    """校验 SMILES 是否合法。合法返回 True，否则 False。"""
    return parse_smiles(smiles) is not None


def mol_to_formula(smiles: str) -> str:
    """SMILES -> 分子式（如 'CH4'）；非法或空返回空串。"""
    mol = parse_smiles(smiles)
    if mol is None:
        return ""
    return CalcMolFormula(mol)


if __name__ == "__main__":
    # Day 1 完成标志验证：validate_smiles('C') 返回 True
    print("validate_smiles('C') =", validate_smiles("C"))
    print("validate_smiles('XYZ') =", validate_smiles("XYZ"))
    print("mol_to_formula('c1ccccc1') =", mol_to_formula("c1ccccc1"))
