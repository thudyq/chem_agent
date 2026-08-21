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

import contextlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple

from .tag_parser import RenderTag

# rdkit 可用性探测：缺失时跳过 SMILES / 原子数语义校验（渲染器内部会兜底）
try:
    from rdkit import Chem  # noqa: F401
    from utils.rdkit_utils import FREE_H_COMPONENT_RE, mute_rdkit_warnings, \
        normalize_h_prefix_smiles, expand_group_abbrevs
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False
    FREE_H_COMPONENT_RE = None
    mute_rdkit_warnings = None
    normalize_h_prefix_smiles = None
    expand_group_abbrevs = None


def _parse_mol(smiles: str):
    """Chem.MolFromSmiles 局部包装：游离氢组分（合法）的无害警告静默。
    保持经本模块命名空间调用 Chem（fake_rdkit fixture 可替换）。

    **20260821：显式 H 保留解析**——XH 并入 STRUCT 后显式 H 是真实原子
    参与编号（如 C([H])([H])([H])[H] 的 1~4 号），校验必须与渲染端
    prepare_mol 同口径（sanitize=False + UpdatePropertyCache，保留显式 H
    原子），否则 atom_counts 少算 H、MECHARROW/HBOND/NEWMAN 序号全偏。
    fake_rdkit（MolFromSmiles 单参 lambda）或解析异常时回退默认解析。

    校验是**探测性解析**（非法 SMILES 是常态输入，P1 要拦截并提示），
    失败时的 RDKit 日志（SMILES Parse Error / Explicit valence 超限）对
    用户与日志都无价值——统一屏蔽 rdApp.error，避免终端被噪音刷屏
    （Drawbacks 九 C-1：O 价态 4 超限的 Explicit valence 日志）。

    H 数字前缀写法（[H3O+]）先经 normalize_h_prefix_smiles 规范化为
    合法 SMILES（[OH3+]）再解析；通用基团缩写（R/X/Ph/Ac 等）经
    expand_group_abbrevs 替换为 dummy 原子（[*:n]）后解析
    （20260815：化学式习惯误写与通用基团占位放行）。
    """
    if normalize_h_prefix_smiles is not None:
        smiles = normalize_h_prefix_smiles(smiles)
    if expand_group_abbrevs is not None:
        smiles, _ = expand_group_abbrevs(smiles)

    def _parse_default(smi):
        # 回退路径：默认 sanitize 解析（折叠显式 H）；fake_rdkit / 异常时用
        if mute_rdkit_warnings is None:
            return Chem.MolFromSmiles(smi)
        with mute_rdkit_warnings(include_error=True):
            return Chem.MolFromSmiles(smi)

    # 显式 H 保留解析（与渲染端 prepare_mol 同口径）：sanitize=False +
    # UpdatePropertyCache，保留显式 H 原子参与编号。fake_rdkit（单参
    # lambda / 无 SanitizeFlags）或解析异常时回退默认解析。
    try:
        if mute_rdkit_warnings is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
        else:
            with mute_rdkit_warnings(include_error=True):
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
        return mol
    except (TypeError, AttributeError):
        return _parse_default(smiles)
    except Exception:
        return _parse_default(smiles)

# label 长度硬上限（字符数）。prompt 建议 ≤10（中文 ≤6），此处为兜底硬拦截
LABEL_MAX_LEN = 24

# COMPOSITE 支持的布局（与 renderers/composite.py 保持一致）：
# reaction（反应式/多步序列 + 守恒 + 机理）、row（横向排列）、energy（势能面）
COMPOSITE_LAYOUTS = ("reaction", "row", "energy")

# 与 renderers/composite.py 相同的引用/端点提取正则
# 端点支持两种：原子序号（0，含显式 H 原子）、键中点（0-1）。
# 与 renderers/composite.py 的 _MECH_PT_RE 保持一致。
# （20260821：XH 并入 STRUCT 后显式 H 是真实原子参与编号，a#k 语法废弃——
# 端点只支持原子序号（含显式 H 原子）与 a-b 键中点。）
_MECH_PT_RE = r"\d+(?:-\d+)?"
_MECH_ARROW_RE = re.compile(
    rf"^\s*([A-Za-z0-9_]+)\s*:\s*({_MECH_PT_RE})\s*(>>|>)\s*"
    rf"([A-Za-z0-9_]+)\s*:\s*({_MECH_PT_RE})"
    r"(?:\s*\+\s*([A-Za-z0-9_]+)\s*:\s*(\d+))?\s*$"
)

# 标记类型 → 中文名（降级提示用）
_TAG_NAMES = {
    "STRUCT": "结构式",
    "COMPOSITE": "复合图",
    "ENERGY": "势能面",
    "HBOND": "氢键标注",
}

# SMILES 字段提取器：输入 RenderTag，返回需要校验的 SMILES 字符串列表。
# 返回 None 表示该标记类型不携带 SMILES（如 ENERGY / PLUS）。
_SMILES_FIELDS = {
    "STRUCT": lambda a: [a[0]] if a and a[0] else [],
    "HBOND": lambda a: [a[0]] if a and a[0] else [],
}


def _split_multi(seg: str) -> list:
    """把多组分段拆为 SMILES 列表（过滤空串）。

    契约分隔符为分号；LLM 偶发用逗号分隔或写出尾逗号（SMILES 不含逗号），
    按 [;,] 拆分归一化。
    """
    return [s.strip() for s in re.split(r"[;,]", seg or "") if s.strip()]


# 系数前缀：整数（2CCO）或 n/2（n 为奇数，1/2O2、3/2O2）。负系数仅用于
# 箭头补足（2b，-H2O），不用于物种列表。
_COEFF_RE = re.compile(r"^(-?\d+)(?:/(\d+))?")


def _parse_coeff(token: str) -> tuple | None:
    """解析系数前缀：返回 (coeff, 余下文本)；非法系数返回 None。

    允许：无系数（1）、正整数（2）、负整数（-1，仅箭头补足）、n/2（n 奇数，
    如 1/2、3/2、-1/2）。其他分数（1/3、2/3）与"0"拒绝。
    """
    m = _COEFF_RE.match(token.strip())
    if not m:
        return (1, token.strip())
    num, den = int(m.group(1)), m.group(2)
    if num == 0:
        return None
    if den is None:
        return (num, token.strip()[m.end():].strip())
    if int(den) != 2 or num % 2 == 0:
        return None  # 仅允许 n/2（n 奇数）
    return (num / 2.0, token.strip()[m.end():].strip())


def _split_multi_coeff(seg: str) -> list:
    """多组分段拆为 (coeff, smiles) 列表（过滤空串/非法系数）。

    系数缺失视为 1；非法系数（0、1/3、2/3）的组分整体丢弃（校验层另行
    报格式错误，此处只负责安全拆分）。
    """
    out = []
    for tok in _split_multi(seg):
        parsed = _parse_coeff(tok)
        if parsed is not None and parsed[1]:
            out.append(parsed)
    return out


@dataclass
class ValidationResult:
    """单个标记的校验结果。"""

    tag: RenderTag
    ok: bool
    reason: str = ""
    warnings: list = None  # 软提示（非拦截）：通过但附提醒，如苯环写法不一致

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


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
# 化学校验（T2）：原子守恒（T2-3）。
# 失败原因统一以「化学校验：」前缀，metrics 据此统计化学正确率维度。
# 均为 best-effort：元素计数无法计算（rdkit 缺失/fake mol）时跳过不放行误判。
# ---------------------------------------------------------------------------

_CHEM_PREFIX = "化学校验："

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

# 可识别的真实元素（有机/常见无机，及常见无机/氧化还原元素 Mn、Cr、
# Ba 等——KMnO4、H2SO4、MnSO4、K2SO4、CrCl3 等教科书化学式由此可解析）。
# 刻意不含 Ar（氩）、Ac（锕）等——它们是 prompt 允许的通用基团缩写
# （Ar=芳基、Ac=乙酰基，见主提示 STRUCT 条目），
# 误判为化学式会把缩写 label 打回。
_REAL_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I",
    "Li", "Na", "K", "Mg", "Ca", "Al", "Fe", "Cu", "Zn", "Ag", "Au",
    "Hg", "Pb", "Sn", "Se", "Te",
    "Be", "Sc", "Ti", "V", "Cr", "Mn", "Co", "Ni", "Ga", "Ge", "As",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Cd", "In",
    "Sb", "Cs", "Ba", "La", "Ce", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Tl", "Bi",
}


def _parse_plain_formula(text: str):
    """把纯化学式（CH3Cl / OH- / NO2+ / H3O+ / FeBr4- 等）解析为候选
    (元素计数 dict, 净电荷) 列表；非纯化学式返回 []——含中文/空格/结构括号、
    通用基团缩写（R/Ar/X/Ph 等）、或含不可识别"元素"（如占位字母 A）一律跳过。

    尾电荷的数字归属有歧义（NO2+ 是 N1O2 带 +1，Ca2+ 是 Ca 带 +2），按
    启发式消解并排序（正确解读排在 cands[0]，调用方取第一个即可）：
    - 尾数字串为空 → 唯一候选（电荷 ±1）；
    - 尾数字串 ≥2 位 → 最后一位归电荷、前面归元素下标（SO42- → SO4 带 -2、
      Cr2O72- → Cr2O7 带 -2）；
    - 尾数字 1 位：元素部分 ≥2 个元素 → 数字归元素（FeBr4- → FeBr4 带 -1、
      NO2+ → NO2 带 +1、NH4+ → NH4 带 +1）；单元素 → 数字归电荷
      （Ca2+ → Ca 带 +2、Al3+ → Al 带 +3）。
    残余本质歧义：单元素双关（如 O2- = 超氧根 O₂⁻ 或 O²⁻）语法无法消除，
    本函数取"归电荷"解读（O²⁻）；精确物种请写 SMILES（如超氧根 [O-][O]）。
    """
    s = (text or "").strip()
    if not s:
        return []
    bodies = []
    m = re.search(r"(\d+)?([+-])$", s)
    if m:
        sign = 1 if m.group(2) == "+" else -1
        digits = m.group(1) or ""
        base = s[: m.start()]
        # 歧义消解：正确解读排在候选前面（见 docstring），其余解读作兜底
        multi = len(_FORMULA_TOKEN_RE.findall(base)) >= 2
        if not digits:
            order = [0]
        elif len(digits) >= 2:
            order = list(range(len(digits) - 1, -1, -1)) + [len(digits)]
        elif multi:
            order = [1, 0]
        else:
            order = [0, 1]
        for j in order:
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
    """RDKit Mol → (元素计数 dict（重原子 + 隐式 H）, 净电荷)；失败返回 None。

    通用基团占位符（dummy 原子，原子序 0，expand_group_abbrevs 引入的
    R/X/Ph/Ac）组成未知——不参与元素守恒比对（跳过计数），净电荷不受影响。
    """
    try:
        counts, charge = {}, 0
        for a in mol.GetAtoms():
            if a.GetAtomicNum() == 0:
                continue  # 通用基团占位符：未知组成，不计入元素守恒
            sym = a.GetSymbol()
            counts[sym] = counts.get(sym, 0) + 1
            h = a.GetTotalNumHs()
            if h:
                counts["H"] = counts.get("H", 0) + h
            charge += a.GetFormalCharge()
        return counts, charge
    except Exception:
        return None


def _formula_or_smiles_counts(token: str):
    """物种计数：优先 SMILES（RDKit），失败回退纯化学式（H2O/O2/H2 等）。

    ARROW/REACTION 的物种可以是 SMILES（CCO、c1ccccc1）或教科书化学式
    （O2 氧气、H2 氢气、H2O 水）——后者不是合法 SMILES，按公式数原子。
    返回 (counts, charge)；都无法解析返回 None。
    SMILES 探测为"试探性"解析：失败是常态（条件字段含大量非物种 token，
    如 Δ、140℃、浓H2SO4），RDKit 警告属噪音，静默。
    """
    if mute_rdkit_warnings is not None:
        with mute_rdkit_warnings(include_error=True):
            mol = _parse_mol(token)
    else:
        mol = _parse_mol(token)
    if mol is not None:
        return _mol_counts(mol)
    cands = _parse_plain_formula(token)
    return cands[0] if cands else None


def _hill_str(counts: dict) -> str:
    """元素计数 → Hill 化学式串（错误提示用；不含电荷）。"""
    parts = []
    for sym in sorted(counts, key=lambda s: (s != "C", s != "H", s)):
        n = counts[sym]
        parts.append(sym + (str(n) if n > 1 else ""))
    return "".join(parts)


def _sum_species(species: list):
    """一组 (coeff, smiles) 的元素计数加总（含净电荷，供比对）。

    系数相乘：1/2O2 → 1 个 O；分数原子（如 1/2×奇数个某元素）产生
    非整数计数 → 返回 None（校验层据此拒绝）。任一物种无法解析（既非
    SMILES 也非纯化学式）返回 None。
    """
    total_c, total_q = {}, 0
    for coeff, smi in species:
        mc = _formula_or_smiles_counts(smi)
        if mc is None:
            return None
        for sym, n in mc[0].items():
            v = n * coeff
            if v != int(v):
                return None  # 分数原子（n/2 但该元素计数为奇数）
            total_c[sym] = total_c.get(sym, 0) + int(v)
        total_q += mc[1] * coeff
    return total_c, total_q


def _sum_c_counts(species: list):
    """ARROW 专用 C-only 计数：Σcoeff×C（只比 C，O/H 增减允许）。

    ARROW 为单→单骨架展示，O/H 的分数原子（如 1/2CCOCC 的 0.5 个 O）是
    氧化/脱氢的常态，不参与守恒判定；但 C 的分数（1/2×奇数 C）仍拒绝。
    任一物种无法解析返回 None。
    """
    total = 0
    for coeff, smi in species:
        mc = _formula_or_smiles_counts(smi)
        if mc is None:
            return None
        c = mc[0].get("C", 0) * coeff
        if c != int(c):
            return None  # 分数 C（n/2 但 C 计数为奇数）
        total += int(c)
    return total


def _balance_reason(left, right, strict_h: bool, step: str,
                    check_charge: bool = True) -> str:
    """两侧元素计数比对：非 H 元素必须相等；strict_h 时 H 也必须相等。
    净电荷：check_charge=True 时两侧电荷必须相等；
    False 时保持"旁观离子省略"惯例不比对。"""
    if left is None or right is None:
        return ""
    lc, rc = dict(left[0]), dict(right[0])
    if not strict_h:
        lc.pop("H", None)
        rc.pop("H", None)
    if lc != rc:
        detail = f"{_hill_str(left[0])} vs {_hill_str(right[0])}"
        # 元素差明细（供修正环节定位多写/漏写的物种；正=右侧多，负=右侧少）
        diff = []
        for k in sorted(set(lc) | set(rc)):
            d = rc.get(k, 0) - lc.get(k, 0)
            if d:
                diff.append(f"{k} {d:+d}")
        diff_txt = f"，右侧相对左侧：{'、'.join(diff)}" if diff else ""
        return (f"{_CHEM_PREFIX}{step}两侧原子不守恒（{detail}{diff_txt}，"
                f"请核对物种 SMILES 是否多写/漏写原子；辅助试剂请写入箭头条件而非省略主物种）")
    if check_charge and left[1] != right[1]:
        return (f"{_CHEM_PREFIX}{step}两侧净电荷不守恒"
                f"（{left[1]:+d} vs {right[1]:+d}，需补全离子或修正电荷）")
    return ""


# ---------------------------------------------------------------------------
# 箭头补足物种（REACTION 2b）：条件字段中可解析为具体化学式的 token，
# 无符号前缀 = 反应物侧补足，"-" 前缀 = 产物侧补足。
# 禁止 [O]/[H] 等占位符作为配平物质（prompt 约束；此处解析不到即忽略）。
# ---------------------------------------------------------------------------

# 条件 token 拆分：逗号分隔（含中文逗号）；系数（如 1/2、-1/2）由 _parse_coeff 处理
_ARROW_TOKEN_SPLIT = re.compile(r"[,，]")


def _arrow_supplement_tokens(condition: str) -> list:
    """把条件字段拆为可参与补足的 (coeff, counts, charge, side) token 列表。

    side: "L"（无符号前缀，补反应物侧）或 "R"（- 前缀，补产物侧）。
    不可解析为具体化学式的 token（催化剂、Δ、温度等）被忽略——
    只有"恰好能匹配差额"的 token 才会在 _balance_reason 2b 分支被选中，
    催化剂不匹配差额 → 自然排除，不误判。
    """
    tokens = []
    for raw in _ARROW_TOKEN_SPLIT.split(condition or ""):
        tok = raw.strip()
        if not tok:
            continue
        if tok in ("[O]", "[H]"):
            continue  # 裸占位符禁止作为配平物质（[H+]/[OH-] 等具体离子放行）
        neg = tok.startswith("-")
        body = tok[1:].strip() if neg else tok
        parsed = _parse_coeff(body)
        if parsed is None:
            continue
        coeff, formula = parsed
        cands = _parse_plain_formula(formula)
        if not cands:
            # 括号式具体物种（[H+]/[OH-] 等合法 SMILES 离子）回退 RDKit 计数
            mc = _formula_or_smiles_counts(formula)
            if mc is None:
                continue
            counts, charge = mc
        else:
            counts, charge = cands[0]  # 纯化学式候选唯一（无 SMILES 定夺需求）
        if neg:
            coeff = -coeff
        tokens.append((coeff, counts, charge, "L" if not neg else "R"))
    return tokens


def _sum_supplements(tokens: list) -> tuple | None:
    """箭头补足 token 加总（元素 + 电荷）；分数原子返回 None。"""
    total_c, total_q = {}, 0
    for coeff, counts, charge, side in tokens:
        for sym, n in counts.items():
            v = n * coeff
            if v != int(v):
                return None
            total_c[sym] = total_c.get(sym, 0) + int(v)
        total_q += charge * coeff
    return total_c, total_q


def _arrow_supplement_matches(left, right, condition: str,
                              strict_h: bool, check_charge: bool) -> bool:
    """2b：条件字段中的具体物质 token 子集恰好补足两侧差额。

    无符号 token 补反应物侧、-X 补产物侧；补足成立 ⟺
    Σ(L) - Σ(R) == 差额（元素，strict_h 时含 H；电荷按 check_charge 比对）。
    禁止 [O]/[H] 占位符（_arrow_supplement_tokens 已过滤）。
    """
    tokens = _arrow_supplement_tokens(condition)
    if not tokens:
        return False
    if not strict_h:
        # H 差容忍：token 的 H 计数与两侧 H 一样不参与比对
        tokens = [(c, {k: v for k, v in ct.items() if k != "H"}, q, s)
                  for c, ct, q, s in tokens]
    lc, rc = dict(left[0]), dict(right[0])
    if not strict_h:
        lc.pop("H", None)
        rc.pop("H", None)
    deficit_c = {k: rc.get(k, 0) - lc.get(k, 0)
                 for k in set(lc) | set(rc)}
    deficit_c = {k: v for k, v in deficit_c.items() if v}
    deficit_q = (right[1] - left[1]) if check_charge else 0
    from itertools import combinations
    for r in range(1, len(tokens) + 1):
        for comb in combinations(tokens, r):
            sup = _sum_supplements(list(comb))
            if sup is None:
                continue
            sc, sq = sup
            if ((sq == deficit_q if check_charge else True)
                    and all(sc.get(k, 0) == deficit_c.get(k, 0)
                            for k in set(sc) | set(deficit_c))):
                return True
    return False





# 箭头类型（大一统架构，20260819）：single=正向 → / reversible=可逆 ⇌ /
# resonance=共振 ↔ / retro=逆合成 ⇒
_ARROW_TYPES = ("single", "reversible", "resonance", "retro")


def _explicit_h_heavy(mol, idx: int):
    """idx 为显式 H 原子时返回其重原子邻居序号；否则 None。

    fake mol（测试 fixture）或无邻居信息时返回 None（跳过配对校验）。
    """
    if mol is None or not isinstance(idx, int):
        return None
    gai = getattr(mol, "GetAtomWithIdx", None)
    if gai is None or not (0 <= idx < mol.GetNumAtoms()):
        return None
    atom = gai(idx)
    if getattr(atom, "GetAtomicNum", lambda: -1)() != 1:
        return None
    for n in getattr(atom, "GetNeighbors", list)():
        if getattr(n, "GetAtomicNum", lambda: 0)() != 1:
            return n.GetIdx()
    return None


def _check_proton_transfer_pairing(mech_children: list, comp_mols: dict,
                                   comps: dict) -> str:
    """质子转移配对校验（20260820 基线驱动，化学复审 4b）。

    双电子 MECHARROW 引用显式 H 时必须画全配对（§五 规则 7/10）：
    - Case A：终点为显式 H（碱夺 H）→ 需同组件配套箭头：X—H 键中点 → X
      （键电子落回与 H 相连的重原子）；
    - Case B：起点为含显式 H 的键中点、且终点恰为同组件的重原子 X
      （脱质子、电子落回给体原子）→ 需配套碱孤对指向该 H 的箭头；
      容器内存在游离 [H+] 组件时豁免（自由脱质子，如 EAS 脱质子步）。
    豁免：鱼钩 >>（自由基模式）；X—H 键指向其他原子（氢负离子迁移等，
    H 随电子对移动，非脱质子）；跨组件终点序号巧合（Case B 要求
    dst_id == src_id）。
    """
    arrows = []   # (src_id, src_pt, dst_id, dst_pt)，仅双电子
    for child in mech_children:
        if not child.args or not child.args[0]:
            continue
        for spec in child.args[0].split(","):
            spec = spec.strip()
            if not spec or ">>" in spec:
                continue
            m = _MECH_ARROW_RE.match(spec)
            if not m or m.group(6) is not None:
                continue    # 格式问题/成键空白（自由基场景）由端点校验处理
            arrows.append((m.group(1), m.group(2), m.group(4), m.group(5)))

    for src_id, src_pt, dst_id, dst_pt in arrows:
        # Case A：终点为显式 H → 碱夺 H
        if dst_pt.isdigit():
            heavy = _explicit_h_heavy(comp_mols.get(dst_id), int(dst_pt))
            if heavy is not None:
                bond_pats = {f"{heavy}-{dst_pt}", f"{dst_pt}-{heavy}"}
                if not any(s == dst_id and sp in bond_pats
                           and d == dst_id and dp == str(heavy)
                           for s, sp, d, dp in arrows):
                    return (f"质子转移缺配对箭头：「{src_id}:{src_pt}>{dst_id}:"
                            f"{dst_pt}」的电子落向显式 H（{dst_id}:{dst_pt}），"
                            f"还需画出 X—H 键电子落回 X 的配套箭头"
                            f"（{dst_id}:{heavy}-{dst_pt}>{dst_id}:{heavy}）")
        # Case B：起点为含显式 H 的键中点、终点恰为同组件重原子 → 脱质子
        if "-" in src_pt and dst_pt.isdigit() and dst_id == src_id:
            a, _, b = src_pt.partition("-")
            for h, x in ((a, b), (b, a)):
                if not h.isdigit():
                    continue
                if _explicit_h_heavy(comp_mols.get(src_id), int(h)) is None:
                    continue
                if dst_pt != x:
                    continue    # 键电子去其他原子（氢负离子迁移等），非脱质子
                # 自由脱质子豁免：容器内有游离 [H+] 组件（H+ 已写出，
                # 无需碱夺 H 箭头，如 EAS 脱质子步）
                has_free_proton = any(
                    (c.get("smiles") or "").strip() == "[H+]"
                    for c in comps.values())
                if has_free_proton:
                    continue
                if not any(dp == h for _s, _sp, _d, dp in arrows):
                    return (f"质子转移缺配对箭头：「{src_id}:{src_pt}>{dst_id}:"
                            f"{dst_pt}」是脱质子（X—H 键电子落回 {x}），还需"
                            f"画出碱孤对指向该 H 的配套箭头"
                            f"（如 base:0>{src_id}:{h}）；若是自由脱质子"
                            f"（无碱参与），产物中应写出 [H+] 组件")
    return ""


def _opaque_comp(comps: dict, cid: str) -> bool:
    """组件是否为立体画法组件（stereo/chair/newman，仅展示的不透明单元，
    不支持 MECHARROW/HBOND/CHARGE/XH/BOND 等原子级引用）。"""
    return comps.get(cid, {}).get("mode") in ("stereo", "chair", "newman")


def _formula_comp(comps: dict, cid: str) -> bool:
    """组件是否为化学式文本组件（双轨制，无原子可索引，不支持端点/标注引用）。"""
    return bool(comps.get(cid, {}).get("formula"))


def _check_block(block_children: list) -> Tuple[bool, str]:
    """BLOCK 共振块校验：块内仅 STRUCT / ARROW(type=resonance) / MECHARROW。

    块 = 共振极限式序列（如 [B1 ↔ B2 ↔ B3]），作为单一复合结构参与外层
    序列；块内 MECHARROW 表示共振式间电子流向转化（20260820 支持），
    其引用存在性与原子范围由外层统一校验（块内组件注册进全局组件表，
    支持跨块混合引用）。
    """
    has_struct = False
    for c in block_children:
        if c.type == "STRUCT":
            has_struct = True
            ok, reason = _validate_struct_args(c.args, c.attrs)
            if not ok:
                return False, f"块内组件 {c.attrs.get('id', '')}: {reason}"
        elif c.type == "ARROW":
            a_type = (c.args[0] if c.args else "") or "single"
            if a_type != "resonance":
                return False, f"BLOCK 内箭头类型应为 resonance（当前 {a_type}）"
        elif c.type == "MECHARROW":
            if not c.args or not c.args[0]:
                return False, "MECHARROW 为空"
        elif c.type == "PLUS":
            return False, "BLOCK 内不支持 [PLUS]（共振式只能单一）"
        else:
            return False, (f"BLOCK 内不支持 {c.type}（仅 STRUCT / "
                           f"ARROW:resonance / MECHARROW）")
    if not has_struct:
        return False, "BLOCK 内缺少 [STRUCT] 组件"
    return True, ""


def _check_reaction_sequence(children: list, comps: dict) -> str:
    """大一统架构守恒（reaction 布局）：按 [ARROW] 分步，每步按箭头类型
    与结构数分派——与 Drawbacks 第一节第 9 条方案一致：

    - type=single/reversible（反应）：
        单→单（主结构 + 附件合计两侧各 1）→ C 当量（原 ARROW 逻辑）；
        任一边 ≥2 → 完整原子+电荷守恒（原 REACTION 2a 逻辑）；
        sup 附件参与补足（+E 副反应物计左侧、-F 副产物计右侧）。
    - type=resonance（共振）：两侧原子守恒（同分子式，含 H，不比对电荷）。
    - type=retro（逆合成）：C 当量。
    - BLOCK 共振块作为单一结构（分子式取块内首个 STRUCT）。

    返回错误原因或 ""（无 ARROW 的纯排列不校验）。
    """
    def _block_smiles(block) -> str:
        for c in block.args[0] if block.args else []:
            if c.type == "STRUCT" and c.args and c.args[0]:
                return c.args[0].strip()
        return ""

    # 序列化：主结构（含 BLOCK）与箭头；arrow 附件 STRUCT 不参与主序列
    seq = []
    for child in children:
        if child.type == "STRUCT":
            if child.attrs.get("arrow"):
                continue
            smi = child.args[0].strip() if child.args and child.args[0] else ""
            seq.append(("struct", smi))
        elif child.type == "BLOCK":
            smi = _block_smiles(child)
            if not smi:
                return "BLOCK 内缺少 [STRUCT] 组件"
            seq.append(("struct", smi))
        elif child.type == "PLUS":
            seq.append(("plus",))
        elif child.type == "ARROW":
            a_type = (child.args[0] if child.args else "") or "single"
            sup = child.args[1] if len(child.args) > 1 else []
            cond = child.args[2] if len(child.args) > 2 else ""
            seq.append(("arrow", a_type, sup, cond))

    # 按箭头分步
    segments, arrows = [], []
    cur = []
    for item in seq:
        if item[0] == "arrow":
            arrows.append(item)
            segments.append(cur)
            cur = []
        else:
            cur.append(item)
    segments.append(cur)
    if not arrows:
        return ""  # 无箭头（纯排列/共振块独立展示）不校验
    for i, arrow in enumerate(arrows):
        left_items, right_items = segments[i], segments[i + 1]
        reason = _check_reaction_step(left_items, right_items, arrow, i, comps)
        if reason:
            return reason
    return ""


def _check_reaction_step(left_items, right_items, arrow, step, comps) -> str:
    """单步守恒分派（_check_reaction_sequence 的步内逻辑）。"""
    a_type, sup, _cond = arrow[1], arrow[2], arrow[3]

    def _species(items):
        out = []
        for it in items:
            if it[0] != "struct" or not it[1]:
                continue
            parsed = _parse_coeff(it[1])
            if parsed is None:
                return None
            coeff, bare = parsed
            out.append((coeff, bare))
        return out

    left = _species(left_items)
    right = _species(right_items)
    n_left_struct = sum(1 for it in left_items if it[0] == "struct" and it[1])
    n_right_struct = sum(1 for it in right_items if it[0] == "struct" and it[1])
    # sup 附件补足：+id 副反应物计左侧、-id 副产物计右侧
    for s in (sup or []):
        s = s.strip()
        if not s:
            continue
        sign, sid = (s[0], s[1:]) if s[0] in "+-" else ("+", s)
        info = comps.get(sid)
        if info is None:
            return f"第 {step + 1} 步：箭头附件引用未知组件「{sid}」"
        smi = (info.get("smiles") or "").strip()
        if not smi:
            return f"第 {step + 1} 步：附件组件「{sid}」SMILES 为空"
        parsed = _parse_coeff(smi)
        if parsed is None:
            return f"第 {step + 1} 步：附件「{sid}」SMILES 非法"
        coeff, bare = parsed
        (left if sign == "+" else right).append((coeff, bare))
        if sign == "+":
            n_left_struct += 1
        else:
            n_right_struct += 1

    def _fail(msg):
        return f"第 {step + 1} 步：{msg}"

    if a_type == "resonance":
        ls, rs = _sum_species(left), _sum_species(right)
        if ls is None or rs is None:
            return ""
        return _balance_reason(ls, rs, strict_h=True, step=f"第 {step + 1} 步",
                               check_charge=False)
    if a_type == "retro":
        # 逆合成（分子拆分）：宽松当量——前体 C 数不得超过目标（断键不增碳；
        # 丢失基团如 formylation 的 CO 允许）。明显反向（前体碳更多）拦截。
        lc, rc = _sum_c_counts(left), _sum_c_counts(right)
        if lc is not None and rc is not None and rc > lc:
            return _fail(f"逆合成前体 C 原子数（{rc}）多于目标（{lc}）")
        return ""
    # 反应（single/reversible）
    if n_left_struct == 1 and n_right_struct == 1:
        # 单→单：C 当量（原 ARROW 逻辑）
        lc, rc = _sum_c_counts(left), _sum_c_counts(right)
        if lc is not None and rc is not None and lc != rc:
            return _fail(f"两侧 C 原子数不等（{lc} vs {rc}，单→单仅校验当量）")
        return ""
    # 任一边 ≥2：完整原子+电荷守恒（原 REACTION 2a 逻辑）
    ls, rs = _sum_species(left), _sum_species(right)
    if ls is None or rs is None:
        return ""
    return _balance_reason(ls, rs, strict_h=True, step=f"第 {step + 1} 步",
                           check_charge=True)


# STRUCT 绘制模式（分子家族重构，20260818）：与 tag_parser._STRUCT_MODES 一致。
# skeleton=键线式/结构简式（默认）；lewis=电子式（+孤对）；stereo=楔形式；
# chair=椅式构象；newman=纽曼投影。
_STRUCT_MODES = ("skeleton", "lewis", "stereo", "chair", "newman")


def _check_newman_angle(angle: str) -> Tuple[bool, str]:
    """NEWMAN 二面角校验（0~360 数字）。"""
    if not angle:
        return False, "缺少角度"
    try:
        a = float(angle)
    except (TypeError, ValueError):
        return False, f"角度「{angle}」不是数字"
    if not 0 <= a <= 360:
        return False, f"角度 {a:g} 超出 0~360"
    return True, ""


def _check_chair_subs(smi: str, spec: str) -> Tuple[bool, str]:
    """CHAIR 取代基规格校验（mode=chair 专用）：SMILES 含环己烷六元碳环 +
    环位 1~6 + ax/eq 格式 + 该环位有非环取代基（真实 rdkit 才查环，
    fake mol 无 GetRingInfo 时降级跳过）。"""
    if _RDKIT_OK:
        mol = _parse_mol(smi)
        if mol is not None and hasattr(mol, "GetRingInfo"):
            from utils.rdkit_utils import cyclohexane_ring
            ring = cyclohexane_ring(mol)
            if ring is None:
                return False, f"SMILES 中未找到环己烷六元环「{smi}」"
            ring_set = set(ring)
            for tok in (spec or "").split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if tok.lower() == "flip":
                    continue    # 镜像画法令牌（翻转对比第二张）
                m = re.fullmatch(
                    r"(\d+):(ax|eq|axial|equatorial)", tok.lower())
                if not m:
                    return False, (f"CHAIR 取代位格式错误「{tok}」（应为 位:ax/eq，"
                                   f"环位 1~6 按环碳 SMILES 序号排序）")
                pos = int(m.group(1))
                if not 1 <= pos <= 6:
                    return False, f"CHAIR 环位 {pos} 超出范围 1~6"
                has_sub = any(
                    nbr.GetIdx() not in ring_set
                    for nbr in mol.GetAtomWithIdx(
                        ring[pos - 1]).GetNeighbors())
                if not has_sub:
                    return False, f"CHAIR 环位 {pos} 无取代基可标注"
    return True, ""


def _check_radical_charge_conflict(mol) -> str:
    """同一原子同时带形式电荷与自由基单电子 → 原因串（""=无冲突）。

    同一原子电荷+自由基在教学场景几乎必是书写错误（如图 32 把 FeBr4- 的
    负电荷画成 Fe⊖ 还带单电子点）；合法自由基离子的电荷与单电子在不同
    原子上（超氧根 [O-][O]），不受影响。孤立小离子的 RDKit 电子簿记
    （[O-]→1 单电子、[NH3+]→1 单电子）同属异常写法，一并拦截。
    """
    atoms_fn = getattr(mol, "GetAtoms", None)
    if atoms_fn is None:
        return ""   # fake mol（测试 fixture）：无原子遍历能力，跳过
    for atom in atoms_fn():
        fc = getattr(atom, "GetFormalCharge", lambda: 0)()
        nre = getattr(atom, "GetNumRadicalElectrons", lambda: 0)()
        if fc != 0 and nre > 0:
            return (f"原子 {atom.GetIdx()}（{atom.GetSymbol()}）同时带形式电荷 "
                    f"{fc:+d} 与 {nre} 个自由基单电子——同一原子不能既是离子又是"
                    f"自由基（电荷与单电子应分开写在不同原子上，如 [O-][O]）")
    return ""


def _validate_struct_args(args: list, attrs: dict = None) -> Tuple[bool, str]:
    """校验单个 STRUCT 参数（顶层或容器内）：SMILES 非空 + label 长度 + 模式参数。

    attrs 携带 mode/subs/bond/angle/charge。mode 分派各画法的专项校验；
    20260821 起 bond/charge 并入 STRUCT 参数（单分子标注，替代顶层
    BOND/CHARGE 新写法）：
      - bond=a-b：键突出标注（mode=newman 时仍是投影观察键，语义分派）；
      - charge=idx:+/-列表：部分电荷标注（0:+,3:-）。
    """
    attrs = attrs or {}
    if not args or not args[0] or not args[0].strip():
        return False, "SMILES 为空"
    ok, reason = _label_ok(args[1] if len(args) > 1 else None)
    if not ok:
        return False, reason
    smi = args[0].strip()
    if not _smiles_ok(smi):
        return False, f"无效 SMILES「{smi}」"
    if _RDKIT_OK:
        mol = _parse_mol(smi)
        if mol is not None:
            conflict = _check_radical_charge_conflict(mol)
            if conflict:
                return False, conflict
    mode = attrs.get("mode", "skeleton")
    if mode not in _STRUCT_MODES:
        return False, (f"未知 STRUCT 模式「{mode}」，支持 "
                       f"{'/'.join(_STRUCT_MODES)}")
    if mode == "chair":
        ok, reason = _check_chair_subs(smi, attrs.get("subs", ""))
        if not ok:
            return False, reason
    elif mode == "newman":
        ok, reason = _check_newman_bond(smi, attrs.get("bond", ""))
        if not ok:
            return False, reason
        ok, reason = _check_newman_angle(attrs.get("angle", ""))
        if not ok:
            return False, reason
    else:
        # 非 newman：bond= 键突出标注（a-b 必须真实成键）、charge= 部分电荷
        ok, reason = _check_struct_bond_mark(smi, attrs.get("bond", ""))
        if not ok:
            return False, reason
        ok, reason = _check_struct_charge(smi, attrs.get("charge", ""))
        if not ok:
            return False, reason
    return True, ""


# STRUCT bond= 键突出标注格式：a-b（原子序号对）
_STRUCT_BOND_MARK_RE = re.compile(r"^\d+-\d+$")


def _check_struct_bond_mark(smi: str, bond_spec: str) -> Tuple[bool, str]:
    """STRUCT bond= 键突出校验（非 newman）：a-b 格式 + 原子范围 + 真实成键。

    bond_spec 为空 → 直接通过。
    """
    if not bond_spec:
        return True, ""
    if not _STRUCT_BOND_MARK_RE.match(bond_spec):
        return False, f"bond 键标注格式错误「{bond_spec}」（应为 a-b，如 1-2）"
    if not _RDKIT_OK:
        return True, ""
    mol = _parse_mol(smi)
    if mol is None:
        return True, ""   # SMILES 已在上层校验，此处静默
    try:
        a, b = (int(x) for x in bond_spec.split("-"))
    except ValueError:
        return False, f"bond 键标注格式错误「{bond_spec}」"
    n = mol.GetNumAtoms()
    if not (0 <= a < n and 0 <= b < n):
        return False, f"bond 键 {a}-{b} 超出原子范围 0~{n - 1}"
    if mol.GetBondBetweenAtoms(a, b) is None:
        return False, f"bond 键 {a}-{b} 不存在（原子 {a} 与 {b} 之间没有化学键）"
    return True, ""


def _check_struct_charge(smi: str, charge_spec: str) -> Tuple[bool, str]:
    """STRUCT charge= 部分电荷校验：idx:+/- 列表格式 + 原子范围。

    charge_spec 为空 → 直接通过。
    """
    if not charge_spec:
        return True, ""
    if not _RDKIT_OK:
        return True, ""
    mol = _parse_mol(smi)
    if mol is None:
        return True, ""
    n = mol.GetNumAtoms()
    idxs = []
    for tok in (charge_spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"(\d+):[+-]", tok)
        if not m:
            return False, (f"charge 标注格式错误「{tok}」（应为 原子号:+/-，"
                           f"如 0:+,3:-）")
        idxs.append(int(m.group(1)))
    for i in idxs:
        if not 0 <= i < n:
            return False, f"charge 原子编号 {i} 超出原子范围 0~{n - 1}"
    return True, ""


# NEWMAN 键参数格式：a-b（原子序号对，投影观察键）
_NEWMAN_BOND_RE = re.compile(r"^\d+-\d+$")


def _check_newman_bond(smi: str, bond_spec: str) -> Tuple[bool, str]:
    """NEWMAN 键参数存在性校验（a-b 为 SMILES 中的一条键）。

    bond_spec 为空（旧格式，自动选键）→ 直接通过。
    """
    if not bond_spec:
        return True, ""
    try:
        a, b = (int(x) for x in bond_spec.split("-"))
    except ValueError:
        return False, f"键参数「{bond_spec}」格式应为 a-b"
    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception:
        return False, f"无效 SMILES「{smi}」"
    if mol is None:
        return False, f"无效 SMILES「{smi}」"
    if a < 0 or b < 0 or a >= mol.GetNumAtoms() or b >= mol.GetNumAtoms():
        return False, f"原子序号越界：{bond_spec}（原子数 {mol.GetNumAtoms()}）"
    if mol.GetBondBetweenAtoms(a, b) is None:
        return False, f"原子 {a} 与 {b} 之间无键"
    return True, ""


def _validate_newman(args: list) -> Tuple[bool, str]:
    """NEWMAN 校验：SMILES + 投影键（a-b，可缺省）+ 二面角（0~360）。

    兼容旧格式 [NEWMAN:SMILES,角度]（第二参数为角度、无键参数）——
    第二参数形如 `a-b` 时按新格式（第三参数为角度）解析。
    """
    if not args or not args[0]:
        return False, "SMILES 为空"
    smi = args[0].strip()
    if not _smiles_ok(smi):
        return False, f"无效 SMILES「{smi}」"
    arg1 = (args[1] or "").strip() if len(args) > 1 else ""
    arg2 = (args[2] or "").strip() if len(args) > 2 else ""
    if _NEWMAN_BOND_RE.match(arg1):
        # 新格式：[SMILES, a-b, 角度]
        bond_spec, angle = arg1, arg2
        if not angle:
            return False, "缺少角度"
        if _RDKIT_OK:
            ok, reason = _check_newman_bond(smi, bond_spec)
            if not ok:
                return False, reason
    else:
        # 旧格式：[SMILES, 角度]
        bond_spec, angle = "", arg1
    if not angle:
        return False, "缺少角度"
    try:
        a = float(angle)
    except (TypeError, ValueError):
        return False, f"角度「{angle}」不是数字"
    if not 0 <= a <= 360:
        return False, f"角度 {a:g} 超出 0~360"
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


def _validate_mech_arrow_pt(pt: str, n_atoms: int,
                            mol=None) -> str:
    """端点（原子序号 / a-b 键）合法性，返回原因串（""=合法）。

    20260821：显式 H 是真实原子参与编号（如 CC([H])CC 的 2 号是 H），
    原 a#k 语法废弃——引用 H 直接写原子序号即可，无需 XH 配对检查。
    mol：RDKit Mol（可选）——"a-b" 键端点用它验证两原子间确实存在化学键，
    防止引用不存在的键（如乙醛 CC=O 的 0-2 无键）；mol 缺失时只查序号范围。
    """
    if "-" in pt:
        a, b = pt.split("-")
        try:
            ia, ib = int(a), int(b)
        except ValueError:
            return f"键端点格式错误「{pt}」（应为 原子a-原子b）"
        if not (0 <= ia < n_atoms and 0 <= ib < n_atoms):
            return f"键端点「{pt}」越界（该分子只有 {n_atoms} 个原子，0 起）"
        gba = getattr(mol, "GetBondBetweenAtoms", None) if mol is not None else None
        if gba is not None:
            bond = gba(ia, ib)
            if bond is None:
                # 列出两端的实际连接原子，帮助模型重数索引（可操作化）
                gai = getattr(mol, "GetAtomWithIdx", None)
                nbrs_a = [n.GetIdx() for n in gai(ia).GetNeighbors()] \
                    if gai is not None else []
                nbrs_b = [n.GetIdx() for n in gai(ib).GetNeighbors()] \
                    if gai is not None else []
                hint = ""
                if nbrs_a:
                    hint += f"，原子 {ia} 实际连接 {nbrs_a}"
                if nbrs_b:
                    hint += f"，原子 {ib} 实际连接 {nbrs_b}"
                return (f"键端点「{pt}」引用原子 {ia} 与 {ib} 之间的键，"
                        f"但该分子中这两原子没有成键（先确认键的真实连接{hint}）")
        return ""
    try:
        ia = int(pt)
    except ValueError:
        return f"端点格式错误「{pt}」"
    if not 0 <= ia < n_atoms:
        return f"原子 {ia} 超出范围（该分子只有 {n_atoms} 个原子，0 起）"
    return ""


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
    # BLOCK 内 STRUCT 也注册进全局组件表（20260820：块内/跨块 MECHARROW
    # 统一引用；显式 id 全局查重，自动编号块内用独立前缀避免冲突）
    comps = {}
    for child in children:
        if child.type == "STRUCT":
            cid = child.attrs.get("id") or f"r{len(comps)}"
            if cid in comps and child.attrs.get("id"):
                return False, f"组件 id 重复「{cid}」"
            # 化学计量系数前缀（2CCO、1/2O2）：剥离后再校验 SMILES 与
            # 统计原子数；守恒比对（_check_reaction_step）自带系数解析
            raw_smi = child.args[0].strip() if child.args and child.args[0] else ""
            parsed_c = _parse_coeff(raw_smi)
            if parsed_c is None:
                return False, f"组件 {cid}: 系数格式错误「{raw_smi}」"
            bare_smi = parsed_c[1]
            # 双轨制：非 SMILES 但为纯化学式（KMnO4、H2SO4、CaCO3 等）
            # 放行——渲染端走文本节点轨道；化学式组件无原子可索引
            is_formula = bool(bare_smi) and not _smiles_ok(bare_smi) \
                and bool(_parse_plain_formula(bare_smi))
            comps[cid] = {
                "smiles": bare_smi,
                "at": child.attrs.get("at"),
                # 大一统架构：arrow 令牌组件 = 箭头上附件（副反应物/副产物），
                # 不参与主序列，由 ARROW 的 sup= 参数引用
                "arrow": bool(child.attrs.get("arrow")),
                "mode": child.attrs.get("mode", "skeleton"),
                "formula": is_formula,
            }
            # STRUCT 子标记本身递归校验（mode 分派）。容器内 mode 按布局
            # 放开（20260821 扩充）：reaction 禁 newman（投影
            # 是整图语义，与反应序列不兼容）；row/energy 不限制。
            # stereo/chair/newman 组件预渲染为不透明单元，仅展示。
            mode = child.attrs.get("mode", "skeleton")
            if mode == "newman" and layout_name == "reaction":
                return False, (f"组件 {cid}: reaction 布局不支持 mode=newman"
                               f"（纽曼投影请用顶层 [STRUCT:...] 或 row 布局）")
            if mode in ("stereo", "chair", "newman") and \
                    child.attrs.get("arrow"):
                return False, (f"组件 {cid}: 箭头附件（arrow 令牌）仅支持 "
                               f"mode=skeleton/lewis（立体画法组件不能挂箭头上）")
            if is_formula:
                # 化学式组件：无结构可画——mode 必须 skeleton，bond=/charge=
                # 等原子级标注不适用（渲染端无分子可挂）
                if mode != "skeleton":
                    return False, (f"组件 {cid}: 化学式组件（文本轨道）不支持 "
                                   f"mode={mode}")
                if child.attrs.get("bond") or child.attrs.get("charge"):
                    return False, (f"组件 {cid}: 化学式组件不支持 "
                                   f"bond=/charge= 标注")
                ok, reason = _label_ok(
                    child.args[1] if len(child.args) > 1 else None)
                if not ok:
                    return False, f"组件 {cid}: {reason}"
            else:
                ok, reason = _validate_struct_args(
                    [bare_smi] + list(child.args[1:]), child.attrs)
                if not ok:
                    return False, f"组件 {cid}: {reason}"
        elif child.type == "BLOCK":
            for bc in (child.args[0] if child.args else []):
                if bc.type != "STRUCT":
                    continue
                cid = bc.attrs.get("id") or f"b{len(comps)}r{len(comps)}"
                if cid in comps and bc.attrs.get("id"):
                    return False, f"组件 id 重复「{cid}」（含 BLOCK 内）"
                comps[cid] = {
                    "smiles": bc.args[0].strip() if bc.args and bc.args[0] else "",
                    "at": None,
                    "arrow": False,
                }

    # row 布局允许无 [STRUCT]（纯箭头/条件/连接符序列也合法）；reaction /
    # energy 仍要求至少一个组件（机理引用与驻点挂靠都依赖组件）；reaction 布局
    # 允许仅含 BLOCK 共振块（块内 STRUCT 由 _check_block 校验）
    has_block = any(c.type == "BLOCK" for c in children)
    if not comps and not has_block and layout_name != "row":
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
    # 顶层 + BLOCK 内 MECHARROW 统一校验（块内组件已注册进全局 comps，
    # 块内/跨块混合引用自动支持）
    mech_children = []
    for child in children:
        if child.type == "MECHARROW":
            mech_children.append(child)
        elif child.type == "BLOCK":
            mech_children.extend(
                bc for bc in (child.args[0] if child.args else [])
                if bc.type == "MECHARROW")
    # 子标记校验（顶层）：ARROW / BLOCK / CHARGE / HBOND / XH / BOND
    # （MECHARROW 单独处理：顶层 + BLOCK 内统一，见下方 mech_children 循环）
    sup_ref_counts = {}   # 附件组件 id → 被 ARROW sup 引用次数（唯一性校验）
    for child in children:
        ctype = child.type
        if ctype == "MECHARROW":
            continue
        elif ctype == "ARROW" and len(child.args) >= 1:
            # 大一统架构容器内箭头：[ARROW:type=..., sup=..., 条件]
            a_type = (child.args[0] or "").strip() or "single"
            if a_type not in _ARROW_TYPES:
                return False, (f"ARROW 类型「{a_type}」非法，支持 "
                               f"{'/'.join(_ARROW_TYPES)}")
            for s in (child.args[1] if len(child.args) > 1 else []):
                s = s.strip()
                if not s:
                    continue
                sid = s[1:] if s[0] in "+-" else s
                if sid not in comps:
                    return False, f"ARROW 附件引用未知组件「{sid}」"
                if not comps[sid].get("arrow"):
                    return False, (f"ARROW 附件组件「{sid}」未声明 arrow 令牌"
                                   f"（副反应物/副产物需 [STRUCT:...,id={sid},arrow]）")
                sup_ref_counts[sid] = sup_ref_counts.get(sid, 0) + 1
        elif ctype == "BLOCK":
            ok, reason = _check_block(child.args[0] if child.args else [])
            if not ok:
                return False, f"BLOCK: {reason}"
        elif ctype in ("CHARGE", "HBOND") and len(child.args) >= 2:
            ref = child.args[0].strip()
            if ref not in comps:
                return False, f"{ctype} 引用未知组件「{ref}」"
            if _opaque_comp(comps, ref):
                return False, (f"{ctype} 引用立体画法组件「{ref}」"
                               f"（stereo/chair/newman 仅展示，不支持标注）")
            if _formula_comp(comps, ref):
                return False, (f"{ctype} 引用化学式组件「{ref}」"
                               f"（文本轨道无原子可索引，不支持标注）")
            pairs = (child.args[1] or "").strip()
            if ctype == "CHARGE":
                idxs = [int(x) for x in re.findall(r"(\d+):", pairs)]
                if not idxs:
                    return False, f"CHARGE 标注格式错误「{pairs}」（应为 原子:δ± 列表）"
                if _RDKIT_OK:
                    n = atom_counts.get(ref, 0)
                    for i in idxs:
                        if not 0 <= i < n:
                            return False, f"CHARGE 原子编号 {i} 超出组件 {ref} 范围 0~{n - 1}"
            else:
                # HBOND（20260821 起语义）：HBOND:idA:a>idB:b——a 为给体组件
                # 中显式 H 原子的真实序号（SMILES 显式 H 参与编号，如
                # [H]OCCO[H] 的 0 号；a#k 语法废弃）；受体为 idB 组件的原子 b。
                found = 0
                for tok in pairs.split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    m = re.fullmatch(r"(\d+)>([A-Za-z0-9_]+):(\d+)", tok)
                    if not m:
                        return False, (f"HBOND 标注格式错误「{tok}」"
                                       f"（应为 给体H原子号>组件id:原子，如 0>w2:1）")
                    a, idb, b = int(m.group(1)), m.group(2), int(m.group(3))
                    if idb not in comps:
                        return False, f"HBOND 引用未知组件「{idb}」"
                    if _opaque_comp(comps, idb):
                        return False, (f"HBOND 引用立体画法组件「{idb}」"
                                       f"（stereo/chair/newman 仅展示，不支持标注）")
                    if _formula_comp(comps, idb):
                        return False, (f"HBOND 引用化学式组件「{idb}」"
                                       f"（文本轨道无原子可索引，不支持标注）")
                    found += 1
                    if _RDKIT_OK:
                        # 给体端点 a 必须落在给体组件范围内且为 H 原子
                        # （显式 H 是真实原子；原子序号需在 SMILES 中真实存在）
                        na = atom_counts.get(ref, 0)
                        if not 0 <= a < na:
                            return False, (f"HBOND 给体 H 原子编号 {a} 超出组件 "
                                           f"{ref} 范围 0~{na - 1}")
                        donor_atom = comp_mols.get(ref)
                        gai = getattr(donor_atom, "GetAtomWithIdx", None)
                        if gai is not None and \
                                gai(a).GetAtomicNum() != 1:
                            return False, (f"HBOND 给体端点「{ref}:{a}」不是 H 原子"
                                           f"（应为 SMILES 显式 H 的原子序号，"
                                           f"如 [H]OCCO[H] 的 0 号）")
                        # 受体原子范围
                        nb = atom_counts.get(idb, 0)
                        if not 0 <= b < nb:
                            return False, (f"HBOND 受体原子编号 {b} 超出组件 "
                                           f"{idb} 范围 0~{nb - 1}")
                if not found:
                    return False, f"HBOND 标注格式错误「{pairs}」（应为 给体H原子号>idB:b 列表）"
        elif ctype == "XH" and len(child.args) >= 2:
            ref = child.args[0].strip()
            if ref not in comps:
                return False, f"XH 引用未知组件「{ref}」"
            if _opaque_comp(comps, ref):
                return False, (f"XH 引用立体画法组件「{ref}」"
                               f"（stereo/chair/newman 仅展示，不支持标注）")
            if _formula_comp(comps, ref):
                return False, (f"XH 引用化学式组件「{ref}」"
                               f"（文本轨道无原子可索引，不支持标注）")
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
            if _opaque_comp(comps, ref):
                return False, (f"BOND 引用立体画法组件「{ref}」"
                               f"（stereo/chair/newman 仅展示，不支持标注）")
            if _formula_comp(comps, ref):
                return False, (f"BOND 引用化学式组件「{ref}」"
                               f"（文本轨道无原子可索引，不支持标注）")
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

    # MECHARROW 统一校验（顶层 + BLOCK 内；块内组件已注册进全局 comps，
    # 块内/跨块混合引用自动支持）
    for child in mech_children:
        if not child.args or not child.args[0]:
            return False, "MECHARROW 为空"
        for spec in child.args[0].split(","):
            m = _MECH_ARROW_RE.match(spec)
            if not m:
                return False, f"MECHARROW 格式错误「{spec}」"
            src_id, src_pt, _, dst_id, dst_pt, dst2_id, dst2_pt = m.groups()
            if src_id not in comps or dst_id not in comps:
                return False, f"MECHARROW 引用未知组件「{src_id}→{dst_id}」"
            if dst2_id is not None and dst2_id not in comps:
                return False, f"MECHARROW 引用未知组件「{dst2_id}」"
            if _opaque_comp(comps, src_id) or _opaque_comp(comps, dst_id) or \
                    (dst2_id is not None and _opaque_comp(comps, dst2_id)):
                return False, (f"MECHARROW 引用立体画法组件「{spec}」"
                               f"（stereo/chair/newman 仅展示，不支持端点引用）")
            if _formula_comp(comps, src_id) or _formula_comp(comps, dst_id) or \
                    (dst2_id is not None and _formula_comp(comps, dst2_id)):
                return False, (f"MECHARROW 引用化学式组件「{spec}」"
                               f"（文本轨道无原子可索引，不支持端点引用）")
            if dst2_id is not None:
                if "-" in dst_pt:
                    return False, f"MECHARROW 成键空白位端点格式错误「{spec}」"
            if _RDKIT_OK:
                src_reason = _validate_mech_arrow_pt(
                    src_pt, atom_counts.get(src_id, 0), comp_mols.get(src_id))
                if src_reason:
                    return False, f"MECHARROW 源端点「{src_id}:{src_pt}」{src_reason}"
                dst_reason = _validate_mech_arrow_pt(
                    dst_pt, atom_counts.get(dst_id, 0), comp_mols.get(dst_id))
                if dst_reason:
                    return False, f"MECHARROW 目标端点「{dst_id}:{dst_pt}」{dst_reason}"
                if dst2_id is not None:
                    dst2_reason = _validate_mech_arrow_pt(
                        dst2_pt, atom_counts.get(dst2_id, 0),
                        comp_mols.get(dst2_id))
                    if dst2_reason:
                        return False, (f"MECHARROW 目标端点「{dst2_id}:{dst2_pt}」"
                                       f"{dst2_reason}")

    # arrow 令牌组件唯一引用：必须被恰好一个 ARROW 的 sup 引用——
    # 不引用则组件不可见（渲染端只经 sup 通道绘制附件），多引用则
    # 同组件画多份、机理箭头引用的坐标写回歧义
    for cid, info in comps.items():
        if not info.get("arrow"):
            continue
        n = sup_ref_counts.get(cid, 0)
        if n == 0:
            return False, (f"箭头附件组件「{cid}」未被任何 ARROW 的 sup 引用"
                           f"（arrow 令牌组件必须被唯一一个 ARROW 引用）")
        if n > 1:
            return False, (f"箭头附件组件「{cid}」被 {n} 个 ARROW 引用"
                           f"（arrow 令牌组件必须被唯一一个 ARROW 引用）")

    if _RDKIT_OK:
        # 质子转移配对（4b）：双电子箭头引用显式 H 时必须画全配对
        reason = _check_proton_transfer_pairing(mech_children, comp_mols, comps)
        if reason:
            return False, reason
        for ref, idxs in xh_usage.items():
            reason = _check_xh_h_usage(comp_mols.get(ref), idxs, f"组件 {ref} ")
            if reason:
                return False, reason
        if layout_name == "reaction":
            # reaction 布局守恒：按 [ARROW] 分步 + 箭头类型 + sup 附件补足
            # （row/energy 跳过守恒）
            reason = _check_reaction_sequence(children, comps)
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
    if ttype == "HBOND":
        # 顶层 HBOND 已移除（2026-08-15 语义分离）：氢由分子渲染（SMILES
        # 显式 H）负责、HBOND 只画点状虚线，且仅支持容器内
        # HBOND:idA:给体H原子号>idB:原子
        return ValidationResult(
            tag, False,
            "HBOND 仅支持容器内使用（格式：给体组件:显式H原子号>受体组件:原子），"
            "氢原子请用 SMILES 显式 H 写出（如 [H]OCCO[H]）")
    if ttype == "ENERGY":
        ok, reason = _validate_energy(args)
        return ValidationResult(tag, ok, reason)
    if ttype == "STRUCT":
        # 分子家族统一入口：mode 分派各画法的专项校验
        ok, reason = _validate_struct_args(args, tag.attrs)
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


def _ring_double_pairs(smi: str) -> tuple:
    """苯环写法标识：环内双键的**相对位置**（与取代基无关）。

    圆圈式（含芳香小写）→ ("circle",)；凯库勒大写 → 环内双键在
    环中的相对索引（0~5，按环原子顺序），如 (1,3,5)。同一种凯库勒
    写法在不同取代基（苯 vs 硝基苯）下相对位置一致，可正确比对。
    """
    import re
    if not smi or not isinstance(smi, str):
        return ()
    if re.search(r"(?<![a-z])[cnops](?![a-z])", smi):
        return ("circle",)
    from rdkit import Chem
    try:
        m = Chem.MolFromSmiles(smi, sanitize=False)
        if m is None:
            return ()
        m.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL
                         ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                         ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        ri = m.GetRingInfo()
        if not ri.AtomRings():
            return ()
        ring = sorted(ri.AtomRings()[0])
        # 环边：ring[k]-ring[(k+1)%6]，索引 k
        idx = {a: k for k, a in enumerate(ring)}
        double_idx = []
        for b in m.GetBonds():
            if b.GetBondTypeAsDouble() >= 1.5:
                ia, ib = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                if ia in idx and ib in idx:
                    ka, kb = idx[ia], idx[ib]
                    # 环边索引 = min 序号（相邻原子）
                    if abs(ka - kb) == 1 or {ka, kb} == {0, len(ring) - 1}:
                        double_idx.append(min(ka, kb) if abs(ka-kb) == 1
                                          else max(ka, kb))
    except Exception:
        return ()
    return tuple(sorted(double_idx)) if len(double_idx) == 3 else ()


def _warn_benzene_consistency(tags: List[RenderTag],
                              results: List[ValidationResult]) -> None:
    """跨 COMPOSITE 软提示：机理中苯环写法不一致。

    同一机理的多步 COMPOSITE 中，所有含苯环的分子（苯/硝基苯/含苯中间体）
    应保持同一种苯环写法（圆圈式，或同一种凯库勒双键位置），否则
    MECHARROW 引用键时 σ/π 判断混淆。按"环内双键对"分组：
    圆圈式=("circle",)、凯库勒=双键对——组间不同即警告（挂到后出现的
    result.warnings，不拦截）。
    """
    from rdkit import Chem

    style_groups = {}  # style_key -> [result_idx, ...]
    for ri, tag in enumerate(tags):
        if tag.type != "COMPOSITE" or not tag.args or len(tag.args) < 2:
            continue
        for child in tag.args[1]:
            if child.type != "STRUCT" or not child.args:
                continue
            smi = child.args[0].strip()
            if not smi:
                continue
            # 化学计量系数前缀（2CCO 等）剥离后再探测；探测性解析统一
            # 静音——化学式组件（KMnO4 等）解析失败是常态，无诊断价值
            parsed_c = _parse_coeff(smi)
            smi = parsed_c[1] if parsed_c else smi
            try:
                with mute_rdkit_warnings(include_error=True):
                    m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
            except Exception:
                continue
            pairs = _ring_double_pairs(smi)
            if not pairs:
                continue  # 非苯环分子（σ 络合物、NO₂⁺ 等）
            style_groups.setdefault(pairs, []).append(ri)

    if len(style_groups) <= 1:
        return
    # 多种苯环写法并存：给最后出现的 result 挂警告
    last_ri = max(max(idxs) for idxs in style_groups.values())
    if 0 <= last_ri < len(results) and results[last_ri].ok:
        desc = [("圆圈式" if k == ("circle",) else f"凯库勒{k}") for k in style_groups]
        results[last_ri].warnings.append(
            f"苯环写法不一致：同一机理中用了 {desc} "
            f"（应统一为一种写法，避免 MECHARROW 引用键时 σ/π 混淆）")


# 最近一次 validate_tags 的软提示（跨 COMPOSITE 苯环写法一致性等）。
# 校验通过但附提醒的警告——调用方（app.py 等）可读取并展示给用户。
_LAST_WARNINGS: list = []


def get_last_warnings() -> list:
    """最近一次 validate_tags 产生的软提示列表（字符串）。"""
    return list(_LAST_WARNINGS)


def validate_tags(tags: List[RenderTag]) -> Tuple[List[RenderTag], List[ValidationResult]]:
    """校验标记列表。

    返回:
        (valid_tags, invalid_results)：
        valid_tags — 通过校验的标记（顺序保持）；
        invalid_results — 校验失败的 ValidationResult 列表（含失败原因）。
    软提示（不拦截，如苯环写法不一致）通过 get_last_warnings() 获取。
    """
    global _LAST_WARNINGS
    valid, invalid = [], []
    results = []
    for tag in tags:
        result = validate_tag(tag)
        results.append(result)
        if result.ok:
            valid.append(tag)
        else:
            invalid.append(result)

    # 软提示：跨 COMPOSITE 苯环写法一致性（挂到后出现的 result 上）
    _LAST_WARNINGS = []
    if _RDKIT_OK:
        _warn_benzene_consistency(tags, results)
        for r in results:
            _LAST_WARNINGS.extend(r.warnings)
    return valid, invalid


def _tag_name_for(tag: RenderTag) -> str:
    """降级提示的标记中文名：分子旧标记归一化后 type=STRUCT，
    优先用 attrs.orig_type（LEWIS→"Lewis 结构式"等）保留友好名。"""
    return _TAG_NAMES.get(tag.attrs.get("orig_type") or tag.type, tag.type)


def degrade_text(tag: RenderTag, reason: str) -> str:
    """校验失败标记的降级提示文本。

    用户可见版本：去掉内部校验类别前缀（化学校验：）与括号内详情/修正指导，
    只留主因（如"方程式两侧原子不守恒"）——完整原因（含元素计数明细）仍
    通过 P2 修正 prompt 与 metrics 详情供内部使用。
    """
    msg = reason
    if msg.startswith(_CHEM_PREFIX):
        msg = msg[len(_CHEM_PREFIX):].split("（", 1)[0].strip()
    return f"（{_tag_name_for(tag)}图示无法渲染：{msg}，已省略）"


def degrade_text_friendly(tag: RenderTag) -> str:
    """用户可见降级提示（友好版）：不含校验技术细节（原子守恒/元素差/索引
    等），只告知该处图示未生成——普通用户不关心内部校验原因。详细原因仍由
    reason（P2 修正 prompt / diagnostics / metrics）承载。"""
    return f"（{_tag_name_for(tag)}图示无法渲染，已省略）"


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
