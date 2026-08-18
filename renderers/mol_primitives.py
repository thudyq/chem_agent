# -*- coding: utf-8 -*-
"""renderers/mol_primitives.py — RDKit 分子骨架绘制的共享工具。

把 mechanism / lewis / stereo / charge / hbond 中重复的原子标签、
2D 坐标计算、键线绘制逻辑抽取到这里，避免复制粘贴。
"""

import math
import re

try:
    from .collide import CHARGE_CIRCLE_R, DOT_R
except ImportError:  # 直接脚本运行（无包上下文）
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from renderers.collide import CHARGE_CIRCLE_R, DOT_R


# 氢化物惯例：H 写在元素前的非金属（电负性 > 2.0，如 HF/HCl/HBr/HI/H2O/H2S）。
# 其余氢化物元素写在 H 前（NH3/PH3/BH3/CH4/SiH4）——N/P/B/Si/C 排除。
_H_PREFIX_ELEMENTS = {9, 17, 35, 53, 8, 16, 34, 52}  # F Cl Br I, O S Se Te


def _only_h_neighbors(atom) -> bool:
    """原子是否只连氢（无重原子邻居）——如孤立水分子 O。

    用于标签方向特例：水的 O 只连 2 个 H，化学上写 H₂O（H 在前，
    氢化物惯例），而 OH₂ 是羟基 -OH 的写法（仅 O 连 C 时正确）。
    """
    return all(nbr.GetAtomicNum() == 1 for nbr in atom.GetNeighbors())


def _covalent_bond_len(atom) -> float:
    """孤立原子 H 键长基准：2×共价半径（RDKit PeriodicTable）。

    与 RDKit 2D 坐标生成器键长一致（未缩放 C-C=1.5≈2×Rc(C)=1.52）。
    孤立原子（无键）的显式 H 键用此基准，使与骨架键等长；
    调用方按自身 scale 缩放——composite 传 h_len×_MOL_SCALE。
    """
    try:
        from rdkit import Chem
        pt = Chem.GetPeriodicTable()
        return 2.0 * pt.GetRcovalent(atom.GetAtomicNum())
    except Exception:
        return 1.0


def atom_label(atom, explicit_hs: int = 0, flip: bool = False) -> str | None:
    """生成非隐式碳原子的标签（如 OH、NH₂、Cl）。

    纯碳原子（原子序 6）返回 None，表示不显示标签（键线式）：
    - 环内碳一律不标（含带电荷的，如 σ 络合物的 [CH+]——正电荷用
      圆圈电荷显示，不标 CH，与 atom_main_label 一致）；
    - 有重原子邻居的链上碳不标（CHn 由骨架线隐含）；**带电碳同样不标**
      （形式电荷由圆圈电荷 ⊕ 标注，键的根数已反映其连接情况）。
    孤立碳（仅连 H，如 CH4 的 C）无键线式可言——必须显示 C/CH4
    （带电孤立碳如 [CH3+] 仍标 CH₃，电荷由圆圈电荷显示）。
    explicit_hs：已显式画出的 H 数（[XH]/氢键给体），从标签 H 计数中
    扣除，保证"标签 H + 画出 H"总数正确（如 OH 画出 H 后标签为 O）。
    水分子特例：O 只连 H 时写 H₂O（H 在前）。
    flip=True：键端在标签右侧时元素符号右移（OH→HO、NH₂→H₂N，
    使键连接的原子靠近键端，Drawbacks 第 6 条）。
    电荷一律不在标签内嵌上标——统一由 charge_tikz 的圆圈电荷显示
    （Instruction-for-Electrons §三；此前与圆圈并存导致双重显示）。
    """
    z = atom.GetAtomicNum()
    # 通用基团占位符（R/X/Ph/Ac 等，expand_group_abbrevs 的 dummy 原子）：
    # 直接显示缩写文本（普通文本节点），不走元素标签逻辑
    if atom.HasProp("_abbr"):
        return atom.GetProp("_abbr")
    if z == 6 and atom.IsInRing():
        # 键线式：环内碳一律不标（含带电荷的，如 σ 络合物的 [CH+]——
        # 正电荷用圆圈电荷显示，不标 CH，与 atom_main_label 一致）
        return None
    if z == 6 and not _only_h_neighbors(atom):
        # 键线式：有重原子邻居的链上碳不标（CHn 由骨架线隐含）；
        # 带电碳同样不标——正电荷由圆圈电荷 ⊕ 标注
        return None
    # 孤立碳（仅连 H，如 CH4 的 C）无键线式可言——必须显示 C/CH4
    # （B1 20260812：自由基夺氢机理的单碳组分，否则图上是空位）

    sym = atom.GetSymbol()
    sym = sym[0].upper() + sym[1:]
    h = max(0, atom.GetTotalNumHs() - explicit_hs)
    if (h >= 1 and atom.GetAtomicNum() in _H_PREFIX_ELEMENTS
            and atom.GetFormalCharge() >= 0 and _only_h_neighbors(atom)):
        # 氢化物惯例 H 前置（HF/HCl/HBr/HI/H2O/H2S）；正离子氢化物同惯例
        # （H3O⁺ 而非 OH3⁺，20260815）；负离子保持 XH（如 OH⁻ 写 OH，
        # 不写 HO）；碳始终 CHn（C 在前，·CH3 也写 CH3）；
        # N/P/B/Si 的氢化物写 NH3/PH3/BH3/SiH4（元素在前）——问题 2（NH3→H3N）
        parts = (f"H$_{{{h}}}$" if h > 1 else "H") + sym
    elif flip:
        parts = (f"H$_{{{h}}}$" if h > 1 else ("H" if h == 1 else "")) + sym
    else:
        parts = sym
        if h == 1:
            parts += "H"
        elif h > 1:
            parts += f"H$_{{{h}}}$"

    return parts


def heavy_atom_count(mol) -> int:
    """分子中非氢原子数（重原子数）。"""
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)


def mol_default_labeler(mol):
    """按分子大小选择默认原子标签函数（统一规则）：

    - 重原子数 ≤ 2 的小分子（CH₃Cl、CH₂=CH₂、CH₄、H₂O…）→ 结构简式
      （`atom_main_label`：非环碳写出 CHn，教科书式 H₃C—Cl）；
    - 其余 → 键线式（`atom_label`：碳原子不标 CHn，骨架线隐含）。

    绘制（molecule_scope_lines）、视觉包围盒（mol_visual_bbox）、
    符号中心（_dot_center）与机理键中点（mech_arrow_origin）共用，
    保证布局感知与画面一致。
    """
    if heavy_atom_count(mol) <= 2:
        return atom_main_label
    return atom_label


def condensed_atom_label(atom) -> str | None:
    """结构简式标签：非环碳原子同样写出（CH₃/CH₂/CH/C），环上碳原子
    保持键线式（返回 None）。

    **已弃用**：默认选择已由 `mol_default_labeler` 的重原子数规则取代
    （结构简式只用于 ≤2 重原子小分子，此时不可能出现环内碳），保留仅
    为兼容外部引用；与 atom_main_label 的带电环内碳差异在新规则下不触发。
    """
    if atom.GetAtomicNum() == 6 and atom.GetFormalCharge() == 0 and atom.IsInRing():
        return None
    return atom_label(atom) if atom.GetAtomicNum() != 6 else _carbon_label(atom)


def _carbon_label(atom) -> str:
    parts = "C"
    h = atom.GetTotalNumHs()
    if h == 1:
        parts += "H"
    elif h > 1:
        parts += f"H$_{{{h}}}$"
    return parts


def label_plain_len(label: str) -> int:
    """标签去掉 LaTeX 排版符号后的可视字符数，用于估算键线留白宽度。"""
    return len(re.sub(r"[$_{}^\\]", "", label))


# 纯化学式 label 识别用元素表（有机/常见无机，与 core.tag_validator 的
# _REAL_ELEMENTS 一致）。刻意不含 Ar/Ac（prompt 允许的通用基团缩写）。
_FORMULA_LABEL_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I",
    "Li", "Na", "K", "Mg", "Ca", "Al", "Fe", "Cu", "Zn", "Ag", "Au",
    "Hg", "Pb", "Sn", "Se", "Te",
    "Be", "Sc", "Ti", "V", "Cr", "Mn", "Co", "Ni", "Ga", "Ge", "As",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Cd", "In",
    "Sb", "Cs", "Ba", "La", "Ce", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Tl", "Bi",
}
_FORMULA_LABEL_RE = re.compile(r"^([A-Z][a-z]?\d*)+([+-]\d*)?$")


def is_formula_label(label: str) -> bool:
    """label 是否为纯化学式（含自由基符号 · 的也算，如 Cl·、·CH3）。

    用于决定 label 是否在分子下方重复显示：纯化学式（CH3Cl、OH-、Cl·）
    分子本身已展示，不重复；中文/角色标注（底物、质子化乙醇）需显示。
    去 · 后需全由可识别元素 + 数字组成（首字符大写）。
    """
    s = (label or "").strip().replace("·", "")
    if not s or not _FORMULA_LABEL_RE.match(s):
        return False
    return all(sym in _FORMULA_LABEL_ELEMENTS
               for sym in re.findall(r"([A-Z][a-z]?)", s))


def _is_wide_char(ch: str) -> bool:
    """是否全角/宽字符（CJK 汉字与全角标点，宽度约为西文的 2 倍）。"""
    o = ord(ch)
    return (0x2E80 <= o <= 0x9FFF       # CJK 部首 ~ 统一表意文字
            or 0xF900 <= o <= 0xFAFF    # 兼容表意文字
            or 0xFE30 <= o <= 0xFE4F    # CJK 兼容形式
            or 0xFF00 <= o <= 0xFFEF)   # 全角形式（含全角标点）


def label_visual_width(label: str) -> float:
    """标签可视宽度（区分全角/半角），用于包围盒与间距估算。

    西文/数字半宽 0.13（全宽 0.26，与旧 "0.13 × 字符数" 估算一致），
    全角字符（中文等）半宽 0.26（全宽 0.52）——修正旧估算对中文标签
    过窄导致的相邻组件重叠（如 6 字中文旧估算半宽仅 0.78，真实约 1.56）。
    """
    plain = re.sub(r"[$_{}^\\]", "", label)
    if not plain:
        return 0.0
    w = 0.0
    for ch in plain:
        w += 0.52 if _is_wide_char(ch) else 0.26
    return w


def label_bond_margin(label: str) -> float:
    """按标签可视长度分档的键线留白（机理场景统一值）。"""
    n = label_plain_len(label)
    if n <= 2:
        return 0.30
    if n == 3:
        return 0.45
    return 0.58


def format_partial_charge(raw: str) -> str:
    """部分电荷文本排版：δ+ → $\\delta^+$，δ- → $\\delta^-$；其他原样返回。"""
    raw = raw.strip()
    if "δ" in raw:
        s = raw.replace("δ", "\\delta")
        if s.endswith("+"):
            s = s[:-1] + "^+"
        elif s.endswith("-"):
            s = s[:-1] + "^-"
        return f"${s}$"
    return raw


def parse_charge_pairs(charges_str: str) -> dict:
    """'0:δ+,1:δ-' → {0: 'δ+', 1: 'δ-'}（原子序号 → 部分电荷标签）。"""
    result = {}
    for pair in (charges_str or "").split(","):
        pair = pair.strip()
        if ":" in pair:
            idx_str, _, label = pair.partition(":")
            try:
                result[int(idx_str.strip())] = label.strip()
            except ValueError:
                pass
    return result


def parse_hbond_pairs(pairs_str: str) -> list:
    """'0-2,3-5' → [(0, 2), (3, 5)]（氢键两端原子序号）。"""
    pairs = []
    for s in (pairs_str or "").split(","):
        s = s.strip()
        if "-" in s:
            try:
                f, t = s.split("-", 1)
                pairs.append((int(f.strip()), int(t.strip())))
            except ValueError:
                pass
    return pairs


def _bond_path(mol, x_idx: int, y_idx: int) -> list | None:
    """两原子间的最短键路径（BFS），返回原子序号列表；不连通返回 None。"""
    from collections import deque
    prev = {x_idx: None}
    dq = deque([x_idx])
    while dq:
        a = dq.popleft()
        if a == y_idx:
            break
        for n in mol.GetAtomWithIdx(a).GetNeighbors():
            ni = n.GetIdx()
            if ni not in prev:
                prev[ni] = a
                dq.append(ni)
    if y_idx not in prev:
        return None
    path = [y_idx]
    while path[-1] != x_idx:
        path.append(prev[path[-1]])
    return path[::-1]


def adjust_hbond_conformation(mol, x_idx: int, y_idx: int) -> None:
    """氢键场景的构象调整（规范：把给体与受体画到主链同一侧）。

    找给体 X → 受体 Y 的最短键路径，若 Y 与 X 在路径中间键异侧，
    把受体侧片段绕中间键反射（2D 镜像，保距）到 X 同侧，
    便于在骨架外侧画出不穿越结构的氢键。原地修改 2D 坐标。
    """
    path = _bond_path(mol, x_idx, y_idx)
    if not path or len(path) < 3:
        return
    mid = len(path) // 2
    a1, a2 = path[mid - 1], path[mid]
    # 受体侧片段：从 a2 出发不经过 a1 的可达原子
    frag, stack = set(), [a2]
    while stack:
        a = stack.pop()
        if a in frag:
            continue
        frag.add(a)
        for n in mol.GetAtomWithIdx(a).GetNeighbors():
            if n.GetIdx() != a1:
                stack.append(n.GetIdx())
    ax, ay = atom_pos(mol, a1)
    bx, by = atom_pos(mol, a2)
    dx, dy = bx - ax, by - ay

    def _side(i):
        px, py = atom_pos(mol, i)
        return dx * (py - ay) - dy * (px - ax)

    if _side(x_idx) * _side(y_idx) >= 0:
        return  # 已同侧
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    conf = mol.GetConformer()
    for i in frag:
        px, py = atom_pos(mol, i)
        rx, ry = px - ax, py - ay
        proj = rx * ux + ry * uy           # 轴向投影（保持）
        perp = -rx * uy + ry * ux          # 法向投影（镜像取反）
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (ax + proj * ux + perp * uy,
                                 ay + proj * uy - perp * ux, p.z))


def _zigzag_continuation_angle(bond_ang: float) -> float | None:
    """端原子显式 H 的锯齿延续方向（链状骨架 zigzag 惯例）。

    bond_ang 为原子→邻居方向（_bond_angles 输出）。在入键方向的 ±60°
    候选中取垂直分量与入键交替者（入键下行则 H 上行，反之亦然）；
    入键近水平无法判断交替时取向上者；近竖直（两候选同侧）返回 None，
    由调用方回退到最大空档逻辑。
    """
    s = math.sin(math.radians(bond_ang))
    base = bond_ang + 180.0
    opts = [(base + 60.0) % 360.0, (base - 60.0) % 360.0]
    if abs(s) < 0.3:
        up = [o for o in opts if math.sin(math.radians(o)) > 0]
        return up[0] if up else None
    want = 1.0 if s > 0 else -1.0
    for o in opts:
        if math.sin(math.radians(o)) * want > 0:
            return o
    return None


def place_explicit_hs(mol, idx: int, count: int = 1,
                      toward: tuple[float, float] | None = None,
                      h_len: float | None = None) -> list:
    """原子 idx 上 count 个显式 H 的位置（互不重叠的空档方向扇形分配）。

    toward 非空时首个 H 优先沿该方向（氢键 X—H···Y 直线）；其余按
    最大空档角平分线依次分配，同一空档内多根 H 扇形展开 ±20°。
    端原子（仅 1 根键）且 toward 为空时，取锯齿延续方向（
    _zigzag_continuation_angle，与入键垂直分量交替、指向上方）。
    h_len 缺省取该原子的平均键长（与普通骨架键等长）。
    """
    x, y = atom_pos(mol, idx)
    if h_len is None:
        lens = []
        for b in mol.GetAtomWithIdx(idx).GetBonds():
            nx, ny = atom_pos(mol, b.GetOtherAtomIdx(idx))
            lens.append(math.hypot(nx - x, ny - y))
        if not lens:
            # 孤立原子（无键，如 CH4 的 C）：先取分子内其他键的平均长度；
            # 分子也无键时用共价半径和 Rc(X)+Rc(H)（未缩放基准，问题 2）。
            all_lens = []
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                xi, yi = atom_pos(mol, i)
                xj, yj = atom_pos(mol, j)
                all_lens.append(math.hypot(xj - xi, yj - yi))
            lens = all_lens or [_covalent_bond_len(mol.GetAtomWithIdx(idx))]
        h_len = sum(lens) / len(lens)
    blocked = sorted(a % 360.0 for a in _bond_angles(mol, idx))
    gaps = []
    if not blocked:
        gaps.append((360.0, 0.0))
    for i, a1 in enumerate(blocked):
        a2 = blocked[(i + 1) % len(blocked)] if i + 1 < len(blocked) \
            else blocked[0] + 360.0
        gaps.append((a2 - a1, (a1 + (a2 - a1) / 2.0) % 360.0))
    gaps.sort(reverse=True)

    angles = []
    if toward is not None:
        ang = math.degrees(math.atan2(toward[1] - y, toward[0] - x)) % 360.0
        if all(_ang_diff(ang, b) > 30.0 for b in blocked):
            angles.append(ang)
    elif len(blocked) == 1 and count == 1:
        # 端原子单 H：锯齿延续方向（与入键垂直分量交替，指向上方）
        ang = _zigzag_continuation_angle(blocked[0])
        if ang is not None:
            angles.append(ang)
    elif len(blocked) == 2 and count == 2:
        # 双键位点双 H（如 CH2 示出两个 H）：以两键的镜像轴（大空档角平分线）
        # 为中心 ±30° 对称扇出；镜像轴贴近 30° 整数倍时吸附——修正 2D 布局
        # 微旋转导致的视觉歪斜（如 83°→90°，左右对称、同高、60° 夹角）
        axis = gaps[0][1]
        snapped = round(axis / 30.0) * 30.0 % 360.0
        if _ang_diff(snapped, axis) <= 10.0:
            axis = snapped
        angles.extend([(axis + 30.0) % 360.0, (axis - 30.0) % 360.0])
    for gi, (gap, mid) in enumerate(gaps):
        if len(angles) >= count:
            break
        if any(_ang_diff(mid, a) < 25.0 for a in angles):
            continue
        slots = min(count - len(angles),
                    max(1, int(gap // 40) if gap < 360 else count))
        for s in range(slots):
            off = (s - (slots - 1) / 2.0) * 20.0
            angles.append((mid + off) % 360.0)

    positions = [(x + h_len * math.cos(math.radians(a)),
                  y + h_len * math.sin(math.radians(a)))
                 for a in angles[:count]]
    if len(positions) < count:
        # 空档不足截断（E6）：相邻空档过近被跳过时少画，告警而非静默
        print(f"[mol_primitives] 显式 H 空档不足：原子 {idx} 请求 {count} 个，"
              f"实际画出 {len(positions)} 个")
    return positions


def place_donor_h(mol, x_idx: int, y_pos: tuple[float, float],
                  h_len: float | None = None) -> tuple[float, float]:
    """给体 X 的显式 H 位置（规范：X—H 用实线画出）。

    优先取 X→Y 方向（X—H···Y 尽量呈直线）；该方向与已有键过近（<30°）
    时改取最大空档的角平分线，避免与骨架重叠。
    h_len 缺省取该原子的平均键长（与普通骨架键等长）。
    """
    return place_explicit_hs(mol, x_idx, 1, toward=y_pos, h_len=h_len)[0]


def place_h_avoiding(mol, idx: int, pos: tuple[float, float],
                     occupancy, radius: float = 0.15) -> tuple[float, float]:
    """显式 H 节点的避障放置（R-8）：规则位置为首选，冲突时绕原子旋转
    ±15°/±30°/±45° 取第一个零冲突候选。pos 为规则给出的局部坐标；
    返回避障后的局部坐标。"""
    ax, ay = atom_pos(mol, idx)
    base = math.atan2(pos[1] - ay, pos[0] - ax)
    dist = math.hypot(pos[0] - ax, pos[1] - ay)
    cands = []
    for off in (0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0):
        r = base + math.radians(off)
        cands.append(("circle", ax + dist * math.cos(r),
                      ay + dist * math.sin(r), radius))
    chosen, _ = occupancy.place(
        cands, warn=f"显式 H（原子 {idx}）候选位全部冲突")
    return chosen[1], chosen[2]


def label_edge_point(mol, idx: int, toward: tuple[float, float], *,
                     labeler=atom_label, margin_fn=label_bond_margin
                     ) -> tuple[float, float]:
    """原子 idx 指向 toward 方向的标签边缘点（新增化学键的起笔点）。

    有可见标签的原子（如 O、OH）从标签边缘起笔，避免新画出的键压住标签；
    键线式碳原子（atom_label 返回 None）无标签，从原子中心起笔。
    供 [XH] 显式氢、氢键 X—H 实线等新增键使用，与骨架键同一留白逻辑。
    """
    x, y = atom_pos(mol, idx)
    tx, ty = toward
    dx, dy = tx - x, ty - y
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    lab = labeler(mol.GetAtomWithIdx(idx))
    m = margin_fn(lab) if lab else 0.0
    return x + ux * m, y + uy * m


def h_label_edge_point(hx: float, hy: float, toward: tuple[float, float],
                       margin_fn=label_bond_margin) -> tuple[float, float]:
    """显式 H 节点指向原子方向的标签边缘点（X—H 键 H 端留白）。

    与原子端 label_edge_point 同一留白口径：H 节点是几何放置的伪标签
    （单字符 "H"，label_bond_margin("H")=0.30），X—H 键线终点停在 H
    标签占位之外，不再画到 H 中心靠 fill=white 遮盖。假骨架水/氨的
    X—H 已用此口径（composite._pseudo_hbond_lines 的 gap_h），普通
    [XH]（顶层与容器内）补齐——三处绘制统一。
    toward 为原子 idx 的坐标（X 端）；返回从 H 向 X 收缩 margin 的点。
    """
    tx, ty = toward
    dx, dy = hx - tx, hy - ty
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    m = margin_fn("H")
    return hx - ux * m, hy - uy * m


def hbond_dots_tikz(fx: float, fy: float, tx: float, ty: float, *,
                    spacing: float = 0.3, radius: float = 0.028,
                    inset_start: float = 0.18, inset_end: float = 0.25,
                    max_dots: int = 10) -> list:
    r"""氢键 H···Y 点状虚线（teal 圆点，3~10 点）。

    从 (fx,fy)（给体 H）到 (tx,ty)（受体 Y）均匀布点；起点内缩 inset_start
    避免首点落在给体 H 标签中心，末端内缩 inset_end 避免压住受体标签。
    全图点径 radius 与间距 spacing 固定一致，距离远时自动增加点数
    （封顶 max_dots，规范第 4 条）。两端内缩各不超过总长 1/4，
    保证 H 与 Y 过近时仍有可布点区间。
    """
    dx, dy = tx - fx, ty - fy
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    inset_start = min(inset_start, length * 0.25)
    inset_end = min(inset_end, length * 0.25)
    sx, sy = fx + ux * inset_start, fy + uy * inset_start
    ex, ey = tx - ux * inset_end, ty - uy * inset_end
    seg = math.hypot(ex - sx, ey - sy)
    n = max(3, min(max_dots, round(seg / spacing) + 1))
    lines = []
    for k in range(n):
        t = k / (n - 1)
        lines.append(
            f"\\fill[teal] ({sx + (ex - sx) * t:.2f},{sy + (ey - sy) * t:.2f}) "
            f"circle ({radius});"
        )
    return lines


_VALENCE_ELECTRONS = {1: 1, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7,
                      14: 4, 15: 5, 16: 6, 17: 7, 35: 7, 53: 7}

_LP_DIST = 0.24           # 孤对电子点到原子的固定距离（原 0.30，调至 0.24 更紧凑）
_BOND_GAP = 0.08          # 双键/三键平行线间距（与 bond_segments 一致）
_MECH_LABEL_GAP = 0.05    # 箭头始末端点距"标签所占位置"边缘的间距（0~0.10 浮动基准）
_LABEL_SQUARE_HALF = 0.13 # 原子标签"所占位置"按边长 0.26 正方形（中心=符号中心，半=单字符半宽）
_LP_DOT_RADIUS = 0.028    # 孤对电子/单电子点半径（lone_pair_tikz 同值）
_ARROW_POINT_GAP = 0.05   # 机理箭头端点与"点/线"（孤对电子、单电子、键线段）的空隙
_ORTHO = [90.0, 180.0, 270.0, 0.0]      # 正交槽位（优先）
_DIAG = [45.0, 135.0, 225.0, 315.0]     # 斜向槽位（正交占满时兜底）
_CHAR_HALF_W = 0.13       # 标签单字符半宽估计（用于元素符号中心修正）
_SUBSCRIPT_CHAR_W = 0.67  # 下标/上标内字符的宽度权重（相对普通字符；实测 OH₂ 反推 ≈0.67）
_SUBSCRIPT_DEPTH_COMP = 0.025  # 含下标标签文字主体相对 node 中心的上移量（实测 ≈0.025）


def _implicit_shown_hs(atom) -> int:
    """以标签后缀形式显示的 H 数（= 总 H 数 − 已显式画出的 H 邻居数）。

    带电原子（如 [OH-]）的 GetNumImplicitHs() 可能返回 0，但标签仍显示 H，
    因此用 totalHs − 显式 H 邻居数作为阻挡/计数依据。
    """
    explicit_h = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
    return max(0, atom.GetTotalNumHs() - explicit_h)


def lone_pair_count(atom) -> tuple[int, int]:
    """返回 (孤对电子对数, 单电子数)。

    非键电子数 = 价电子 - 键级和 - 形式电荷 - 自由基电子数 - 标签氢数；
    不在表中的元素（金属等）返回 (0, 0)。
    """
    ve = _VALENCE_ELECTRONS.get(atom.GetAtomicNum())
    if ve is None:
        return 0, 0
    bonds = sum(int(round(b.GetBondTypeAsDouble())) for b in atom.GetBonds())
    radicals = atom.GetNumRadicalElectrons()
    nonbonding = max(0, ve - bonds - atom.GetFormalCharge() - radicals
                     - _implicit_shown_hs(atom))
    return nonbonding // 2, radicals


def _bond_angles(mol, idx: int) -> list:
    x0, y0 = atom_pos(mol, idx)
    angles = []
    for b in mol.GetAtomWithIdx(idx).GetBonds():
        x1, y1 = atom_pos(mol, b.GetOtherAtomIdx(idx))
        angles.append(math.degrees(math.atan2(y1 - y0, x1 - x0)))
    return angles


def _ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _label_h_is_left(mol, idx: int) -> bool:
    """渲染标签中 H 是否在元素符号左侧（H 前缀，如 H₂O/HF/HCl）。

    同时覆盖两类来源：化学惯例（孤立氢化物 H 前置）与标签翻转
    （flip：键端在右侧时 OH₂ → H₂O）。按**实际渲染标签文本**判断——
    标签以 H 开头且非元素符号开头即为 H 前缀（阻挡左侧 180°）；
    H 后缀（OH₂/CH₃）阻挡右侧（0°）。此前仅按化学惯例判断，
    带电 flip 场景（如 [OH2+] 标签 H₂O）误判 H 在右 → 孤对/电荷
    压住左侧 H 前缀（20260815 修复）。
    """
    atom = mol.GetAtomWithIdx(idx)
    if _implicit_shown_hs(atom) == 0:
        return False
    lab = mol_default_labeler(mol)(atom, flip=_label_flip_for(mol, idx))
    plain = re.sub(r"[$_{}^\\]", "", lab or "")
    sym = atom.GetSymbol()
    return plain.startswith("H") and not plain.startswith(sym)


def _bond_blocks(mol, idx: int) -> list:
    """规范定义的 block：直接相连的化学键和原子方向 + 标签氢方向。

    标签氢方向：H 前缀（H₂O/HF/HCl，含 flip 场景）→ 左侧 180°；
    H 后缀（OH₂/CH₃ 等）→ 右侧 0°——问题 7：原固定 0° 使 HCl 的
    孤对电子误占左侧与 H 标签重叠。电荷不是 block（规范仅要求
    电荷与孤对电子不重叠），单独作为避让约束。
    """
    atom = mol.GetAtomWithIdx(idx)
    blocked = _bond_angles(mol, idx)
    if _implicit_shown_hs(atom) > 0:
        # H 前缀（标签以 H 开头，含 flip）→ H 在左侧 180°；否则 H 在右侧 0°
        blocked.append(180.0 if _label_h_is_left(mol, idx) else 0.0)
    return blocked


def _blocked_angles(mol, idx: int) -> list:
    """孤对电子的避让方向全集：block + 电荷位置（45°/135°）。"""
    blocked = _bond_blocks(mol, idx)
    if mol.GetAtomWithIdx(idx).GetFormalCharge() != 0:
        blocked.append(_charge_angle(mol, idx))
    return blocked


def _nudge_from_avoid(ang: float, avoid: list) -> float:
    """角度与避让方向过近（≤30°）时，向 ±30°/±60°/90° 微调至安全位置。"""
    if _min_ang_diff(ang, avoid) > 30:
        return ang
    for delta in (30.0, -30.0, 60.0, -60.0, 90.0, -90.0):
        cand = (ang + delta) % 360.0
        if _min_ang_diff(cand, avoid) > 30:
            return cand
    return ang


def _cardinal(ang: float) -> float:
    """角度归入上下左右四个方位（倾斜键归入左/右）。"""
    a = ang % 360.0
    if 45.0 <= a < 135.0:
        return 90.0
    if 135.0 <= a < 225.0:
        return 180.0
    if 225.0 <= a < 315.0:
        return 270.0
    return 0.0


def _min_ang_diff(ang: float, blocked: list) -> float:
    return min((_ang_diff(ang, b) for b in blocked), default=180.0)


def _separate_cardinals(count: int, blocked: list, taken: list,
                        cardinals: tuple | None = None) -> list:
    """核心原则兜底：正交四向优先、斜向补充，避开阻挡与已占槽位（>30°）。

    cardinals: 候选序（默认上→左→下→右→斜向）。单电子（自由基）传
    水平优先序（左→右→上→下），使 ·CH₃ 的电子点画在碳左侧更美观
    （问题 5：默认序优先上方，自由基单电子在水平方向更自然）。
    """
    result = list(taken)
    for cand in cardinals or (90.0, 180.0, 270.0, 0.0,
                              45.0, 135.0, 225.0, 315.0):
        if len(result) >= count + len(taken):
            break
        if (_min_ang_diff(cand, blocked) > 30
                and _min_ang_diff(cand, result) > 30):
            result.append(cand)
    return result[len(taken):]


def _place_pairs(pairs: int, blocked: list) -> list:
    """按《孤对电子标注规范》（Drawbacks 第 3 条）摆放 pairs 对孤对电子。

    规则按 (pairs, block 数) 分派；未列出者按核心原则（正交、尽量分离）。
    """
    n_blocks = len(blocked)
    if pairs == 3 and n_blocks == 1:
        # 按 block 方位分离（如 block 左 → 上右下）
        b = _cardinal(blocked[0])
        return [c for c in (90.0, 0.0, 270.0, 180.0) if c != b][:3]
    if pairs == 2 and n_blocks == 2:
        a1, a2 = blocked[0] % 360.0, blocked[1] % 360.0
        # 特例：左右两根倾斜键同朝上（水型）→ 左下/右下张开45°；朝下同理
        # （角度带 ±10° 容差，兼容 RDKit 浮点坐标）
        if all(20.0 <= a <= 160.0 for a in (a1, a2)):
            return [225.0, 315.0]
        if all(200.0 <= a <= 340.0 for a in (a1, a2)):
            return [45.0, 135.0]
        b1, b2 = _cardinal(a1), _cardinal(a2)
        return [c for c in (90.0, 180.0, 270.0, 0.0) if c not in (b1, b2)][:2]
    if pairs == 2 and n_blocks == 1:
        # 三者夹角 ≈120°（三角对称 ±120°，允许 30° 整数倍）
        b = blocked[0] % 360.0
        return [(b - 120.0) % 360.0, (b + 120.0) % 360.0]
    if pairs == 1 and n_blocks == 3:
        # 正交四向中取未被 block 方位占用者
        bs = {_cardinal(a) for a in blocked}
        for c in (90.0, 180.0, 270.0, 0.0):
            if c not in bs:
                return [c]
        return [90.0]
    if pairs == 1 and n_blocks == 2:
        # 30° 整数倍中离两个 block 最远（理想与两者各成 120°）
        best, best_score = 90.0, -1.0
        for cand in range(0, 360, 30):
            score = min(_ang_diff(float(cand), b % 360.0) for b in blocked)
            if score > best_score:
                best, best_score = float(cand), score
        return [best]
    if pairs == 1 and n_blocks == 1:
        # block 正对侧
        return [(blocked[0] + 180.0) % 360.0]
    return _separate_cardinals(pairs, blocked, [])


def lone_pair_angles(mol, idx: int) -> list:
    """孤对电子的放置角度（按规范规则；block 为键与标签氢，电荷仅避让）。"""
    atom = mol.GetAtomWithIdx(idx)
    pairs, _ = lone_pair_count(atom)
    if pairs == 0:
        return []
    # 结构简式 H₂O（隐式 H，O 无显式邻居，度 0）：按 1 block & 2 pair 规则
    # （Instruction-for-Electrons §3.3 第 24 行：block 在 180°（左侧标签 H）
    # → pair 在 60° 和 300°）。显式 H 的 Lewis 水（度 2，两根 O-H 键）不走
    # 此分支，由 _place_pairs 按 block=2 特例产出左下/右下 225°/315°。
    if atom.GetAtomicNum() == 8 and pairs == 2 and atom.GetDegree() == 0:
        return [60.0, 300.0]
    avoid = [_charge_angle(mol, idx)] if atom.GetFormalCharge() != 0 else []
    angles = _place_pairs(pairs, _bond_blocks(mol, idx))
    if avoid:
        angles = [_nudge_from_avoid(a, avoid) for a in angles]
    return angles


def single_electron_angles(mol, idx: int) -> list:
    """单电子（自由基）的放置角度，规则同孤对电子并避开已占电子点。

    候选序水平优先（左 180 → 右 0 → 上 90 → 下 270）——自由基单电子
    画在原子水平方向更自然（如 ·CH3 的电子点在碳左侧，问题 5）；
    默认 `_separate_cardinals` 优先上方是孤对电子惯例，自由基不适用。
    """
    _, singles = lone_pair_count(mol.GetAtomWithIdx(idx))
    if singles == 0:
        return []
    taken = lone_pair_angles(mol, idx)
    blocked = _blocked_angles(mol, idx) + list(taken)
    return _separate_cardinals(
        singles, blocked, taken,
        cardinals=(180.0, 0.0, 90.0, 270.0, 135.0, 225.0, 45.0, 315.0),
    )[:singles]


def _weighted_plain_len(lab: str) -> float:
    """标签可视宽度（普通字符 1 单位；$...$ 数学块内字符降权）。

    下标/上标是 LaTeX 数学模式小字，视觉宽度小于普通字符；元素符号
    中心的偏移估算需按加权宽度，否则对含下标标签（如 OH₂ 的 $_{2}$）
    偏大。权重取 _SUBSCRIPT_CHAR_W（实测 OH₂ 反推 ≈0.67）。
    """
    w = 0.0
    in_math = False
    for c in lab:
        if c == "$":
            in_math = not in_math
        elif c in "{}^_\\":
            pass
        else:
            w += _SUBSCRIPT_CHAR_W if in_math else 1.0
    return w


def _dot_center(mol, idx: int, explicit_hs: int = 0) -> tuple[float, float]:
    """孤对电子点的环绕中心：元素符号在标签内的估计位置。

    元素符号在标签**左端**（如 OH、CH₃）→ 左移修正（点绕 O 而非绕 OH）；
    在标签**右端**（如水的 H₂O）→ 右移修正。偏移按加权宽度估算
    （下标/上标小字降权）。含下标标签文字主体相对 node 中心上移
    （下标占下方空间）→ 环绕中心随之上移补偿，避免电荷/电子点重叠。
    explicit_hs 已显式画出的 H 会同步缩小后缀宽度。
    flip 感知（_label_flip_for）：键端在标签右侧时绘制标签翻转
    （OH→HO，元素符号从标签左端变右端），环绕中心必须按翻转后的
    标签文本修正，否则电荷/孤对电子点错位约半个标签宽。
    """
    atom = mol.GetAtomWithIdx(idx)
    x, y = atom_pos(mol, idx)
    lab = mol_default_labeler(mol)(atom, explicit_hs,
                                   flip=_label_flip_for(mol, idx))
    if lab:
        sym = atom.GetSymbol()
        sym = sym[0].upper() + sym[1:]
        plain = re.sub(r"[$_{}^\\]", "", lab)
        if plain.startswith(sym):
            x -= _CHAR_HALF_W * (_weighted_plain_len(lab)
                                 - _weighted_plain_len(sym))
        elif plain.endswith(sym):
            x += _CHAR_HALF_W * (_weighted_plain_len(lab)
                                 - _weighted_plain_len(sym))
        if "$_{" in lab:
            y += _SUBSCRIPT_DEPTH_COMP
    return x, y


def symbol_center(mol, idx: int, explicit_hs: int = 0) -> tuple[float, float]:
    """元素符号中心坐标（孤对电子/部分电荷等标注的环绕中心）。"""
    return _dot_center(mol, idx, explicit_hs)


def atom_main_label(atom, explicit_hs: int = 0, flip: bool = False) -> str | None:
    """主标签（元素符号 + H 计数，**不含电荷**；纯碳环原子返回 None）。

    explicit_hs：已显式画出的 H 数（[XH]/氢键给体），从标签 H 计数中
    扣除，保证"标签 H + 画出 H"总数正确（如 OH 画出 H 后标签为 O）。
    电荷由 atom_charge_label 单独给出（圆圈形式标注）。
    flip=True：键端在标签右侧时元素符号右移（OH→HO，Drawbacks 第 6 条）。
    """
    if atom.GetAtomicNum() == 6 and atom.IsInRing():
        return None
    # 通用基团占位符（R/X/Ph/Ac 等）：显示缩写文本
    if atom.HasProp("_abbr"):
        return atom.GetProp("_abbr")
    if atom.GetAtomicNum() == 6:
        sym = "C"
    else:
        sym = atom.GetSymbol()
        sym = sym[0].upper() + sym[1:]
    h = max(0, atom.GetTotalNumHs() - explicit_hs)
    # 氢化物惯例 H 前置（HF/HCl/HBr/HI/H2O/H2S，见 _H_PREFIX_ELEMENTS）；
    # 正离子氢化物同惯例（H3O⁺ 而非 OH3⁺，20260815）；负离子保持 XH
    # （如 OH⁻ 写 OH，不写 HO）；碳始终 CHn；N/P/B/Si
    # 氢化物写 NH3/PH3/BH3/SiH4（元素在前）——问题 2（NH3→H3N）
    if (h >= 1 and atom.GetAtomicNum() in _H_PREFIX_ELEMENTS
            and atom.GetFormalCharge() >= 0 and _only_h_neighbors(atom)):
        parts = (f"H$_{{{h}}}$" if h > 1 else "H") + sym
    elif flip:
        parts = (f"H$_{{{h}}}$" if h > 1 else ("H" if h == 1 else "")) + sym
    else:
        parts = sym
        if h == 1:
            parts += "H"
        elif h > 1:
            parts += f"H$_{{{h}}}$"
    return parts


def atom_charge_label(atom) -> str | None:
    """电荷标签（+/−/2+/2−），无形式电荷返回 None。"""
    fc = atom.GetFormalCharge()
    if not fc:
        return None
    num = str(abs(fc)) if abs(fc) > 1 else ""
    sign = "+" if fc > 0 else "-"
    return f"${num}{sign}$"


_CHARGE_POS_DIST = 0.34    # 电荷到元素符号中心的距离（原 0.42，调至 0.34；不与孤对电子重叠）
# 无标签碳（键线式：atom_label 返回 None，含带电碳）的电荷圈距离——
# 使 45° 方向上水平/竖直分量 ≈ 0.15（原 0.10；0.34×cos45° ≈ 0.24 为
# 有标签原子的分量），电荷圈更贴近原子，不与键线交点混淆。
# 保持角度只改距离：任意角度下水平/竖直分量 |dist·cos/sin(ang)| 均 ≤ 0.15。
_CHARGE_POS_DIST_NO_LABEL = 0.15 / math.cos(math.radians(45.0))
_CHARGE_SCALE = 0.5        # 电荷圈缩放（为默认大小的一半）


def _charge_dist(mol, idx: int, explicit_hs: int = 0) -> float:
    """电荷圈径向距离：原子无可见标签（键线式碳，含带电碳）时贴近
    （水平/竖直分量 0.10）；有标签（杂原子/结构简式碳）保持 0.34。
    与绘制端同一 labeler（mol_default_labeler）判断，保证口径一致。"""
    if mol_default_labeler(mol)(mol.GetAtomWithIdx(idx), explicit_hs) is None:
        return _CHARGE_POS_DIST_NO_LABEL
    return _CHARGE_POS_DIST


def _charge_angle(mol, idx: int) -> float:
    """电荷圈方位角：已被候选位放置选定（mol prop 缓存）时读缓存；
    否则按规则——右侧有标签氢阻碍且左侧无阻碍时在左上（135°），
    否则在右上（45°）（左右都有阻碍时保持右上）。标签 H 前缀
    （H₂O/HF，含 flip）视为左侧阻碍（_label_h_is_left）。"""
    prop = f"_charge_ang_{idx}"
    if mol.HasProp(prop):
        return float(mol.GetProp(prop))
    if _implicit_shown_hs(mol.GetAtomWithIdx(idx)) == 0:
        return 45.0
    left_blocked = _label_h_is_left(mol, idx) or any(
        _ang_diff(a, 180.0) <= 45.0 for a in _bond_angles(mol, idx)
    )
    return 45.0 if left_blocked else 135.0


def charge_tikz(mol, idx: int, shift=(0.0, 0.0), explicit_hs: int = 0,
                occupancy=None) -> str | None:
    r"""圆圈电荷节点；无电荷返回 None。

    occupancy 非空时启用候选位放置（R-8）：候选角 = [现有规则首选, 镜像,
    上, 下, 右, 左] × 距离（有标签 0.34/0.44；无标签碳 0.141/0.241），
    逐个查占据表取第一个零冲突者（不撞键/标签/其他电荷圈）；
    首选即现有规则输出，零冲突时图面不变。
    选定角度写入 mol 属性（_charge_ang_{idx}），孤对电子避让读取同一角度。
    """
    text = atom_charge_label(mol.GetAtomWithIdx(idx))
    if text is None:
        return None
    cx, cy = _dot_center(mol, idx, explicit_hs)
    d0 = _charge_dist(mol, idx, explicit_hs)
    if occupancy is not None:
        base = _charge_angle(mol, idx)
        cands = []
        for dist in (d0, d0 + 0.10):
            for ang in (base, (180.0 - base) % 360.0, 90.0, 270.0, 0.0, 180.0):
                r = math.radians(ang)
                cands.append(("circle",
                              cx + dist * math.cos(r), cy + dist * math.sin(r),
                              CHARGE_CIRCLE_R))
        chosen, _ = occupancy.place(
            cands, warn=f"电荷圈（原子 {idx}）候选位全部冲突")
        ang = math.degrees(math.atan2(chosen[2] - cy, chosen[1] - cx)) % 360.0
        mol.SetProp(f"_charge_ang_{idx}", f"{ang:.1f}")
        x, y = chosen[1] + shift[0], chosen[2] + shift[1]
    else:
        r = math.radians(_charge_angle(mol, idx))
        x = cx + shift[0] + d0 * math.cos(r)
        y = cy + shift[1] + d0 * math.sin(r)
    return (f"\\node[draw, circle, inner sep=0.6pt, font=\\scriptsize, "
            f"scale={_CHARGE_SCALE}] at ({x:.2f},{y:.2f}) {{{text}}};")


def partial_charge_angle(mol, idx: int, explicit_hs: int = 0) -> float:
    """部分电荷（δ+/δ-）标注方位角（Drawbacks 手动测试第 7 条）。

    起点与形式电荷圈一致（_charge_angle：默认右上 45°；标签氢在右侧
    且左侧无阻碍时左上 135°）；**原子带形式电荷时起点 +45° 偏移**，
    避免 δ 标注与圆圈电荷同位重叠（P2）；再按避让方向（直接相连的
    键/原子 + 标签氢 + **该原子实际孤对电子点槽位**）微调（过近 ≤30°
    时向 ±30°/±60°/90° 微调；极端拥挤保持原角）。
    """
    base = _charge_angle(mol, idx)
    if mol.GetAtomWithIdx(idx).GetFormalCharge() != 0:
        base = (base + 45.0) % 360.0
    avoid = _bond_blocks(mol, idx)
    # 孤对电子点槽位角度（P3'）：δ 标注不得压住电子点
    groups, singles = lone_pair_dot_groups(mol, idx, explicit_hs=explicit_hs)
    cx, cy = _dot_center(mol, idx, explicit_hs)
    for px, py in [p for g in groups for p in g] + singles:
        avoid.append(math.degrees(math.atan2(py - cy, px - cx)) % 360.0)
    return _nudge_from_avoid(base, avoid)


def partial_charge_pos(mol, idx: int, shift=(0.0, 0.0),
                       explicit_hs: int = 0) -> tuple[float, float]:
    """部分电荷标注的画布坐标（元素符号中心 + 方向避让，见 partial_charge_angle）。

    距离复用 _charge_dist（与形式电荷圈一致）：无标签碳（键线式端点）
    水平/竖直分量 0.10，有标签原子保持 0.34。
    """
    cx, cy = _dot_center(mol, idx, explicit_hs)
    r = math.radians(partial_charge_angle(mol, idx, explicit_hs))
    d = _charge_dist(mol, idx, explicit_hs)
    return (cx + shift[0] + d * math.cos(r),
            cy + shift[1] + d * math.sin(r))


def lone_pair_dot_groups(mol, idx: int, shift=(0.0, 0.0), explicit_hs: int = 0):
    """孤对电子点的画布坐标。

    点以元素符号为中心（_dot_center，explicit_hs 已显式画出的 H 会同步
    缩小标签后缀），距离固定 _LP_DIST。
    返回:
        (groups, singles)：groups 为每对电子的两个点坐标列表
        [((x1,y1),(x2,y2)), ...]，singles 为单电子点坐标列表 [(x,y), ...]。
    """
    atom = mol.GetAtomWithIdx(idx)
    pairs, _ = lone_pair_count(atom)
    cx0, cy0 = _dot_center(mol, idx, explicit_hs)
    cx0 += shift[0]
    cy0 += shift[1]
    groups = []
    for ang in lone_pair_angles(mol, idx):
        r = math.radians(ang)
        cx = cx0 + _LP_DIST * math.cos(r)
        cy = cy0 + _LP_DIST * math.sin(r)
        px, py = -math.sin(r), math.cos(r)
        sep = 0.055
        groups.append(((cx + px * sep, cy + py * sep),
                       (cx - px * sep, cy - py * sep)))
    singles = []
    for ang in single_electron_angles(mol, idx):
        r = math.radians(ang)
        singles.append((cx0 + _LP_DIST * math.cos(r), cy0 + _LP_DIST * math.sin(r)))
    return groups, singles


def lone_pair_tikz(mol, idx: int, shift=(0.0, 0.0), explicit_hs: int = 0) -> list:
    r"""生成孤对电子点的 \fill 圆点线条列表（裸行，缩进由调用方决定）。"""
    groups, singles = lone_pair_dot_groups(mol, idx, shift, explicit_hs)
    lines = []
    for (x1, y1), (x2, y2) in groups:
        lines.append(f"\\fill ({x1:.2f},{y1:.2f}) circle (0.028);")
        lines.append(f"\\fill ({x2:.2f},{y2:.2f}) circle (0.028);")
    for x, y in singles:
        lines.append(f"\\fill ({x:.2f},{y:.2f}) circle (0.028);")
    return lines


def mol_visual_bbox(mol, labeler=None,
                    include_lone_pairs: bool = True,
                    charge_mirror: bool = True):
    """分子视觉包围盒 (min_x, min_y, max_x, max_y)。

    在原子坐标基础上计入标签半径与孤对电子点的外延，
    供布局间距计算使用，避免相邻组件重叠。
    labeler: 原子标签函数；缺省按 mol_default_labeler 的
    重原子数规则选择（与绘制端一致）。
    charge_mirror: 形式电荷圈只在上半部（45°/135°），默认把其上界镜像
    到 y 下界（使带电分子 bbox 中心仍落在原子线上，布局排布不整体下移）；
    为 False 时只计真实上界——用于"朝某侧附件贴靠"的场景（如副反应物/
    副产物贴箭头放置，需要真实朝向边，镜像下界会失真）。
    """
    xs, ys = [], []
    labeler = labeler or mol_default_labeler(mol)
    for atom in mol.GetAtoms():
        x, y = atom_pos(mol, atom.GetIdx())
        hw = hh = 0.05
        lab = labeler(atom)
        if lab:
            hw = max(0.18, label_visual_width(lab) / 2.0)
            hh = 0.18
        xs += [x - hw, x + hw]
        ys += [y - hh, y + hh]
        if atom.GetFormalCharge() != 0:
            d0 = _charge_dist(mol, atom.GetIdx())
            r = math.radians(_charge_angle(mol, atom.GetIdx()))
            xs.append(x + d0 * math.cos(r) + 0.1 * math.cos(r))
            dy = d0 * math.sin(r) + 0.1
            ys.append(y + dy)
            if charge_mirror:
                # 电荷圈总在上半部（45°/135°），只计入上界会把包围盒中心抬高、
                # 使带电分子整体下移（OH⁻/Cl⁻ 标签比中性分子低 ~0.05）；镜像
                # 下界抵消单侧偏移，让包围盒中心仍落在原子线上。
                ys.append(y - dy)
        if include_lone_pairs:
            groups, singles = lone_pair_dot_groups(mol, atom.GetIdx())
            for (x1, y1), (x2, y2) in groups:
                xs += [x1, x2]
                ys += [y1, y2]
            for x1, y1 in singles:
                xs.append(x1)
                ys.append(y1)
    return min(xs), min(ys), max(xs), max(ys)


def mol_visual_bbox_xh(mol, xh_counts: dict, h_len_scale: float = 1.0,
                       include_lone_pairs: bool = True):
    """含 [XH] 显式 H 外延的分子视觉包围盒（问题 6）。

    在 mol_visual_bbox 基础上，把 XH 画出的 H 节点坐标计入外延——
    H 在 X—H 键端点，可能远超原子标签 bbox（孤立碳 CH4 的 H 在
    C—H 键长 2×Rc×scale 处）。h_len_scale 为调用方分子缩放因子
    （composite 用 _MOL_SCALE；h_len 逻辑与渲染端 place_explicit_hs
    一致，保证布局感知与画面一致）。
    """
    min_x, min_y, max_x, max_y = mol_visual_bbox(
        mol, include_lone_pairs=include_lone_pairs)
    if not xh_counts:
        return (min_x, min_y, max_x, max_y)
    mol_has_bond = mol.GetNumBonds() > 0
    for a, count in xh_counts.items():
        if a >= mol.GetNumAtoms():
            continue
        h_len = None
        if not mol_has_bond:
            h_len = _covalent_bond_len(mol.GetAtomWithIdx(a)) * h_len_scale
        for hx, hy in place_explicit_hs(mol, a, count, h_len=h_len):
            min_x = min(min_x, hx)
            max_x = max(max_x, hx)
            min_y = min(min_y, hy)
            max_y = max(max_y, hy)
    return (min_x, min_y, max_x, max_y)


def aromatic_ring_info(mol) -> list:
    """全芳香单环的几何信息：[(原子集合, 质心x, 质心y, 半径), ...]。

    仅含"所有原子都是芳香原子"的单环（不碰并环）——用于画圈：
    bond_segments 跳过这些环的环内键，调用方在质心画圆。
    非全芳香环（如 σ 络合物的环己二烯、含 sp³ 碳的环）不在此列，保持交替键。
    """
    out = []
    try:
        from rdkit import Chem
        ri = mol.GetRingInfo()
    except Exception:
        return out
    for ring in ri.AtomRings():
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        if any(ri.NumAtomRings(i) != 1 for i in ring):
            continue
        xs = [atom_pos(mol, i)[0] for i in ring]
        ys = [atom_pos(mol, i)[1] for i in ring]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        # 圈半径取"环键中点到质心"的平均距离（对正六边形 = 0.866×顶点半径），
        # 再 ×0.70 使圈明显小于内切圆、位于六边形内部（不碰边、不穿顶点，
        # 教科书带圈苯风格）
        radii = []
        for k in range(len(ring)):
            aj = ring[(k + 1) % len(ring)]
            mx = (xs[k] + atom_pos(mol, aj)[0]) / 2.0
            my = (ys[k] + atom_pos(mol, aj)[1]) / 2.0
            radii.append(math.hypot(mx - cx, my - cy))
        radius = 0.70 * sum(radii) / len(radii)
        out.append((set(ring), cx, cy, radius))
    return out


def _regularize_kekule(mol) -> None:
    """全芳香单环按几何规则重排交替单双键（同类环画法一致）。

    PrepareMolForDrawing 的 Kekulé 由原子规范序决定，同一苯环在不同分子
    （苯 / 苯胺 / 硝基苯）中可能得到不同键型，反应前后看起来像两个共振式。
    改为按环键中点绕环质心的角度排序，从最小角起 0/2/4 位设双键——
    同类环几何一致则键型一致。仅处理全芳香原子组成的单环（不碰并环）。
    """
    try:
        from rdkit import Chem
    except ImportError:
        return
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        if any(ring_info.NumAtomRings(i) != 1 for i in ring):
            continue
        xs = [atom_pos(mol, i)[0] for i in ring]
        ys = [atom_pos(mol, i)[1] for i in ring]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        bonds = []
        for k, ai in enumerate(ring):
            aj = ring[(k + 1) % len(ring)]
            b = mol.GetBondBetweenAtoms(ai, aj)
            if b is None:
                bonds = None
                break
            x1, y1 = atom_pos(mol, ai)
            x2, y2 = atom_pos(mol, aj)
            ang = math.degrees(
                math.atan2((y1 + y2) / 2 - cy, (x1 + x2) / 2 - cx)) % 360.0
            bonds.append((ang, b))
        if bonds is None:
            continue
        bonds.sort()
        for k, (_, b) in enumerate(bonds):
            b.SetBondType(Chem.BondType.DOUBLE if k % 2 == 0
                          else Chem.BondType.SINGLE)
            b.SetIsAromatic(False)


def has_aromatic_lowercase(smiles: str) -> bool:
    """SMILES 字符串是否含芳香小写原子符号（c/n/o/s/p 小写）。

    区分两种苯环写法：`c1ccccc1`（芳香小写→画圈）与 `C1=CC=CC=C1`
    （凯库勒大写→交替键）。RDKit 解析后两者无法区分（芳香信息被归一化），
    必须从原始字符串判断。环外取代基（如甲苯的甲基 C）不受影响——
    只要环本身用芳香小写即返回 True。
    """
    if not smiles or not isinstance(smiles, str):
        return False
    # 芳香小写原子 c/n/o/s/p（小写，独立原子符号）：排除元素名内小写
    # （Cl 的 l、Br 的 r、Na 的 a）——(?<![a-z]) 允许大写前缀（如 Cc 的 c 是芳香碳）
    return bool(re.search(r"(?<![a-z])[cnops](?![a-z])", smiles))


def prepare_mol(smiles: str, *, add_hs: bool = False, kekulize: bool = False,
                use_prepare: bool = True, allow_aromatic: bool | None = None):
    """SMILES → RDKit Mol：解析、可选加氢/Kekulize、计算 2D 坐标。

    参数:
        smiles: 输入 SMILES。
        add_hs: 是否调用 AddHs（Lewis 结构需要显示所有 H）。
        kekulize: 是否 Kekulize（Lewis 需要明确单双键）。
        use_prepare: 是否优先用 rdMolDraw2D.PrepareMolForDrawing；
                     为 False 时直接用 AllChem.Compute2DCoords。
        allow_aromatic: 芳香化判定开关。None（默认）= 按原始 SMILES 自动：
            芳香小写（c1ccccc1）→ True（画圈模式）；凯库勒大写
            （C1=CC=CC=C1）→ False（保留输入单双键，不同 Kekulé 式
            渲染不同）。显式传 True/False 时尊重调用方。

    返回:
        RDKit Mol 对象；解析失败返回 None。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
        from utils.rdkit_utils import mute_rdkit_warnings, \
            normalize_h_prefix_smiles, expand_group_abbrevs
    except ImportError:
        return None

    # H 数字前缀写法（[H3O+]）规范化为合法 SMILES（[OH3+]）再解析；
    # 通用基团缩写（R/X/Ph/Ac 等）替换为 dummy 原子（[*:n]），解析后
    # 给 dummy 原子设 _abbr prop（标签端显示缩写文本）
    # （20260815：与 tag_validator._parse_mol 同口径）
    smiles = normalize_h_prefix_smiles(smiles)
    smiles, abbr_map = expand_group_abbrevs(smiles)

    # 解析阶段统一屏蔽 rdApp.error：探测性解析（含双轨制下 KMnO4/H2SO4 等
    # 公式物种试解析）失败是常态，RDKit 的 SMILES Parse Error 对用户与日志
    # 均无价值（与 core.tag_validator._parse_mol 的屏蔽口径一致）；
    # 失败由返回 None + 调用方的可读错误串承接。游离氢警告同样被屏蔽。
    if allow_aromatic is None:
        # 自动：芳香小写→True（画圈），凯库勒大写→False（保留键级）
        allow_aromatic = has_aromatic_lowercase(smiles)
    with mute_rdkit_warnings(include_error=True):
        if allow_aromatic:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
        else:
            mol = Chem.MolFromSmiles(smiles, sanitize=False) if smiles else None
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(
                    mol,
                    Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                )
        if mol is None:
            return None

        if add_hs:
            mol = Chem.AddHs(mol)

        if kekulize:
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                pass

        if use_prepare:
            try:
                prepared = rdMolDraw2D.PrepareMolForDrawing(mol)
                if prepared is not None:
                    mol = prepared
            except Exception:
                AllChem.Compute2DCoords(mol)
        else:
            AllChem.Compute2DCoords(mol)

    # 不再调用 _regularize_kekule：芳香/凯库勒画法由原始 SMILES 大小写决定
    # （has_aromatic_lowercase）。把判断结果存到 mol property——
    # molecule_scope_lines 等调用方可直接读取决定画圈（避免逐层传参）。
    if mol is not None:
        try:
            mol.SetProp("_aromatic_lowercase", "1" if has_aromatic_lowercase(smiles) else "0")
        except Exception:
            pass
        # 通用基团占位符（dummy 原子，expand_group_abbrevs 的 [*:n]）：
        # 按 atom map 回写缩写文本，标签端（atom_label/atom_main_label）
        # 读 _abbr prop 显示 R/X/Ph/Ac 等。PrepareMolForDrawing 会复制
        # 分子，因此 prop 在最终分子上设置。数字转下标（R1 → R$_1$）。
        if abbr_map:
            for a in mol.GetAtoms():
                mp_ = a.GetAtomMapNum()
                if mp_ and mp_ in abbr_map:
                    try:
                        a.SetProp("_abbr", format_chem_text(abbr_map[mp_]))
                    except Exception:
                        pass
    return mol


def atom_pos(mol, idx: int) -> tuple[float, float]:
    """返回原子 idx 的 2D 坐标 (x, y)。"""
    conf = mol.GetConformer()
    p = conf.GetAtomPosition(idx)
    return p.x, p.y


def _label_flip_for(mol, idx: int) -> bool:
    """标签是否应翻转（键端在标签右侧 → 元素符号右移，OH → HO）。

    唯一重原子邻居在原子**右侧**且水平距离占主导时翻转；竖直键
    （90°/270°）、多重原子邻居、无重原子邻居（如水）、**碳原子标签**
    （甲基 CH₃，Drawbacks 第 6 条例外）不翻转。
    """
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetAtomicNum() == 6:
        return False            # 甲基 CH₃ 不替换
    heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
    if len(heavy) != 1:
        return False
    hx, hy = atom_pos(mol, heavy[0].GetIdx())
    ax, ay = atom_pos(mol, idx)
    dx, dy = hx - ax, hy - ay
    return dx > 0.05 and abs(dx) > abs(dy)


def scale_mol_coords(mol, factor: float) -> None:
    """按比例缩放分子 2D 坐标（原地修改），用于机理场景的整体紧凑化。

    只缩放坐标，不影响 TikZ 文字字号；标签留白/电子点距为绝对值，
    由 labeler/margin_fn 另行控制。
    """
    if factor == 1.0:
        return
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x * factor, p.y * factor, p.z))


def rotate_mol_coords(mol, angle_deg: float,
                      center: tuple[float, float] | None = None) -> None:
    """绕 center（默认几何中心）旋转全部原子 2D 坐标（原地修改）。

    用于氢键受体对齐等"分子级刚体变换"：旋转后 bbox/标签随原子移动，
    布局引擎自动按新 bbox 重排。原子顺序不变（SMILES 顺序索引保持）。
    """
    if not angle_deg or angle_deg % 360.0 == 0.0:
        return
    conf = mol.GetConformer()
    if center is None:
        xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
        ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
        center = (sum(xs) / len(xs), sum(ys) / len(ys)) if xs else (0.0, 0.0)
    r = math.radians(angle_deg)
    cosr, sinr = math.cos(r), math.sin(r)
    cx, cy = center
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        dx, dy = p.x - cx, p.y - cy
        conf.SetAtomPosition(
            i, (cx + dx * cosr - dy * sinr, cy + dx * sinr + dy * cosr, p.z))


def bond_type_order(bond) -> int:
    """把 BondTypeAsDouble 规整为 1/2/3（单/双/三键）。"""
    order = bond.GetBondTypeAsDouble()
    if order >= 2.5:
        return 3
    if order >= 1.5:
        return 2
    return 1


def _inner_double_segment(xi, yi, xj, yj, cx, cy, gap):
    """环内双键的内侧平行线（教科书式内缩短双键）。

    从键中点朝环质心方向偏移 gap；端点钳制在「质心 → 两顶点」的
    射线上（中心、双键端点、顶点三点共线），长度自然满足
    长度 = 距中心距离 × 2/√3（正多边形几何）。
    退化（质心在键上/射线平行）返回 None，由调用方回退简单偏移。
    """
    mx, my = (xi + xj) / 2.0, (yi + yj) / 2.0
    vx, vy = cx - mx, cy - my
    dist = math.hypot(vx, vy)
    if dist < 1e-6:
        return None
    sx, sy = vx / dist, vy / dist
    dx, dy = xj - xi, yj - yi
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    nx, ny = mx + sx * gap, my + sy * gap

    def _ray_hit(vx_, vy_):
        ddx, ddy = vx_ - cx, vy_ - cy
        det = -ddx * uy + ddy * ux
        if abs(det) < 1e-9:
            return None
        t = (-(nx - cx) * uy + (ny - cy) * ux) / det
        return cx + ddx * t, cy + ddy * t

    p1 = _ray_hit(xi, yi)
    p2 = _ray_hit(xj, yj)
    if p1 is None or p2 is None:
        return None
    return p1[0], p1[1], p2[0], p2[1]


def bond_segments(mol, *, label_margin: float = 0.25, bond_gap: float = 0.08,
                  labeler=atom_label, margin_fn=None,
                  skip_aromatic_rings: list | None = None):
    """把分子中所有化学键转换为 TikZ 线段坐标列表。

    返回:
        每个键对应一个列表，包含 1 条（单键）或 2/3 条（双/三键）线段坐标：
        [
            [(x1, y1, x2, y2)],                       # 单键
            [(x1, y1, x2, y2), (x1', y1', x2', y2')], # 双键
            ...
        ]

    参数:
        label_margin: 标签原子两端留出的空白距离，避免键压住标签。
        bond_gap: 双键/三键平行线之间的间距。
        labeler: 原子标签函数（默认 atom_label 键线式；调用方可按
            分子大小传 mol_default_labeler(mol) 以对齐结构简式规则）。
        margin_fn: 按标签文本计算留白距离的函数；缺省统一用 label_margin。
        skip_aromatic_rings: 全芳香单环的原子集合列表——这些环的**环内键
            全部按单线绘制**（保留完整骨架，双键不画平行线，由调用方补画
            圆圈），环外键（取代基）正常画。
    """
    skip_bonds = set()
    for ring_atoms in (skip_aromatic_rings or []):
        rset = set(ring_atoms)
        for b in mol.GetBonds():
            if b.GetBeginAtomIdx() in rset and b.GetEndAtomIdx() in rset:
                # 环内键：强制单线（圈替代双键平行线，保留六边形骨架）
                skip_bonds.add((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))

    def _margin(a):
        lab = labeler(a)
        if not lab:
            return 0.0
        if margin_fn is not None:
            return margin_fn(lab)
        return label_margin

    # 各环的原子集合与质心（环内双键内侧偏移用）
    ring_centers = []
    try:
        atom_rings = mol.GetRingInfo().AtomRings()
    except Exception:
        atom_rings = []
    for ring in atom_rings:
        xs = [atom_pos(mol, i)[0] for i in ring]
        ys = [atom_pos(mol, i)[1] for i in ring]
        ring_centers.append((set(ring), sum(xs) / len(xs), sum(ys) / len(ys)))

    segments = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        in_aromatic = (i, j) in skip_bonds or (j, i) in skip_bonds
        if in_aromatic:
            # 芳香环内键：强制单线（圈替代双键平行线）
            xi, yi = atom_pos(mol, i)
            xj, yj = atom_pos(mol, j)
            dx, dy = xj - xi, yj - yi
            L = math.hypot(dx, dy) or 1.0
            ux, uy = dx / L, dy / L
            si = _margin(mol.GetAtomWithIdx(i))
            sj = _margin(mol.GetAtomWithIdx(j))
            segments.append([(xi + ux * si, yi + uy * si,
                              xj - ux * sj, yj - uy * sj)])
            continue
        xi, yi = atom_pos(mol, i)
        xj, yj = atom_pos(mol, j)

        order = bond_type_order(b)
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        px, py = -uy, ux

        si = _margin(mol.GetAtomWithIdx(i))
        sj = _margin(mol.GetAtomWithIdx(j))
        x1, y1 = xi + ux * si, yi + uy * si
        x2, y2 = xj - ux * sj, yj - uy * sj

        if order == 1:
            segments.append([(x1, y1, x2, y2)])
        else:
            # 偏移侧：环内双键朝环质心，链上双键保持原方向
            sx, sy = px, py
            center = None
            if b.IsInRing():
                mx, my = (xi + xj) / 2.0, (yi + yj) / 2.0
                for atoms, cx, cy in ring_centers:
                    if i in atoms and j in atoms:
                        center = (cx, cy)
                        if (cx - mx) * px + (cy - my) * py < 0:
                            sx, sy = -px, -py
                        break
            segs = [(x1, y1, x2, y2)]
            inner = None
            if order == 2 and center is not None:
                inner = _inner_double_segment(xi, yi, xj, yj,
                                              center[0], center[1], 0.18 * L)
            if inner is not None:
                segs.append(inner)
            else:
                segs.append((x1 + sx * bond_gap, y1 + sy * bond_gap,
                             x2 + sx * bond_gap, y2 + sy * bond_gap))
            if order == 3:
                segs.append((x1 - sx * bond_gap, y1 - sy * bond_gap,
                             x2 - sx * bond_gap, y2 - sy * bond_gap))
            segments.append(segs)

    return segments


def bond_segments_for(mol, a: int, b: int, *, labeler=atom_label,
                      margin_fn=None, bond_gap: float = 0.08) -> list | None:
    """指定原子对 (a, b) 的键线段坐标（与 bond_segments 同一修剪逻辑）。

    返回与骨架完全一致（含标签留白修剪、双键偏移）的线段列表；
    原子对不存在键时返回 None。供 [BOND] 突出覆盖、[XH] 校验等复用，
    保证新增线条与已有骨架键完全对齐、不压标签。
    """
    segs_all = bond_segments(mol, labeler=labeler, margin_fn=margin_fn,
                             bond_gap=bond_gap)
    for bi, bond in enumerate(mol.GetBonds()):
        if (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) in ((a, b), (b, a)):
            if bi < len(segs_all):
                return segs_all[bi]
    return None


def fmt_coord(x: float, y: float) -> str:
    """把坐标格式化为 TikZ 常用的两位小数字符串。"""
    return f"({x:.2f},{y:.2f})"


_SUBSCRIPT_RE = re.compile(r"([A-Za-z\)])(\d+)")
_CHARGE_TAIL_RE = re.compile(r"(?:(?<![A-Za-z\)])(\d+))?([+-])$")

# Unicode 上下标映射（直接转 LaTeX 上/下标命令，保留上下标语义——
# 不能 normalize 成普通数字，否则 Ca²⁺ 的 ² 会被 _SUBSCRIPT_RE 误转下标）
_SUBSCRIPT_CHARS = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-",
}
_SUPERSCRIPT_CHARS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁼": "=",
}


def _convert_unicode_scripts(text: str) -> str:
    """Unicode 上下标字符 → LaTeX 上/下标命令（连续字符合成一组）。

    例：H₂SO₄ → H$_{2}$SO$_{4}$；Ca²⁺ → Ca$^{2+}$；SO₄²⁻ → SO$_{4}$$^{2-}$。
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _SUBSCRIPT_CHARS:
            group = []
            while i < n and text[i] in _SUBSCRIPT_CHARS:
                group.append(_SUBSCRIPT_CHARS[text[i]])
                i += 1
            out.append(f"$_{{{''.join(group)}}}$")
        elif ch in _SUPERSCRIPT_CHARS:
            group = []
            while i < n and text[i] in _SUPERSCRIPT_CHARS:
                group.append(_SUPERSCRIPT_CHARS[text[i]])
                i += 1
            out.append(f"$^{{{''.join(group)}}}$")
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# 希腊字母 → LaTeX 数学命令。lmroman 文本字体缺希腊字形（渲染空白），
# 数学模式两引擎（xelatex/pdflatex）均可靠——与 △/Δ 同机制统一转换。
# 覆盖全部 24 小写 + 12 有大写命令的大写（Α Ε Ζ Η Ι Κ Μ Ν Ο Ρ Τ Χ 无
# \uppercase 命令，文本模式字体可用，不转换）。
_GREEK_TO_MATH = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\varepsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "ο": r"\omicron", "π": r"\pi",
    "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon",
    "φ": r"\varphi", "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon",
    "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
}
_GREEK_RE = re.compile("|".join(map(re.escape, _GREEK_TO_MATH)))


def format_chem_text(text: str) -> str:
    """化学文本自动排版：字母/括号后的数字转下标，尾部电荷转上标。

    先剥离尾部电荷（含电荷数），再对余下文本转下标，保证「SO42-」
    中 4 为下标、2- 为上标。已含 $（已手工排版）或为空时原样返回。
     Unicode 上下标（H₂SO₄、H⁺、SO₄²⁻、Ca²⁺ 等）先转 LaTeX 命令。
     加热/希腊符号转数学模式：△（U+25B3）→ $\\triangle$；全部小写
     希腊字母与有大写命令的大写（α β γ δ ε ν π σ ω Γ Δ Θ Λ Ξ Π Σ Υ Φ
     Ψ Ω 等）→ $\\alpha$ / $\\Delta$ 等——lmroman 文本字体缺希腊字形
     （渲染为空白），数学模式两引擎（xelatex/pdflatex）均可靠（与
     energy.py 的 Ea/ΔH 标注约定一致）。

    示例：H2SO4 → H$_2$SO$_4$；CH3Cl → CH$_3$Cl；OH- → OH$^{-}$；
    NH4+ → NH$_4$$^{+}$；SO42- → SO$_4$$^{2-}$；
    H₂SO₄ → H$_{2}$SO$_{4}$；H⁺ → H$^{+}$；Ca²⁺ → Ca$^{2+}$；
    CuO, △ → CuO, $\\triangle$；hν → h$\\nu$；α-碳 → $\\alpha$-碳。
    """
    if not text or "$" in text:
        return text
    text = _convert_unicode_scripts(text)
    charge = ""
    m = _CHARGE_TAIL_RE.search(text)
    if m:
        charge = f"$^{{{m.group(1) or ''}{m.group(2)}}}$"
        text = text[: m.start()]
    out = _SUBSCRIPT_RE.sub(r"\1$_\2$", text) + charge
    # 必须最后替换：提前插入 $ 会使尾部电荷正则 _CHARGE_TAIL_RE 失效
    out = out.replace("△", r"$\triangle$").replace("Δ", r"$\Delta$")
    return _GREEK_RE.sub(lambda m: f"${_GREEK_TO_MATH[m.group(0)]}$", out)


# ---------------------------------------------------------------------------
# 标签自动换行（C2 / P3 暂缓项）
# ---------------------------------------------------------------------------

_LABEL_WRAP_MAX = 3.5    # 标签/条件换行默认最大行宽（视觉宽度单位）
_LABEL_LINE_H = 0.35     # 标签单行高（TikZ 单位，估）


def wrap_label_lines(text: str, max_width: float = _LABEL_WRAP_MAX) -> list:
    """按可视宽度换行长标签/条件（在原始文本上断行）。

    返回行列表；总宽不超过 max_width 时原样单行返回。
    - 含空格（中英混排/条件，如 "浓H2SO4, 加热"）：贪心断行，优先落在
      行内最后一个空格处（保持单词完整）；
    - 纯中文（无空格）：按字符数均分（行数 = ceil(总宽/max_width)），
      避免贪心在中词断行（如"条件"被拆成"条/件"）。
    按字符测量 label_visual_width（CJK 全角 ×2），单字宽度超过
    max_width 时该字独占一行。
    """
    if not text:
        return []
    if label_visual_width(text) <= max_width:
        return [text]
    if " " not in text:
        n = max(1, math.ceil(label_visual_width(text) / max_width))
        per = math.ceil(len(text) / n)
        return [text[i:i + per] for i in range(0, len(text), per)]
    lines, cur = [], ""
    for ch in text:
        trial = cur + ch
        if cur and label_visual_width(trial) > max_width:
            sp = cur.rfind(" ")
            if sp >= 0:
                lines.append(cur[:sp])
                cur = cur[sp + 1:] + ch
            else:
                lines.append(cur)
                cur = ch
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def label_wrapped_size(text: str, max_width: float = _LABEL_WRAP_MAX) -> tuple:
    """换行后标签的 (总宽, 总高)（视觉单位），供布局间距估算。

    总宽 = 各行 label_visual_width 最大值；总高 = 行数 × 行高。
    """
    lines = wrap_label_lines(text, max_width)
    w = max((label_visual_width(l) for l in lines), default=0.0)
    return w, len(lines) * _LABEL_LINE_H


def wrap_format_text(text: str, max_width: float = _LABEL_WRAP_MAX) -> str:
    """长标签/条件自动换行 + 化学排版（TikZ 节点文本）。

    先按可视宽度在原始文本上换行（避免拆开 $...$ 数学段），再逐行
    format_chem_text（下标/电荷/加热符号），多行以 \\\\ 连接——调用方
    需为节点加 align=center。总宽未超 max_width 时等价 format_chem_text。
    """
    if not text:
        return ""
    lines = wrap_label_lines(text, max_width)
    if len(lines) == 1:
        return format_chem_text(lines[0])
    return "\\\\".join(format_chem_text(l) for l in lines)


def split_arrow_condition(cond: str) -> tuple:
    """反应条件按箭头上下拆分：带 "-" 前缀的 token（2b 产物侧补足，
    如 -H2O、-3H2）放箭头下方，其余（催化剂/温度/反应物侧补足）放上方。

    返回 (上方文本, 下方文本)；任一侧为空返回 ""。上方 join 用 ", "，
    末尾不带逗号（拆分后自然无尾逗号）。供 reaction/composite 主箭头渲染。
    """
    tokens = [t.strip() for t in re.split(r"[，,]", cond or "") if t.strip()]
    above, below = [], []
    for t in tokens:
        (below if t.startswith("-") else above).append(t)
    return ", ".join(above), ", ".join(below)


# 可逆令牌：仅 ⇌（20260816 用户决策）。写在条件字段中切换双向箭头，
# 渲染时从显示文本剥离；校验层（_arrow_supplement_tokens）无法把 ⇌ 解析
# 为化学式，自动忽略，无需改动。
_REV_ARROW_TOKENS = re.compile(r"⇌")


def parse_arrow_kind(cond: str) -> tuple:
    """识别条件文本中的可逆令牌 ⇌ 并剥离。

    返回 (kind, 剥离后的条件文本)。kind ∈ {"single", "reversible"}；
    ⇌ 不进入条件显示文本（条件节点只显示实际条件）。
    """
    if _REV_ARROW_TOKENS.search(cond or ""):
        return "reversible", _REV_ARROW_TOKENS.sub("", cond or "").strip(" ,，")
    return "single", cond


def main_arrow_lines(x1: float, x2: float, condition: str = "", *,
                     y: float = 0.0, style: str = "very thick",
                     kind: str = None) -> list:
    """主反应箭头 TikZ 行——单向 → / 双向 ⇌ / 共振 ↔ / 逆合成 ⇒ 统一。

    kind 显式传入（大一统架构 [ARROW:type=...]）时优先；否则按条件中的
    ⇌ 令牌自动识别（旧语法兼容）。单向/可逆输出与迁移前逐字符一致。

    单向：一条右箭头；条件经 split_arrow_condition 上下分挂（正条件/无 - 前缀
    在 above、-X 补足在 below），输出与迁移前逐字符一致（回归锚点）。
    双向（⇌）：两条半箭头水平交错上下排列——上条右指覆盖右侧 ~2/3（尖端在
    x2）、下条左指覆盖左侧 ~2/3（尖端在 x1），教科书平衡符号；条件同上分挂。
    共振（↔）：居中 ↔ 节点（与 RESARROW 一致）。
    逆合成（⇒）：双线推导箭头（普通粗细双线杆 + 开放式折线尖）。
    """
    if kind is not None:
        k, cond = kind, condition
    else:
        k, cond = parse_arrow_kind(condition)
    above, below = split_arrow_condition(cond)

    def _node(text: str, pos: str) -> str:
        t = wrap_format_text(text)
        if not t:
            return ""
        align = "align=center, " if "\\\\" in t else ""
        return f" node[midway, {align}{pos}] {{{t}}}"

    if k == "resonance":
        return [f"  \\node[font=\\large] at ({(x1 + x2) / 2:.2f},{y:.2f}) "
                f"{{\\$\\leftrightarrow$}};"]
    if k == "retro":
        # 逆合成双线推导箭头（⇒，与 retro.py 画法一致，提取为共享）
        tip, add = 0.23, 0.14
        base = x2 - tip
        tail = base + add
        return [
            f"  \\draw ({x1:.2f},{y + 0.05:.2f}) -- ({tail:.2f},{y + 0.05:.2f});",
            f"  \\draw ({x1:.2f},{y - 0.05:.2f}) -- ({tail:.2f},{y - 0.05:.2f});",
            f"  \\draw ({base:.2f},{y + 0.13:.2f}) -- ({x2:.2f},{y:.2f}) "
            f"-- ({base:.2f},{y - 0.13:.2f});",
        ]
    if k == "reversible":
        # 双向（⇌）按 fast_latex_test.tex 参考写法：四段裸 \draw 拼成，
        # 不用 -> 箭头样式——两条等长横线（间距 0.10，y=±0.05）+ 两个
        # 45° 箭头尖（偏移 ±0.10：上尖在右端右上、下尖在左端左下）。
        # 横线长度 = x2-x1（布局决定，可调）。
        h, t = 0.05, 0.10
        return [
            f"  \\draw ({x1:.2f},{y + h:.2f}) -- ({x2:.2f},{y + h:.2f})"
            f"{_node(above, 'above')};",
            f"  \\draw ({x2:.2f},{y + h:.2f}) -- ({x2 - t:.2f},{y + h + t:.2f});",
            f"  \\draw ({x1:.2f},{y - h:.2f}) -- ({x2:.2f},{y - h:.2f})"
            f"{_node(below, 'below')};",
            f"  \\draw ({x1:.2f},{y - h:.2f}) -- ({x1 + t:.2f},{y - h - t:.2f});",
        ]
    node = _node(above, "above") + _node(below, "below")
    return [
        f"  \\draw[->, {style}] ({x1:.2f},{y:.2f}) -- "
        f"({x2:.2f},{y:.2f}){node};",
    ]


# 物种系数前缀：整数（2CCO）或 n/2（1/2O2、3/2O2）；与 core.tag_validator
# 的 _parse_coeff 规则一致（渲染端独立实现，避免跨包导入）。
_SP_COEFF_RE = re.compile(r"^(-?\d+)(?:/(\d+))?")


def split_species_coeff(seg: str) -> list:
    """多物种分段 → [(coeff, bare_smiles), ...]（支持系数前缀）。

    "2CCO;1/2O2" → [(2, "CCO"), (0.5, "O2")]。非法系数组分丢弃（返回空对）；
    系数 1 省略。供 REACTION/ARROW 渲染时剥离系数、显示系数节点。
    """
    out = []
    for tok in re.split(r"[;,]", seg or ""):
        tok = tok.strip()
        if not tok:
            continue
        m = _SP_COEFF_RE.match(tok)
        if not m:
            out.append((1, tok))
            continue
        num, den = int(m.group(1)), m.group(2)
        if num == 0 or (den is not None and (int(den) != 2 or num % 2 == 0)):
            continue  # 非法系数（0 / 非 n/2）跳过，与校验层一致
        coeff = num / 2.0 if den else float(num)
        bare = tok[m.end():].strip()
        if bare:
            out.append((coeff, bare))
    return out


def bond_order_of(mol, spec: str) -> int:
    """"a-b" 键引用对应的键级（1/2/3）；非法或不存在返回 0。"""
    a, _, b = spec.partition("-")
    try:
        ia, ib = int(a), int(b)
    except ValueError:
        return 0
    if ia >= mol.GetNumAtoms() or ib >= mol.GetNumAtoms():
        return 0
    bond = mol.GetBondBetweenAtoms(ia, ib)
    return bond_type_order(bond) if bond is not None else 0


def mech_arrow_origin(mol, spec: str, shift=(0.0, 0.0),
                      lone_pair_offset: bool = True, toward=None,
                      prefer_single: bool = False, labeler=None,
                      label_gap: float = _MECH_LABEL_GAP, bend_side: float = 1.0,
                      xh_points: dict | None = None,
                      as_target: bool = False):
    """解析机理箭头端点引用为画布坐标。

    参数:
        mol: RDKit Mol（需已有 2D 坐标）。
        spec: "a"（原子 a）、"a-b"（原子 a 与 b 之间的键中点）或
            "a#k"（原子 a 的第 k 个显式 H，k 从 1 起；需 xh_points 提供坐标）。
        shift: 分子在画布上的平移量。
        lone_pair_offset: 为 True 且端点是有孤对电子的杂原子时，坐标落在
            孤对电子点上（教科书风格）；箭头终点应为 False。
        toward: 箭头另一端点的画布坐标 (x, y)，用于选择朝向目标的孤对槽位；
            同朝向槽位中优先取靠近正上方（90°）者。纯原子端点且给出
            labeler 时，还用于把端点吸附到标签边缘（见 labeler）。
        prefer_single: 为 True（鱼钩箭头）且原子有单电子时，落在单电子点上。
        labeler: 提供后，纯原子端点（非键中点、非电子点）返回元素符号中心
            （symbol_center，与孤对电子/部分电荷同源）：碳与杂原子标签目标端
            统一返回符号中心，由 mech_arrow_tikz 的 aim_end 沿切线退让到
            "标签所占位置"（边长 0.26 正方形）边缘外 label_gap，尖端指向元素符号。
        label_gap: aim_end 末端到标签正方形边缘的间距（_MECH_LABEL_GAP）。
        bend_side: 弯向（-1 向下 / +1 向上），多键端点按弯向取"靠外杠"：
            <0 取 y 最小杠（下）、>0 取 y 最大杠（上），再沿弯向外移 0.05。
        xh_points: {原子序号: [(hx, hy), ...]} 显式 H 画布坐标（局部，未加 shift）；
            spec 为 "a#k" 时必需——定位到第 k 个显式 H 节点。
        as_target: True 时本端为箭头**终点**——"a#k" 定位到 H 节点本身
            （箭头尖指向 H，而非 X—H 键中点）；False（起点）时保持断键语义
            （从 X—H 键线中点出发）。

    返回:
        (x, y, from_bond, on_electron, on_label)；spec 无效或原子越界返回 None。
        on_electron 为 True 时调用方不应再内缩起点（已在电子点上）；
        on_label 为 True 时端点已吸附到标签边缘（调用方不应再内缩）。
    """
    spec = spec.strip()
    if "#" in spec:
        # 显式 H 端点："a#k" = 原子 a 的第 k 个显式 H（k 从 1 起）。
        # 作起点（as_target=False）：该 H 是 X—H σ 键的一端，箭头从**键线中点**
        # 出发（断键语义，向下弯）。键线 = 标签边缘（label_edge_point）→ H 节点：
        # 与 a-b 键中点用修剪后键线段一致（问题 8：起点应落在 C—H 可视键中点）。
        # 作终点（as_target=True）：箭头尖直接指向 H 节点本身（如碱夺 H 的
        # 去质子箭头，靶点是 H 而非键）。
        a, _, k = spec.partition("#")
        try:
            ia, ik = int(a), int(k)
        except ValueError:
            return None
        if ia >= mol.GetNumAtoms():
            return None
        pts = (xh_points or {}).get(ia)
        if not pts or not 1 <= ik <= len(pts):
            return None
        hx, hy = pts[ik - 1]
        if as_target:
            return (hx + shift[0], hy + shift[1], False, False, True)
        sx, sy = label_edge_point(mol, ia, (hx, hy),
                                  labeler=labeler or mol_default_labeler(mol))
        mx = (sx + hx) / 2.0      # 键线中点（标签边缘 → H，局部坐标）
        my = (sy + hy) / 2.0
        return (mx + shift[0], my + shift[1], True, False, False)
    if "-" in spec:
        a, _, b = spec.partition("-")
        try:
            ia, ib = int(a), int(b)
        except ValueError:
            return None
        if ia >= mol.GetNumAtoms() or ib >= mol.GetNumAtoms():
            return None
        xa, ya = atom_pos(mol, ia)
        xb, yb = atom_pos(mol, ib)
        # 键中点基于"渲染键线"（标签留白修剪后）的可视线段，与骨架键对齐：
        # 键线两端按 label_bond_margin 修剪（如 C–Cl：C 端 CH₃=0.45、
        # Cl 端=0.30），原子坐标中点会相对视觉键偏移 0.075。
        segs = bond_segments_for(mol, ia, ib,
                                 labeler=labeler or mol_default_labeler(mol),
                                 margin_fn=label_bond_margin)
        bond = mol.GetBondBetweenAtoms(ia, ib)
        if segs and bond is not None and bond_type_order(bond) >= 2 \
                and len(segs) >= 2:
            # 多键按"靠外杠"计算：取弯向一侧（bend_side<0 向下 → y 最小杠；
            # >0 向上 → y 最大杠）的杠中点，再沿弯向（画布 y）外移 0.05。
            cands = [((s[0] + s[2]) / 2.0, (s[1] + s[3]) / 2.0) for s in segs]
            pick = (min if bend_side < 0 else max)
            mx, my = pick(cands, key=lambda p: p[1])
            my += -_ARROW_POINT_GAP if bend_side < 0 else _ARROW_POINT_GAP
        elif segs:
            x1, y1, x2, y2 = segs[0]
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        else:
            mx, my = (xa + xb) / 2.0, (ya + yb) / 2.0
        return (mx + shift[0], my + shift[1], True, False, False)
    try:
        ia = int(spec)
    except ValueError:
        return None
    if ia >= mol.GetNumAtoms():
        return None
    x, y = atom_pos(mol, ia)
    atom = mol.GetAtomWithIdx(ia)
    # 电子点分支：lone_pair_offset 且非碳原子（孤对/单电子），或 碳自由基
    # （鱼钩 prefer_single 且原子带单电子）——碳自由基的单电子点是合法
    # 鱼钩起点（B1）；碳无孤对电子，进攻箭头（非 prefer_single）对碳
    # 保持原子中心；lone_pair_offset=False（目标端）不落电子点。
    has_radical = atom.GetNumRadicalElectrons() > 0
    if (lone_pair_offset and atom.GetAtomicNum() != 6) \
            or (prefer_single and has_radical):
        if prefer_single:
            angles = single_electron_angles(mol, ia)
        else:
            angles = lone_pair_angles(mol, ia)
        is_single = prefer_single
        if not angles and not prefer_single:
            angles = single_electron_angles(mol, ia)
            is_single = True
        if angles:
            cx, cy = _dot_center(mol, ia)
            cx += shift[0]
            cy += shift[1]
            if toward is not None:
                tdir = math.degrees(math.atan2(toward[1] - cy, toward[0] - cx))
                cands = [a for a in angles if _ang_diff(a, tdir) <= 95] or angles
            else:
                cands = angles
            best = min(cands, key=lambda a: _ang_diff(a, 90.0))
            r = math.radians(best)
            # 起点 = 电子点中心（槽位方向 _LP_DIST）——gap 由 mech_arrow_tikz
            # 沿弯向（bend_side 法线方向）追加，保证 gap 方向与弧线弯向一致
            # （20260815：原在槽位方向外移 gap，方向与弯向可能不一致）。
            d = _LP_DIST
            return (cx + d * math.cos(r), cy + d * math.sin(r), False, True, False)
    if labeler is not None and toward is not None:
        lab = labeler(atom)
        if lab:
            sx, sy = symbol_center(mol, ia)
            gx, gy = sx + shift[0], sy + shift[1]
            if atom.GetAtomicNum() == 6:
                # 碳原子（CH₃/CH₂/CH）：目标端取元素符号中心（C 字形位置），
                # 由 mech_arrow_tikz 的 aim_end 沿切线退让到正方形边缘外，
                # 尖端指向原子中心（与杂原子目标端一致）。
                return (gx, gy, False, False, True)
            # 杂原子目标端返回元素符号中心（不沿入射偏移）——由
            # mech_arrow_tikz 的 aim_end 沿末端切线退让到标签外，
            # 保证切线指向元素符号中心（与碳目标端一致，Drawbacks 需求）。
            return (gx, gy, False, False, True)
    return x + shift[0], y + shift[1], False, False, False


def mech_arrow_between(fx: float, fy: float, tx: float, ty: float,
                       kind: str = "standard", from_bond: bool = False,
                       inset_start: float = 0.15, inset_end: float = 0.10,
                       bond_break: bool = False, aim_end: bool = False,
                       text_box=None, bend_side: float | None = None,
                       gap_along_bend: bool = False) -> list:
    """按教科书风格生成弯箭头：键中点出发的箭头向下弯（断键方向），
    孤对电子/原子出发的箭头向上弯（进攻方向）；弧线贴近分子，
    弯曲幅度随跨度自适应（上限 0.6）。

    bond_break（σ 断键源）/ gap_along_bend（电子点起点）的起点 gap 沿弯向
    法线方向（mech_arrow_tikz 的 gap_along_bend，bend_side 决定侧——
    "向上弯则向上 gap"，20260815 起取代原固定画布向下偏移 fy-=inset_start）；
    aim_end（字母标签目标）让终点沿末端切线退到标签正方形外 inset_end 处，
    尖端指向原子中心且不压标签。
    bend_side: 显式弯向（±1，弦法线方向）——空间感知弯向由调用方计算
    （draw_mech_arrows）；None 时按 bend 符号 + 法线 y 近似选择（旧逻辑）。
    """
    dist = math.hypot(tx - fx, ty - fy)
    if aim_end:
        mag = min(0.30 * dist + 0.15, 1.15)
    else:
        mag = min(0.22 * dist + 0.15, 0.6)
    bend = -mag if from_bond else mag
    return mech_arrow_tikz(fx, fy, tx, ty, kind, bend=bend,
                           inset_start=inset_start, inset_end=inset_end,
                           aim_end=aim_end, text_box=text_box,
                           bend_side=bend_side,
                           gap_along_bend=gap_along_bend)


def _text_extent_out(cx: float, cy: float, hw: float, hh: float,
                     px: float, py: float, dx: float, dy: float) -> float:
    """从盒内点 (px,py) 沿 (dx,dy) 方向到文本包围盒 [cx±hw]×[cy±hh] 边缘的距离。"""
    tx_ = ((cx + hw - px) / dx) if dx > 1e-9 else (
        ((cx - hw - px) / dx) if dx < -1e-9 else float("inf"))
    ty_ = ((cy + hh - py) / dy) if dy > 1e-9 else (
        ((cy - hh - py) / dy) if dy < -1e-9 else float("inf"))
    t = min(tx_, ty_)
    return t if t > 1e-9 else 0.0


def mech_arrow_tikz(fx: float, fy: float, tx: float, ty: float,
                    kind: str = "standard", bend: float = 0.5,
                    inset_start: float = 0.15, inset_end: float = 0.10,
                    aim_end: bool = False, text_box=None,
                    bend_side: float | None = None,
                    gap_along_bend: bool = False) -> list:
    r"""生成一条电子推进弯箭头的 TikZ 线条列表。

    bend 为正向上弯、为负向下弯；起点内缩 inset_start、终点内缩 inset_end，
    避免压住标签（起点已在电子点上时 inset_start 传 0）。

    参数:
        fx, fy: 起点（电子供体）坐标。
        tx, ty: 终点（电子受体）坐标。
        kind: "standard" 双电子全箭头 / "fishhook" 单电子鱼钩箭头。
        bend: 弯曲幅度（控制点到连线的垂直距离），符号决定弯向。
        inset_start / inset_end: 两端内缩距离。
        aim_end: 为 True（字母标签目标，如 C）时，先按自然弯向求控制点，
            再沿末端切线方向把终点内缩——终点坐标适配切线、尖端指向原子中心。
        text_box: (cx, cy, hw, hh) 目标原子标签的"所占位置"正方形
            （中心=符号中心、hw=hh=_LABEL_SQUARE_HALF）；aim_end 时终点沿
            切线退到正方形边缘外 inset_end（_MECH_LABEL_GAP=0.05），
            控制点随后按"起点→末端"实际箭头段重算（短箭头不重叠）。
        bend_side: 显式弯向（±1，弦法线方向；None=按 bend 符号 + 法线 y
            近似选择，20260815 起空间感知弯向由调用方计算后传入）。
        gap_along_bend: 起点 gap 沿弯向法线方向偏移（断键/电子点起点——
            "向上弯则向上 gap"），而非沿弦方向。
    """
    dx, dy = tx - fx, ty - fy
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    if gap_along_bend and bend_side is not None:
        # 起点 gap 沿弯向法线方向（断键/孤对电子点起点）
        sx, sy = fx + px * bend_side * inset_start, fy + py * bend_side * inset_start
    else:
        sx, sy = fx + ux * inset_start, fy + uy * inset_start
    ex, ey = tx, ty
    if not aim_end:
        ex, ey = tx - ux * inset_end, ty - uy * inset_end
    mid_x, mid_y = (sx + ex) / 2.0, (sy + ey) / 2.0
    mag = abs(bend)
    if bend_side is not None:
        mx, my = mid_x + px * bend_side * mag, mid_y + py * bend_side * mag
    else:
        mx_a, my_a = mid_x + px * mag, mid_y + py * mag
        mx_b, my_b = mid_x - px * mag, mid_y - py * mag
        if (bend >= 0) == (my_a >= my_b):
            mx, my = mx_a, my_a
        else:
            mx, my = mx_b, my_b
    if aim_end:
        ddx, ddy = tx - mx, ty - my
        dl = math.hypot(ddx, ddy) or 1.0
        ndx, ndy = ddx / dl, ddy / dl
        ins = inset_end
        if text_box is not None:
            bcx, bcy, bhw, bhh = text_box
            ins += _text_extent_out(bcx, bcy, bhw, bhh, tx, ty, -ndx, -ndy)
        # 末端不越过起点（否则切线反向、箭头退化为点/指向外面）：
        # 退让距离不超过"起点沿切线方向到目标的投影距离"的 85%——
        # 保证末端落在起点与目标之间，切线（控制点→末端）指向目标标签。
        proj = (fx - tx) * ndx + (fy - ty) * ndy
        ins = min(ins, abs(proj) * 0.85)
        ex, ey = tx - ins * ndx, ty - ins * ndy
        # 控制点基于"起点→末端"（退让后的实际箭头段）重算：短箭头（断键）
        # 时避免控制点悬在目标侧导致尖端与杆重叠；弯曲幅度随实际段长自适应。
        d2x, d2y = ex - sx, ey - sy
        l2 = math.hypot(d2x, d2y) or 1.0
        p2x, p2y = -d2y / l2, d2x / l2
        mid2_x, mid2_y = (sx + ex) / 2.0, (sy + ey) / 2.0
        mag2 = min(0.30 * l2 + 0.15, 1.15)
        mx_a, my_a = mid2_x + p2x * mag2, mid2_y + p2y * mag2
        mx_b, my_b = mid2_x - p2x * mag2, mid2_y - p2y * mag2
        if bend_side is not None:
            # 空间感知弯向：重算后仍保持 bend_side 侧（切线校验的翻转
            # 仅作为最后手段，语义由空间感知决定）
            mx, my = (mx_a, my_a) if bend_side >= 0 else (mx_b, my_b)
        else:
            if (bend >= 0) == (my_a >= my_b):
                mx, my = mx_a, my_a
            else:
                mx, my = mx_b, my_b
        # 切线校验：切线（控制点→末端）必须指向标签方向——标签视为
        # "中心=符号中心、边长 0.26 的正方形"，切线方向应穿过它。
        # 若切线方向与"末端→符号中心"方向夹角 > 90°（指向标签外），
        # 翻转控制点到连线另一侧（bend 弯向修正）。
        tdx, tdy = ex - mx, ey - my
        gdx, gdy = tx - ex, ty - ey
        if tdx * gdx + tdy * gdy < 0:
            mx, my = (mx_b, my_b) if (mx, my) == (mx_a, my_a) else (mx_a, my_a)

    lines = []
    if kind == "fishhook":
        lines.append(
            f"  \\draw[thick, red] ({sx:.2f},{sy:.2f}) "
            f".. controls ({mx:.2f},{my:.2f}) .. ({ex:.2f},{ey:.2f});"
        )
        incoming = math.atan2(ey - my, ex - mx)
        barb_ang = incoming + math.pi + math.radians(25)
        blen = 0.18
        bx = ex + blen * math.cos(barb_ang)
        by = ey + blen * math.sin(barb_ang)
        lines.append(f"  \\draw[thick, red] ({ex:.2f},{ey:.2f}) -- ({bx:.2f},{by:.2f});")
    else:
        lines.append(
            f"  \\draw[->, thick, red] ({sx:.2f},{sy:.2f}) "
            f".. controls ({mx:.2f},{my:.2f}) .. ({ex:.2f},{ey:.2f});"
        )
    return lines


def tikz_draw_line(x1: float, y1: float, x2: float, y2: float, style: str = "") -> str:
    r"""生成一条 \draw 命令。"""
    cmd = "  \\draw"
    if style:
        cmd += f"[{style}]"
    cmd += f" {fmt_coord(x1, y1)} -- {fmt_coord(x2, y2)};"
    return cmd
