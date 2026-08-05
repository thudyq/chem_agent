# -*- coding: utf-8 -*-
"""renderers/mol_primitives.py — RDKit 分子骨架绘制的共享工具。

把 mechanism / lewis / stereo / charge / hbond 中重复的原子标签、
2D 坐标计算、键线绘制逻辑抽取到这里，避免复制粘贴。
"""

import math
import re


def _only_h_neighbors(atom) -> bool:
    """原子是否只连氢（无重原子邻居）——如孤立水分子 O。

    用于标签方向特例：水的 O 只连 2 个 H，化学上写 H₂O（H 在前，
    氢化物惯例），而 OH₂ 是羟基 -OH 的写法（仅 O 连 C 时正确）。
    """
    return all(nbr.GetAtomicNum() == 1 for nbr in atom.GetNeighbors())


def atom_label(atom, explicit_hs: int = 0, flip: bool = False) -> str | None:
    """生成非隐式碳原子的标签（如 OH、NH₂、Cl、$^{+}$ 等）。

    纯碳原子（原子序 6、形式电荷 0）返回 None，表示不显示标签（键线式）。
    explicit_hs：已显式画出的 H 数（[XH]/氢键给体），从标签 H 计数中
    扣除，保证"标签 H + 画出 H"总数正确（如 OH 画出 H 后标签为 O）。
    水分子特例：O 只连 H 时写 H₂O（H 在前）。
    flip=True：键端在标签右侧时元素符号右移（OH→HO、NH₂→H₂N，
    使键连接的原子靠近键端，Drawbacks 第 6 条）。
    """
    z = atom.GetAtomicNum()
    if z == 6 and atom.GetFormalCharge() == 0:
        return None

    sym = atom.GetSymbol()
    sym = sym[0].upper() + sym[1:]
    h = max(0, atom.GetTotalNumHs() - explicit_hs)
    if sym == "O" and h == 2 and _only_h_neighbors(atom):
        parts = f"H$_{{{h}}}$O" if h > 1 else "HO"
    elif flip:
        parts = (f"H$_{{{h}}}$" if h > 1 else ("H" if h == 1 else "")) + sym
    else:
        parts = sym
        if h == 1:
            parts += "H"
        elif h > 1:
            parts += f"H$_{{{h}}}$"

    fc = atom.GetFormalCharge()
    if fc:
        num = str(abs(fc)) if abs(fc) > 1 else ""
        sign = "+" if fc > 0 else "-"
        parts += f"$^{{{num}{sign}}}$"

    return parts


def condensed_atom_label(atom) -> str | None:
    """结构简式标签：非环碳原子同样写出（CH₃/CH₂/CH/C），环上碳原子
    保持键线式（返回 None）。

    机理场景（COMPOSITE 容器内）使用，使小分子呈现
    教科书式的 H₃C—Cl 风格而非键线式。
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
    fc = atom.GetFormalCharge()
    if fc:
        num = str(abs(fc)) if abs(fc) > 1 else ""
        sign = "+" if fc > 0 else "-"
        parts += f"$^{{{num}{sign}}}$"
    return parts


def label_plain_len(label: str) -> int:
    """标签去掉 LaTeX 排版符号后的可视字符数，用于估算键线留白宽度。"""
    return len(re.sub(r"[$_{}^\\]", "", label))


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


def place_explicit_hs(mol, idx: int, count: int = 1,
                      toward: tuple[float, float] | None = None,
                      h_len: float = 0.75) -> list:
    """原子 idx 上 count 个显式 H 的位置（互不重叠的空档方向扇形分配）。

    toward 非空时首个 H 优先沿该方向（氢键 X—H···Y 直线）；其余按
    最大空档角平分线依次分配，同一空档内多根 H 扇形展开 ±20°。
    """
    x, y = atom_pos(mol, idx)
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

    return [(x + h_len * math.cos(math.radians(a)),
             y + h_len * math.sin(math.radians(a))) for a in angles[:count]]


def place_donor_h(mol, x_idx: int, y_pos: tuple[float, float],
                  h_len: float = 0.75) -> tuple[float, float]:
    """给体 X 的显式 H 位置（规范：X—H 用实线画出）。

    优先取 X→Y 方向（X—H···Y 尽量呈直线）；该方向与已有键过近（<30°）
    时改取最大空档的角平分线，避免与骨架重叠。
    """
    return place_explicit_hs(mol, x_idx, 1, toward=y_pos, h_len=h_len)[0]


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
_ARROW_LABEL_GAP = 0.05   # 机理箭头端点与"字母标签"（C/Cl 等文字）的空隙
_ARROW_POINT_GAP = 0.05   # 机理箭头端点与"点/线"（孤对电子点、断键键线）的空隙
_LABEL_TEXT_HALF_H = 0.15 # 标签文字半高估计（含上下标，垂直吸附基准）
_ORTHO = [90.0, 180.0, 270.0, 0.0]      # 正交槽位（优先）
_DIAG = [45.0, 135.0, 225.0, 315.0]     # 斜向槽位（正交占满时兜底）
_CHAR_HALF_W = 0.13       # 标签单字符半宽估计（用于元素符号中心修正）


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


def _bond_blocks(mol, idx: int) -> list:
    """规范定义的 block：直接相连的化学键和原子方向 + 标签氢方向（右侧）。

    电荷不是 block（规范仅要求电荷与孤对电子不重叠），单独作为避让约束。
    """
    blocked = _bond_angles(mol, idx)
    if _implicit_shown_hs(mol.GetAtomWithIdx(idx)) > 0:
        blocked.append(0.0)
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


def _separate_cardinals(count: int, blocked: list, taken: list) -> list:
    """核心原则兜底：正交四向优先、斜向补充，避开阻挡与已占槽位（>30°）。"""
    result = list(taken)
    for cand in (90.0, 180.0, 270.0, 0.0, 45.0, 135.0, 225.0, 315.0):
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
    avoid = [_charge_angle(mol, idx)] if atom.GetFormalCharge() != 0 else []
    angles = _place_pairs(pairs, _bond_blocks(mol, idx))
    if avoid:
        angles = [_nudge_from_avoid(a, avoid) for a in angles]
    return angles


def single_electron_angles(mol, idx: int) -> list:
    """单电子（自由基）的放置角度，规则同孤对电子并避开已占电子点。"""
    _, singles = lone_pair_count(mol.GetAtomWithIdx(idx))
    if singles == 0:
        return []
    taken = lone_pair_angles(mol, idx)
    blocked = _blocked_angles(mol, idx) + list(taken)
    return _separate_cardinals(singles, blocked, taken)[:singles]


def _dot_center(mol, idx: int, explicit_hs: int = 0) -> tuple[float, float]:
    """孤对电子点的环绕中心：元素符号在标签内的估计位置。

    标签后缀（H 计数部分）使元素符号偏离标签中心向左，
    按后缀可视宽度的一半左移修正（如 OH 的点绕 O 而非绕 OH 整体）。
    explicit_hs 已显式画出的 H 会同步缩小后缀宽度。
    """
    atom = mol.GetAtomWithIdx(idx)
    x, y = atom_pos(mol, idx)
    lab = atom_main_label(atom, explicit_hs)
    if lab:
        sym = atom.GetSymbol()
        sym = sym[0].upper() + sym[1:]
        plain = re.sub(r"[$_{}^\\]", "", lab)
        if plain.startswith(sym):
            x -= _CHAR_HALF_W * (len(plain) - len(sym))
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
    if atom.GetAtomicNum() == 6:
        sym = "C"
    else:
        sym = atom.GetSymbol()
        sym = sym[0].upper() + sym[1:]
    h = max(0, atom.GetTotalNumHs() - explicit_hs)
    # 水分子特例：O 只连 H 时写 H₂O（H 在前），否则 OH 是羟基写法
    if sym == "O" and h == 2 and _only_h_neighbors(atom):
        parts = f"H$_{{{h}}}$O" if h > 1 else "HO"
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
_CHARGE_SCALE = 0.5        # 电荷圈缩放（为默认大小的一半）


def _charge_angle(mol, idx: int) -> float:
    """电荷圈方位角：右侧有标签氢阻碍且左侧无阻碍时在左上（135°），
    否则在右上（45°）（左右都有阻碍时保持右上）。"""
    if _implicit_shown_hs(mol.GetAtomWithIdx(idx)) == 0:
        return 45.0
    left_blocked = any(
        _ang_diff(a, 180.0) <= 45.0 for a in _bond_angles(mol, idx)
    )
    return 45.0 if left_blocked else 135.0


def charge_tikz(mol, idx: int, shift=(0.0, 0.0), explicit_hs: int = 0) -> str | None:
    r"""圆圈电荷节点（右上/左上角，draw circle，半尺寸）；无电荷返回 None。"""
    text = atom_charge_label(mol.GetAtomWithIdx(idx))
    if text is None:
        return None
    cx, cy = _dot_center(mol, idx, explicit_hs)
    r = math.radians(_charge_angle(mol, idx))
    x = cx + shift[0] + _CHARGE_POS_DIST * math.cos(r)
    y = cy + shift[1] + _CHARGE_POS_DIST * math.sin(r)
    return (f"\\node[draw, circle, inner sep=0.6pt, font=\\scriptsize, "
            f"scale={_CHARGE_SCALE}] at ({x:.2f},{y:.2f}) {{{text}}};")


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


def mol_visual_bbox(mol, labeler=condensed_atom_label,
                    include_lone_pairs: bool = True):
    """分子视觉包围盒 (min_x, min_y, max_x, max_y)。

    在原子坐标基础上计入标签半径与孤对电子点的外延，
    供布局间距计算使用，避免相邻组件重叠。
    """
    xs, ys = [], []
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
            r = math.radians(_charge_angle(mol, atom.GetIdx()))
            xs.append(x + _CHARGE_POS_DIST * math.cos(r) + 0.1 * math.cos(r))
            dy = _CHARGE_POS_DIST * math.sin(r) + 0.1
            ys.append(y + dy)
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


def prepare_mol(smiles: str, *, add_hs: bool = False, kekulize: bool = False,
                use_prepare: bool = True, allow_aromatic: bool = True):
    """SMILES → RDKit Mol：解析、可选加氢/Kekulize、计算 2D 坐标。

    参数:
        smiles: 输入 SMILES。
        add_hs: 是否调用 AddHs（Lewis 结构需要显示所有 H）。
        kekulize: 是否 Kekulize（Lewis 需要明确单双键）。
        use_prepare: 是否优先用 rdMolDraw2D.PrepareMolForDrawing；
                     为 False 时直接用 AllChem.Compute2DCoords。
        allow_aromatic: 为 False 时跳过芳香化判定与再 Kekulé 化，
                     保留输入的显式键级——共振极限式（如两个 Kekulé 苯）
                     必须如此，否则会被统一芳香化成同一结构。

    返回:
        RDKit Mol 对象；解析失败返回 None。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return None

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

    mol = _remove_free_hydrogens(mol)

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

    if allow_aromatic:
        _regularize_kekule(mol)
    return mol


def _remove_free_hydrogens(mol):
    """移除游离氢原子（degree=0，如 SMILES 中独立的 [H]/[H+] 质子组分）。

    RDKit PrepareMolForDrawing 内部 RemoveHs 对无邻居的氢打
    "not removing hydrogen atom without neighbors" 警告（C4）；游离氢
    不属于任何化学键（LLM 违规输出质子的产物），提前移除让图干净并消除警告。
    正常氢（如 [H]O[H] 的 H，有 O 邻居）不受影响。
    """
    from rdkit import Chem
    free = [a.GetIdx() for a in mol.GetAtoms()
            if a.GetAtomicNum() == 1 and a.GetDegree() == 0]
    if not free:
        return mol
    rw = Chem.RWMol(mol)
    for idx in sorted(free, reverse=True):
        rw.RemoveAtom(idx)
    return rw.GetMol()


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
                  labeler=atom_label, margin_fn=None):
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
        labeler: 原子标签函数（默认 atom_label；机理场景用 condensed_atom_label）。
        margin_fn: 按标签文本计算留白距离的函数；缺省统一用 label_margin。

    环内双键的平行线朝环质心偏移：双键为内缩短线（端点在中心→顶点
    射线上，内缩量 0.18×键长）；三键保持等长双侧平行线。
    """
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


def format_chem_text(text: str) -> str:
    """化学文本自动排版：字母/括号后的数字转下标，尾部电荷转上标。

    先剥离尾部电荷（含电荷数），再对余下文本转下标，保证「SO42-」
    中 4 为下标、2- 为上标。已含 $（已手工排版）或为空时原样返回。
    Unicode 上下标（H₂SO₄、H⁺、SO₄²⁻、Ca²⁺ 等）先转 LaTeX 命令。

    示例：H2SO4 → H$_2$SO$_4$；CH3Cl → CH$_3$Cl；OH- → OH$^{-}$；
    NH4+ → NH$_4$$^{+}$；SO42- → SO$_4$$^{2-}$；
    H₂SO₄ → H$_{2}$SO$_{4}$；H⁺ → H$^{+}$；Ca²⁺ → Ca$^{2+}$。
    """
    if not text or "$" in text:
        return text
    text = _convert_unicode_scripts(text)
    charge = ""
    m = _CHARGE_TAIL_RE.search(text)
    if m:
        charge = f"$^{{{m.group(1) or ''}{m.group(2)}}}$"
        text = text[: m.start()]
    return _SUBSCRIPT_RE.sub(r"\1$_\2$", text) + charge


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
                      label_gap: float = 0.11, bend_side: float = 1.0):
    """解析机理箭头端点引用为画布坐标。

    参数:
        mol: RDKit Mol（需已有 2D 坐标）。
        spec: "a"（原子 a）或 "a-b"（原子 a 与 b 之间的键中点）。
        shift: 分子在画布上的平移量。
        lone_pair_offset: 为 True 且端点是有孤对电子的杂原子时，坐标落在
            孤对电子点上（教科书风格）；箭头终点应为 False。
        toward: 箭头另一端点的画布坐标 (x, y)，用于选择朝向目标的孤对槽位；
            同朝向槽位中优先取靠近正上方（90°）者。纯原子端点且给出
            labeler 时，还用于把端点吸附到标签边缘（见 labeler）。
        prefer_single: 为 True（鱼钩箭头）且原子有单电子时，落在单电子点上。
        labeler: 提供后，纯原子端点（非键中点、非电子点）吸附到标签边缘，
            复用元素符号定位逻辑（symbol_center，与孤对电子/部分电荷同源）：
            碳原子标签（CH₃/CH₂/CH）目标端返回元素符号中心（C 字形位置），
            由 mech_arrow_tikz 沿末端切线内缩 _ARROW_LABEL_GAP，使终点坐标
            适配切线、尖端指向原子中心；杂原子标签沿入射方向吸附在标签
            文字前的空隙（margin - label_gap）。
        label_gap: 杂原子沿向吸附相对键线留白（label_bond_margin）的内收量。
        bend_side: 保留参数（兼容旧调用），碳标签已改为切线内缩定位，不再使用。

    返回:
        (x, y, from_bond, on_electron, on_label)；spec 无效或原子越界返回 None。
        on_electron 为 True 时调用方不应再内缩起点（已在电子点上）；
        on_label 为 True 时端点已吸附到标签边缘（调用方不应再内缩）。
    """
    spec = spec.strip()
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
        mx, my = (xa + xb) / 2.0, (ya + yb) / 2.0
        bond = mol.GetBondBetweenAtoms(ia, ib)
        if bond is not None and bond_type_order(bond) >= 2:
            # π 键电子云位于双键两杠之间：沿垂直方向偏移 bond_gap/2
            dx, dy = xb - xa, yb - ya
            L = math.hypot(dx, dy) or 1.0
            ux, uy = dx / L, dy / L
            px, py = -uy, ux
            if bond.IsInRing():
                # 环内双键的第二条平行线朝向环质心（与 bond_segments 同侧）
                try:
                    rings = mol.GetRingInfo().AtomRings()
                except Exception:
                    rings = []
                for ring in rings:
                    if ia in ring and ib in ring:
                        cxs = [atom_pos(mol, i)[0] for i in ring]
                        cys = [atom_pos(mol, i)[1] for i in ring]
                        cx, cy = sum(cxs) / len(cxs), sum(cys) / len(cys)
                        if (cx - mx) * px + (cy - my) * py < 0:
                            px, py = -px, -py
                        break
            mx += px * _BOND_GAP / 2
            my += py * _BOND_GAP / 2
        return (mx + shift[0], my + shift[1], True, False, False)
    try:
        ia = int(spec)
    except ValueError:
        return None
    if ia >= mol.GetNumAtoms():
        return None
    x, y = atom_pos(mol, ia)
    atom = mol.GetAtomWithIdx(ia)
    if lone_pair_offset and atom.GetAtomicNum() != 6:
        if prefer_single:
            angles = single_electron_angles(mol, ia)
        else:
            angles = lone_pair_angles(mol, ia)
        if not angles and not prefer_single:
            angles = single_electron_angles(mol, ia)
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
            # 箭头起点沿槽位方向再外移 _ARROW_POINT_GAP，使其不与电子点完全重合
            # （顶部槽位时即"起点比电子点高 0.05"）。
            d = _LP_DIST + _ARROW_POINT_GAP
            return (cx + d * math.cos(r), cy + d * math.sin(r), False, True, False)
    if labeler is not None and toward is not None:
        lab = labeler(atom)
        if lab:
            sx, sy = symbol_center(mol, ia)
            gx, gy = sx + shift[0], sy + shift[1]
            if atom.GetAtomicNum() == 6:
                # 碳原子（CH₃/CH₂/CH）：目标端取元素符号中心（C 字形位置），
                # 由 mech_arrow_tikz 沿末端切线内缩 _ARROW_LABEL_GAP，
                # 让终点坐标适配切线方向、尖端指向原子中心（不强行改切线）。
                return (gx, gy, False, False, True)
            tx, ty = toward
            dx, dy = tx - gx, ty - gy
            L = math.hypot(dx, dy) or 1.0
            m = label_bond_margin(lab) - label_gap
            return (gx + dx / L * m,
                    gy + dy / L * m, False, False, True)
    return x + shift[0], y + shift[1], False, False, False


def mech_arrow_between(fx: float, fy: float, tx: float, ty: float,
                       kind: str = "standard", from_bond: bool = False,
                       inset_start: float = 0.15, inset_end: float = 0.10,
                       bond_break: bool = False, aim_end: bool = False,
                       text_box=None) -> list:
    """按教科书风格生成弯箭头：键中点出发的箭头向下弯（断键方向），
    孤对电子/原子出发的箭头向上弯（进攻方向）；弧线贴近分子，
    弯曲幅度随跨度自适应（上限 0.6）。

    bond_break（σ 断键源）的起点 inset 改为纵坐标向下偏移 inset_start
    （贴近键线下方），而非沿箭头方向内缩；aim_end（字母标签目标）让
    终点沿末端切线退到 text_box 外 inset_end 处，尖端指向原子中心且不压标签。"""
    if bond_break:
        fy -= inset_start
        inset_start = 0.0
    dist = math.hypot(tx - fx, ty - fy)
    if aim_end:
        mag = min(0.30 * dist + 0.15, 1.15)
    else:
        mag = min(0.22 * dist + 0.15, 0.6)
    bend = -mag if from_bond else mag
    return mech_arrow_tikz(fx, fy, tx, ty, kind, bend=bend,
                           inset_start=inset_start, inset_end=inset_end,
                           aim_end=aim_end, text_box=text_box)


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
                    aim_end: bool = False, text_box=None) -> list:
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
        text_box: (cx, cy, hw, hh) 目标原子标签的文本包围盒；aim_end 时
            终点在切线方向上退到盒外 inset_end（0.10）处，避免压住标签文字。
    """
    dx, dy = tx - fx, ty - fy
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    sx, sy = fx + ux * inset_start, fy + uy * inset_start
    ex, ey = tx, ty
    if not aim_end:
        ex, ey = tx - ux * inset_end, ty - uy * inset_end
    px, py = -uy, ux
    mid_x, mid_y = (sx + ex) / 2.0, (sy + ey) / 2.0
    mag = abs(bend)
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
        ex, ey = tx - ins * ndx, ty - ins * ndy

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
