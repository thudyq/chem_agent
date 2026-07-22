# -*- coding: utf-8 -*-
"""renderers/mol_primitives.py — RDKit 分子骨架绘制的共享工具。

把 mechanism / lewis / stereo / charge / hbond 中重复的原子标签、
2D 坐标计算、键线绘制逻辑抽取到这里，避免复制粘贴。
"""

import math
import re


def atom_label(atom) -> str | None:
    """生成非隐式碳原子的标签（如 OH、NH₂、Cl、$^{+}$ 等）。

    纯碳原子（原子序 6、形式电荷 0）返回 None，表示不显示标签。
    """
    z = atom.GetAtomicNum()
    if z == 6 and atom.GetFormalCharge() == 0:
        return None

    sym = atom.GetSymbol()
    sym = sym[0].upper() + sym[1:]
    h = atom.GetTotalNumHs()
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

    机理场景（MECH / REACTIONMECH / COMPOSITE）使用，使小分子呈现
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


def hbond_line_tikz(fx: float, fy: float, tx: float, ty: float,
                    margin: float = 0.25) -> str:
    r"""生成一条氢键虚线（teal dashed，两端内缩避免压住原子标签）。"""
    dx, dy = tx - fx, ty - fy
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    x1, y1 = fx + ux * margin, fy + uy * margin
    x2, y2 = tx - ux * margin, ty - uy * margin
    return f"\\draw[dashed, teal, thick] ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});"


_VALENCE_ELECTRONS = {1: 1, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7,
                      14: 4, 15: 5, 16: 6, 17: 7, 35: 7, 53: 7}

_LP_DIST = 0.30           # 孤对电子点到原子的固定距离
_ORTHO = [90.0, 180.0, 270.0, 0.0]      # 正交槽位（优先）
_DIAG = [45.0, 135.0, 225.0, 315.0]     # 斜向槽位（正交占满时兜底）
_CHAR_HALF_W = 0.13       # 标签单字符半宽估计（用于元素符号中心修正）


def lone_pair_count(atom) -> tuple[int, int]:
    """返回 (孤对电子对数, 单电子数)。

    非键电子数 = 价电子 - 键级和 - 形式电荷 - 自由基电子数 - 隐含氢数；
    不在表中的元素（金属等）返回 (0, 0)。
    """
    ve = _VALENCE_ELECTRONS.get(atom.GetAtomicNum())
    if ve is None:
        return 0, 0
    bonds = sum(int(round(b.GetBondTypeAsDouble())) for b in atom.GetBonds())
    radicals = atom.GetNumRadicalElectrons()
    nonbonding = max(0, ve - bonds - atom.GetFormalCharge() - radicals
                     - atom.GetNumImplicitHs())
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


def _blocked_angles(mol, idx: int) -> list:
    """孤对电子的阻挡方向：显式键方向 + 标签右侧（隐含氢写在元素符号右侧）。"""
    blocked = _bond_angles(mol, idx)
    if mol.GetAtomWithIdx(idx).GetNumImplicitHs() > 0:
        blocked.append(0.0)
    return blocked


def _dot_center(mol, idx: int) -> tuple[float, float]:
    """孤对电子点的环绕中心：元素符号在标签内的估计位置。

    标签后缀（H、电荷）使元素符号偏离标签中心向左，
    按后缀可视宽度的一半左移修正（如 OH 的点绕 O 而非绕 OH 整体）。
    """
    atom = mol.GetAtomWithIdx(idx)
    x, y = atom_pos(mol, idx)
    lab = condensed_atom_label(atom)
    if lab:
        sym = atom.GetSymbol()
        sym = sym[0].upper() + sym[1:]
        plain = re.sub(r"[$_{}^\\]", "", lab)
        if plain.startswith(sym):
            x -= _CHAR_HALF_W * (len(plain) - len(sym))
    return x, y


def _pick_angles(blocked: list, taken: list, count: int) -> list:
    """正交优先、斜向兜底，从空槽位中选 count 个角度（避开阻挡与已占槽位）。"""
    result = list(taken)
    for pool in (_ORTHO, _DIAG):
        for ang in pool:
            if len(result) >= count + len(taken):
                break
            if (all(_ang_diff(ang, b) > 30 for b in blocked)
                    and all(_ang_diff(ang, t) > 30 for t in result)):
                result.append(ang)
        if len(result) >= count + len(taken):
            break
    return result[len(taken):]


def lone_pair_angles(mol, idx: int) -> list:
    """孤对电子的放置角度：正交优先，避开键与标签氢的方向。"""
    pairs, _ = lone_pair_count(mol.GetAtomWithIdx(idx))
    if pairs == 0:
        return []
    return _pick_angles(_blocked_angles(mol, idx), [], pairs)


def single_electron_angles(mol, idx: int) -> list:
    """单电子（自由基）的放置角度，避开键、标签氢与孤对电子。"""
    _, singles = lone_pair_count(mol.GetAtomWithIdx(idx))
    if singles == 0:
        return []
    return _pick_angles(_blocked_angles(mol, idx),
                        lone_pair_angles(mol, idx), singles)


def lone_pair_dot_groups(mol, idx: int, shift=(0.0, 0.0)):
    """孤对电子点的画布坐标。

    点以元素符号为中心（_dot_center），距离固定 _LP_DIST。
    返回:
        (groups, singles)：groups 为每对电子的两个点坐标列表
        [((x1,y1),(x2,y2)), ...]，singles 为单电子点坐标列表 [(x,y), ...]。
    """
    atom = mol.GetAtomWithIdx(idx)
    pairs, _ = lone_pair_count(atom)
    cx0, cy0 = _dot_center(mol, idx)
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


def lone_pair_tikz(mol, idx: int, shift=(0.0, 0.0)) -> list:
    r"""生成孤对电子点的 \fill 圆点线条列表（裸行，缩进由调用方决定）。"""
    groups, singles = lone_pair_dot_groups(mol, idx, shift)
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
            n = label_plain_len(lab)
            hw = max(0.18, 0.13 * n)
            hh = 0.18
        xs += [x - hw, x + hw]
        ys += [y - hh, y + hh]
        if include_lone_pairs:
            groups, singles = lone_pair_dot_groups(mol, atom.GetIdx())
            for (x1, y1), (x2, y2) in groups:
                xs += [x1, x2]
                ys += [y1, y2]
            for x1, y1 in singles:
                xs.append(x1)
                ys.append(y1)
    return min(xs), min(ys), max(xs), max(ys)


def prepare_mol(smiles: str, *, add_hs: bool = False, kekulize: bool = False, use_prepare: bool = True):
    """SMILES → RDKit Mol：解析、可选加氢/Kekulize、计算 2D 坐标。

    参数:
        smiles: 输入 SMILES。
        add_hs: 是否调用 AddHs（Lewis 结构需要显示所有 H）。
        kekulize: 是否 Kekulize（Lewis 需要明确单双键）。
        use_prepare: 是否优先用 rdMolDraw2D.PrepareMolForDrawing；
                     为 False 时直接用 AllChem.Compute2DCoords。

    返回:
        RDKit Mol 对象；解析失败返回 None。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles) if smiles else None
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

    return mol


def atom_pos(mol, idx: int) -> tuple[float, float]:
    """返回原子 idx 的 2D 坐标 (x, y)。"""
    conf = mol.GetConformer()
    p = conf.GetAtomPosition(idx)
    return p.x, p.y


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
    """
    def _margin(a):
        lab = labeler(a)
        if not lab:
            return 0.0
        if margin_fn is not None:
            return margin_fn(lab)
        return label_margin

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
            # 双键：两条线；三键：三条线
            segs = [(x1, y1, x2, y2)]
            segs.append((x1 + px * bond_gap, y1 + py * bond_gap,
                         x2 + px * bond_gap, y2 + py * bond_gap))
            if order == 3:
                segs.append((x1 - px * bond_gap, y1 - py * bond_gap,
                             x2 - px * bond_gap, y2 - py * bond_gap))
            segments.append(segs)

    return segments


def fmt_coord(x: float, y: float) -> str:
    """把坐标格式化为 TikZ 常用的两位小数字符串。"""
    return f"({x:.2f},{y:.2f})"


_SUBSCRIPT_RE = re.compile(r"([A-Za-z\)])(\d+)")
_CHARGE_TAIL_RE = re.compile(r"(?:(?<![A-Za-z\)])(\d+))?([+-])$")


def format_chem_text(text: str) -> str:
    """化学文本自动排版：字母/括号后的数字转下标，尾部电荷转上标。

    先剥离尾部电荷（含电荷数），再对余下文本转下标，保证「SO42-」
    中 4 为下标、2- 为上标。已含 $（已手工排版）或为空时原样返回。

    示例：H2SO4 → H$_2$SO$_4$；CH3Cl → CH$_3$Cl；OH- → OH$^{-}$；
    NH4+ → NH$_4$$^{+}$；SO42- → SO$_4$$^{2-}$。
    """
    if not text or "$" in text:
        return text
    charge = ""
    m = _CHARGE_TAIL_RE.search(text)
    if m:
        charge = f"$^{{{m.group(1) or ''}{m.group(2)}}}$"
        text = text[: m.start()]
    return _SUBSCRIPT_RE.sub(r"\1$_\2$", text) + charge


def mech_arrow_origin(mol, spec: str, shift=(0.0, 0.0),
                      lone_pair_offset: bool = True, toward=None,
                      prefer_single: bool = False):
    """解析机理箭头端点引用为画布坐标。

    参数:
        mol: RDKit Mol（需已有 2D 坐标）。
        spec: "a"（原子 a）或 "a-b"（原子 a 与 b 之间的键中点）。
        shift: 分子在画布上的平移量。
        lone_pair_offset: 为 True 且端点是有孤对电子的杂原子时，坐标落在
            孤对电子点上（教科书风格）；箭头终点应为 False。
        toward: 箭头另一端点的画布坐标 (x, y)，用于选择朝向目标的孤对槽位；
            同朝向槽位中优先取靠近正上方（90°）者。
        prefer_single: 为 True（鱼钩箭头）且原子有单电子时，落在单电子点上。

    返回:
        (x, y, from_bond, on_electron)；spec 无效或原子越界返回 None。
        on_electron 为 True 时调用方不应再内缩起点（已在电子点上）。
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
        return (xa + xb) / 2.0 + shift[0], (ya + yb) / 2.0 + shift[1], True, False
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
            return (cx + _LP_DIST * math.cos(r),
                    cy + _LP_DIST * math.sin(r), False, True)
    return x + shift[0], y + shift[1], False, False


def mech_arrow_between(fx: float, fy: float, tx: float, ty: float,
                       kind: str = "standard", from_bond: bool = False,
                       inset_start: float = 0.15, inset_end: float = 0.10) -> list:
    """按教科书风格生成弯箭头：键中点出发的箭头向下弯（断键方向），
    孤对电子/原子出发的箭头向上弯（进攻方向）；弧线贴近分子，
    弯曲幅度随跨度自适应（上限 0.6）。"""
    dist = math.hypot(tx - fx, ty - fy)
    mag = min(0.22 * dist + 0.15, 0.6)
    bend = -mag if from_bond else mag
    return mech_arrow_tikz(fx, fy, tx, ty, kind, bend=bend,
                           inset_start=inset_start, inset_end=inset_end)


def mech_arrow_tikz(fx: float, fy: float, tx: float, ty: float,
                    kind: str = "standard", bend: float = 0.5,
                    inset_start: float = 0.15, inset_end: float = 0.10) -> list:
    r"""生成一条电子推进弯箭头的 TikZ 线条列表。

    bend 为正向上弯、为负向下弯；起点内缩 inset_start、终点内缩 inset_end，
    避免压住标签（起点已在电子点上时 inset_start 传 0）。

    参数:
        fx, fy: 起点（电子供体）坐标。
        tx, ty: 终点（电子受体）坐标。
        kind: "standard" 双电子全箭头 / "fishhook" 单电子鱼钩箭头。
        bend: 弯曲幅度（控制点到连线的垂直距离），符号决定弯向。
        inset_start / inset_end: 两端内缩距离。
    """
    dx, dy = tx - fx, ty - fy
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    sx, sy = fx + ux * inset_start, fy + uy * inset_start
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
