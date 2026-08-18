# -*- coding: utf-8 -*-
r"""renderers/composite.py — [COMPOSITE] 容器式复合标记渲染器。

LLM 在容器内显式列出结构组件、连接符与机理箭头，渲染器只负责把这些组件
放进统一 TikZ 坐标系组合绘制，实现“LLM 组装组件、渲染器组合绘制”的架构。

容器语法：
    [COMPOSITE:reaction_mech]
    [STRUCT:CCl,label=CH3Cl]          （id 省略，自动编号 r0）
    [PLUS]
    [STRUCT:[OH-],id=nu,label=OH-]    （显式 id）
    [RXNARROW]                         （主反应箭头，兼作反应物/产物分界）
    [STRUCT:CO,label=CH3OH]
    [PLUS]
    [STRUCT:[Cl-],label=Cl-]
    [MECHARROW:nu:0>r0:0]              （孤对电子进攻箭头，双电子；>> 为鱼钩）
    [MECHARROW:r0:0-1>r0:1]            （σ 键断裂箭头：从 C(0)-Cl(1) 键中点指向 Cl(1)）
    [CONDITION:SN2]                    （主箭头上方的条件文本）
    [/COMPOSITE]

布局种类：
    reaction_mech: 反应式 + 机理场景，必须包含至少一个 [RXNARROW]；
    row: 纯横向组件排列（共振式、多步序列等），[RXNARROW] 可选。
    energy: 势能面 + 驻点结构（R-3）：容器内需一个 [ENERGY:点序列]，
        每个 STRUCT 用 at=点序号 挂到驻点上（pos=above/below 可选，默认 above）。
    共振式（R-6）：任意布局内用显式连接符组装——
        [RESARROW] 共振箭头 ↔（手动插入）；[NEWLINE] 换行（组件在多行中上下排列）。
        含 [RESARROW] 时自动保留各极限式显式键级（Kekulé 式不统一芳香化）。
    头部可追加标志：[COMPOSITE:reaction_mech,numbering] 打开原子序号标注
    （默认不显示；仅在碳原子较多、需要指明参与反应的原子时使用）。

连接符规则：
    - 相邻 [STRUCT] 之间默认只留间距；需要“+”必须显式写 [PLUS]；
    - [RXNARROW:条件] 可内联条件文本；[CONDITION:x] 填充第一个无内联条件的
      主箭头；
    - 多个 [RXNARROW] 可形成 A → B → C 多步序列。

机理箭头引用：组件 id（显式 id= 或自动 r0/r1/...）+ 端点引用。
端点可以是原子序号（SMILES 顺序，0 起），也可以是 "a-b" 形式的键中点
（σ 键断裂箭头从键发出，如 r0:0-1>r0:1）；目标端还支持 "id:a+id:b" 形式的
成键空白位（两原子位置中点，可跨组件）——自由基机理中两个成键鱼钩汇聚于
新键形成处（如 br:0>>br:0+cc:0），钩尖自动留出小间隙、不指向任何原子标签。
杂原子起点自动上移到孤对电子区域；键中点出发的箭头向下弯，其余向上弯。
引用未知 id 或越界原子的箭头会被跳过，不影响整体渲染。

组件级标注（R-2，随分子 scope 一起移动）：
    [CHARGE:ref|idx:δ+,idx:δ-,...]   组件 ref 上的部分电荷（红色）
    [HBOND:ref|from-to,...]          组件 ref 内的氢键虚线（teal dashed）

绘制风格（统一标签规则，heavy_atom_count）：>2 重原子分子按键线式
（碳原子不标 CHn，骨架线隐含）；≤2 重原子小分子（CH3Cl、CH2=CH2、CH4
等）用结构简式（非环碳写出 CH₃/CH₂/CH）。label 中的纯化学式（如 CH3Cl）
不会重复显示——小分子本身已是简式；中文名称/角色标注（如 底物、亲核
试剂）仍显示在分子下方。
label 与条件写普通文本即可（CH3Cl、OH-、H2SO4），渲染器自动把数字转为
下标、尾部电荷转为上标。
"""

import math
import re

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from renderers.collide import Occupancy
    from renderers.mol_primitives import (
        _ARROW_POINT_GAP, _LABEL_SQUARE_HALF, _MECH_LABEL_GAP,
        bond_order_of, format_chem_text,
        format_partial_charge,
        hbond_dots_tikz, mech_arrow_between, mech_arrow_origin,
        mol_visual_bbox, mol_visual_bbox_xh, parse_charge_pairs, parse_hbond_pairs, atom_label,
        atom_main_label, bond_segments, bond_segments_for, label_bond_margin,
        label_edge_point, h_label_edge_point, prepare_mol, rotate_mol_coords, scale_mol_coords,
        symbol_center,
        atom_pos, place_donor_h, place_explicit_hs, place_h_avoiding,
        adjust_hbond_conformation, lone_pair_angles, _ang_diff,
        _covalent_bond_len,
        partial_charge_pos, main_arrow_lines, split_species_coeff, wrap_format_text,
        is_formula_label, heavy_atom_count,
    )
    from renderers.layout import (
        energy_annotation_placement, energy_point_coords, energy_point_roles,
        layout_row, layout_rows, molecule_scope_lines, place_bbox,
    )
else:
    from .collide import Occupancy
    from .mol_primitives import (
        _ARROW_POINT_GAP, _LABEL_SQUARE_HALF, _MECH_LABEL_GAP,
        bond_order_of, format_chem_text,
        format_partial_charge,
        hbond_dots_tikz, mech_arrow_between, mech_arrow_origin,
        mol_visual_bbox, mol_visual_bbox_xh, parse_charge_pairs, parse_hbond_pairs, atom_label,
        atom_main_label, bond_segments, bond_segments_for, label_bond_margin,
        label_edge_point, h_label_edge_point, prepare_mol, rotate_mol_coords, scale_mol_coords,
        symbol_center,
        atom_pos, place_donor_h, place_explicit_hs, place_h_avoiding,
        adjust_hbond_conformation, lone_pair_angles, _ang_diff,
        _covalent_bond_len,
        partial_charge_pos, main_arrow_lines, split_species_coeff, wrap_format_text,
        is_formula_label, heavy_atom_count,
    )
    from .layout import (
        energy_annotation_placement, energy_point_coords, energy_point_roles,
        layout_row, layout_rows, molecule_scope_lines, place_bbox,
    )


_MOL_GAP = 0.8    # 无连接符时相邻分子的水平间距
_PLUS_W = 1.1     # [PLUS] 连接符占宽
_ARR_W = 2.6      # [RXNARROW] 占宽
_ARR_PAD = 0.65   # 主箭头两端内缩余量（箭头实际长度 1.3）
_MOL_SCALE = 0.8  # 分子坐标缩放因子（紧凑化，不影响字号）


def _fmt_coeff(c: float) -> str:
    """系数显示：整数原样、n/2 分数形式（1/2、3/2）。"""
    if c == int(c):
        return str(int(c))
    return f"{int(c * 2)}/2"

# 端点支持三种：原子序号（0）、键中点（0-1）、显式 H（0#1 = 原子 0 的第 1 个 XH）
_MECH_PT_RE = r"\d+(?:-\d+)?(?:#\d+)?"
_MECH_ARROW_RE = re.compile(
    rf"^\s*([A-Za-z0-9_]+)\s*:\s*({_MECH_PT_RE})\s*(>>|>)\s*"
    rf"([A-Za-z0-9_]+)\s*:\s*({_MECH_PT_RE})"
    r"(?:\s*\+\s*([A-Za-z0-9_]+)\s*:\s*(\d+))?\s*$"
)

_SUPPORTED_LAYOUTS = ("reaction_mech", "reaction", "row", "energy")

# 氢键 spec：a#k>idB:b（给体组件的第 k 个显式 H 与组件 idB 的原子 b 成氢键）。
# 分子内/分子间统一：分子内时 idB 与给体组件同 id。氢由 [XH] 负责绘制。
_HBOND_RE = re.compile(r"^(\d+)#(\d+)>([A-Za-z0-9_]+):(\d+)$")


def _parse_hbond_specs(annotations: dict) -> list:
    """容器内 HBOND spec（a#k>idB:b）→ [(idA, a, k, idB, b), ...]。

    从各组件 anno["hbond"] 提取；给体组件为 idA（spec 所在组件）、受体
    组件为 idB（分子内时同 id）。
    """
    out = []
    for ida, anno in annotations.items():
        for tok in (anno.get("hbond", "") or "").split(","):
            tok = tok.strip()
            if not tok:
                continue
            m = _HBOND_RE.match(tok)
            if m:
                out.append((ida, int(m.group(1)), int(m.group(2)),
                            m.group(3), int(m.group(4))))
    return out


def _rects_intersect(a: tuple, b: tuple, pad: float = 0.15) -> bool:
    """两个 (min_x, min_y, max_x, max_y) 矩形是否相交（含 pad 间距）。"""
    return not (a[2] + pad < b[0] or b[2] + pad < a[0]
                or a[3] + pad < b[1] or b[3] + pad < a[1])


def _parse_mech_arrows(specs):
    arrows = []
    for spec in specs:
        m = _MECH_ARROW_RE.match(spec)
        if not m:
            continue
        src_id, src_pt, sep, dst_id, dst_pt, dst2_id, dst2_pt = m.groups()
        kind = "fishhook" if sep == ">>" else "standard"
        arrows.append((src_id, src_pt, dst_id, dst_pt, kind, dst2_id, dst2_pt))
    return arrows


def _mech_labeler(info):
    """组件级标签器（与分子绘制同款，heavy_atom_count 规则）：
    ≤2 重原子小分子用结构简式（非环碳写 CHn），其余用键线式（碳不标）。
    供机理箭头端点吸附到标签边缘用。"""
    hs = info["explicit_hs"]
    if heavy_atom_count(info["mol"]) <= 2:
        return lambda a: atom_main_label(a, hs.get(a.GetIdx(), 0))
    return lambda a: atom_label(a, hs.get(a.GetIdx(), 0))


def _bond_form_midpoint(mols: dict, id_a: str, pt_a: str,
                        id_b: str, pt_b: str,
                        avoid: list | None = None) -> tuple | None:
    """成键空白位终点：两原子（可跨组件）原子位置的中点。

    自由基机理中新键形成于此前不相连的两原子之间，两个成键鱼钩汇聚于该
    空白位置而非任何原子标签（钩尖由 inset_end 留出小间隙）。
    avoid: [(x, y), ...] 需避让的点（如 "+" 号位置）——中点与某点过近
    （水平距离 < 0.35）时沿垂直方向偏移 ±0.3 避开（"+ 号在 y=0"）。
    返回 (x, y, False, False, False)；原子越界返回 None。
    """
    ma, mb = mols[id_a]["mol"], mols[id_b]["mol"]
    ia, ib = int(pt_a), int(pt_b)
    if ia >= ma.GetNumAtoms() or ib >= mb.GetNumAtoms():
        return None
    xa, ya = atom_pos(ma, ia)
    xb, yb = atom_pos(mb, ib)
    sa, sb = mols[id_a]["shift"], mols[id_b]["shift"]
    t = 0.5
    mx = (1 - t) * (xa + sa[0]) + t * (xb + sb[0])
    my = (1 - t) * (ya + sa[1]) + t * (yb + sb[1])
    for ax_, ay_ in avoid or []:
        if abs(mx - ax_) < 0.35 and abs(my - ay_) < 0.35:
            my += 0.3 if my >= ay_ else -0.3  # 垂直偏移，避开 y=0 的 +
            break
    return (mx, my, False, False, False)


def _seg_point_dist(seg, pt):
    """点到线段的最短距离（空间感知弯向用）。"""
    x1, y1, x2, y2 = seg
    px, py = pt
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return math.hypot(px - x1, py - y1)
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return math.hypot(px - x2, py - y2)
    b = c1 / c2
    return math.hypot(px - (x1 + b * vx), py - (y1 + b * vy))


def _arrow_near_segments(sm, dm):
    """源/目标组件的键线段（全局坐标），供弯向空间感知。

    bond_segments 返回局部坐标（分子 2D），平移 shift 到画布坐标。
    """
    segs = []
    for info in (sm, dm):
        mol, sh = info["mol"], info["shift"]
        for s in bond_segments(mol, labeler=_mech_labeler(info),
                               margin_fn=label_bond_margin):
            segs.extend((a + sh[0], b + sh[1], c + sh[0], d + sh[1])
                        for a, b, c, d in s)
    return segs


def _arc_mid(fx, fy, tx, ty, mag, side):
    """贝塞尔 t=0.5 弧线中点：弦中点 + 0.5×mag×法线×side（弯向侧）。"""
    dx, dy = tx - fx, ty - fy
    L = math.hypot(dx, dy) or 1.0
    px, py = -dy / L, dx / L
    return ((fx + tx) / 2.0 + side * 0.5 * mag * px,
            (fy + ty) / 2.0 + side * 0.5 * mag * py)


def _pick_bend_side(p0, p1, from_bond, aim_end, sm, dm):
    """空间感知弯向（20260815）：评估两个候选弯向（弦法线 ±）的弧线中点
    距源/目标组件键线的最短距离，选空旷侧。

    默认侧复现原 bend 语义（from_bond 断键→法线反侧、进攻→法线正侧，
    法线 y 分量近似"上下"；弦竖直退化时按 bend 符号固定）；默认侧贴近
    键线（< 0.20）而反侧明显更空旷（> 默认+0.05）时翻转弯向。
    返回 ±1（弦法线方向 side）。
    """
    fx, fy = p0[0], p0[1]
    tx, ty = p1[0], p1[1]
    dx, dy = tx - fx, ty - fy
    dist = math.hypot(dx, dy) or 1.0
    py = dx / dist                         # 法线 y 分量（px, py = -uy, ux）
    bend_sign = -1.0 if from_bond else 1.0
    if abs(py) < 1e-9:
        side = 1.0 if bend_sign >= 0 else -1.0
    else:
        side = (1.0 if py > 0 else -1.0) * bend_sign
    segs = _arrow_near_segments(sm, dm)
    if not segs:
        return side
    mag = (min(0.30 * dist + 0.15, 1.15) if aim_end
           else min(0.22 * dist + 0.15, 0.6))
    m1 = _arc_mid(fx, fy, tx, ty, mag, side)
    m2 = _arc_mid(fx, fy, tx, ty, mag, -side)
    d1 = min(_seg_point_dist(s, m1) for s in segs)
    d2 = min(_seg_point_dist(s, m2) for s in segs)
    if d1 < 0.20 and d2 > d1 + 0.05:
        return -side
    return side


def draw_mech_arrows(mols: dict, arrows: list,
                     plus_positions: list | None = None) -> list:
    """绘制机理弯箭头（p0/p1 定位、端点吸附避让），返回 TikZ 行列表。

    mols: {组件 id: {"mol": RDKit Mol, "shift": (x, y), ...}} 组件表。
    arrows: [(src_id, src_pt, dst_id, dst_pt, kind, dst2_id, dst2_pt), ...]
        ——_parse_mech_arrows 输出；src_pt/dst_pt 为原子序号、"a-b" 键中点
        或 "a#k" 显式 H（断键语义）；dst2_id/dst2_pt 非 None 时目标端为
        "dst_id:dst_pt+dst2_id:dst2_pt" 的成键空白位（两原子位置中点，
        可跨组件，避开 "+" 号位置）。
    plus_positions: [(x, y), ...] 加号位置——成键空白位与其重叠时偏移。
    未知组件 id / 无效端点的箭头跳过，不影响整体渲染。

    逻辑：目标端先按原子中心定位（供源端选孤对槽位）；源端确定后，
    再按实际源端把目标端吸附到标签边缘空隙（箭头尖不压标签）。
    """
    lines = []
    for src_id, src_pt, dst_id, dst_pt, kind, dst2_id, dst2_pt in arrows:
        if src_id not in mols or dst_id not in mols:
            continue
        if dst2_id is not None and (dst2_id not in mols or "-" in dst_pt):
            continue
        sm = mols[src_id]
        dm = mols[dst_id]
        slab = _mech_labeler(sm)
        dlab = _mech_labeler(dm)
        if dst2_id is None:
            p1 = mech_arrow_origin(dm["mol"], dst_pt, dm["shift"],
                                   lone_pair_offset=False,
                                   xh_points=dm.get("xh_points"),
                                   as_target=True)
        else:
            p1 = _bond_form_midpoint(mols, dst_id, dst_pt, dst2_id, dst2_pt,
                                     avoid=plus_positions)
        if p1 is None:
            continue
        p0 = mech_arrow_origin(sm["mol"], src_pt, sm["shift"],
                               toward=(p1[0], p1[1]),
                               prefer_single=(kind == "fishhook"),
                               labeler=slab,
                               bend_side=-1.0 if "-" in src_pt else 1.0,
                               xh_points=sm.get("xh_points"))
        if p0 is None:
            continue
        if dst2_id is None:
            p1 = mech_arrow_origin(dm["mol"], dst_pt, dm["shift"],
                                   lone_pair_offset=False,
                                   toward=(p0[0], p0[1]), labeler=dlab,
                                   bend_side=-1.0 if p0[2] else 1.0,
                                   xh_points=dm.get("xh_points"),
                                   as_target=True)
            if p1 is None:
                continue
        # 断键起点：σ 键中点（a-b）或显式 H（a#k，X—H 键端点）——都从键出发
        bond_break = (("-" in src_pt and bond_order_of(sm["mol"], src_pt) == 1)
                      or "#" in src_pt)
        # 起点 gap：断键/电子点起点沿弯向法线方向（gap_along_bend——
        # "向上弯则向上 gap"），普通原子起点沿弦方向内缩 0.15；
        # 吸附标签/靠外杠起点（from_bond 非断键）无 gap
        gap_along_bend = bond_break or p0[3]
        inset_start = (_ARROW_POINT_GAP if gap_along_bend
                       else (0.0 if (p0[2] or p0[4]) else 0.15))
        # aim_end（末端沿切线退让到标签正方形外 0.05）：纯原子终点（元素标签）
        # 与 a#k 终点（H 节点标签）都启用——H 节点在渲染中同为 \node{H}
        # （fill=white, inner sep=1pt），占位约边长 0.26（_LABEL_SQUARE_HALF=0.13），
        # 箭头尖端沿切线退到正方形边缘外 _MECH_LABEL_GAP（20260815：
        # 原 a#k 只沿箭头方向内缩 0.05，斜向入射仍压 H 占位）。
        tb = None
        aim_end = False
        if p1[4] and "-" not in dst_pt:
            if "#" in dst_pt:
                a_id, _, k_s = dst_pt.partition("#")
                try:
                    ia, ik = int(a_id), int(k_s)
                except ValueError:
                    ia, ik = -1, -1
                pts = dm.get("xh_points", {}).get(ia)
                if pts and 1 <= ik <= len(pts):
                    hx, hy = pts[ik - 1]
                    tb = (hx + dm["shift"][0], hy + dm["shift"][1],
                          _LABEL_SQUARE_HALF, _LABEL_SQUARE_HALF)
                    aim_end = True
            else:
                da = dm["mol"].GetAtomWithIdx(int(dst_pt))
                ax, ay = symbol_center(dm["mol"], int(dst_pt))
                tb = (ax + dm["shift"][0], ay + dm["shift"][1],
                      _LABEL_SQUARE_HALF, _LABEL_SQUARE_HALF)
                aim_end = True
        inset_end = (_MECH_LABEL_GAP if aim_end
                     else (0.0 if p1[4] else 0.10))
        # 弯向空间感知（在 aim_end 确定后）：默认侧（from_bond 断键向下/
        # 进攻向上——含双键起点，与 mech_arrow_between 的 bend 符号一致）
        # 贴近键线而反侧空旷时翻转，起点 gap 随弯向对齐
        bend_side = _pick_bend_side(p0, p1, p0[2], aim_end, sm, dm)
        lines.extend(
            mech_arrow_between(p0[0], p0[1], p1[0], p1[1], kind,
                               from_bond=p0[2], inset_start=inset_start,
                               inset_end=inset_end, bond_break=bond_break,
                               aim_end=aim_end, text_box=tb,
                               bend_side=bend_side,
                               gap_along_bend=gap_along_bend)
        )
    return lines


def _collect_components(children):
    structs = []
    sequence = []
    mech_specs = []
    global_cond = ""
    annotations = {}
    blocks = []   # 大一统架构 [BLOCK] 共振块（内部子标记列表）
    for child in children:
        if child.type == "STRUCT":
            cid = child.attrs.get("id") or f"r{len(structs)}"
            label = child.args[1] if len(child.args) > 1 else None
            parsed = split_species_coeff(child.args[0])
            smi = parsed[0][1] if parsed else child.args[0].strip()
            structs.append({
                "id": cid,
                "smiles": smi,
                "coeff": parsed[0][0] if parsed else 1.0,
                "label": label,
                "at": child.attrs.get("at"),
                "pos": child.attrs.get("pos", "above"),
                # 分子家族重构（20260818）：容器内 mode ∈ {skeleton, lewis}
                # （校验层限制）；mode=lewis 时该组件显示孤对电子点
                "mode": child.attrs.get("mode", "skeleton"),
                # 大一统架构（20260819）：arrow 令牌 = 箭头上附件（副反应物/
                # 副产物），不参与主序列，由 ARROW 的 sup= 参数引用
                "arrow": bool(child.attrs.get("arrow")),
            })
            if not child.attrs.get("arrow"):
                sequence.append(("mol", len(structs) - 1))
        elif child.type == "PLUS":
            sequence.append(("plus",))
        elif child.type == "RESARROW":
            sequence.append(("resarrow",))
        elif child.type == "NEWLINE":
            sequence.append(("newline",))
        elif child.type == "RXNARROW":
            # 旧箭头标记（兼容保留）：正向 + 条件；kind=None 让主箭头函数
            # 按条件中的 ⇌ 令牌自动识别（可逆）
            cond = child.args[0].strip() if child.args else ""
            sequence.append(("arrow", cond, None, []))
        elif child.type == "ARROW":
            # 大一统架构箭头：type= 类型、sup= 附件列表、其余为条件
            kind = (child.args[0] if child.args else "") or "single"
            sup = child.args[1] if len(child.args) > 1 else []
            cond = child.args[2] if len(child.args) > 2 else ""
            sequence.append(("arrow", cond, kind, sup))
        elif child.type == "BLOCK":
            blocks.append(child.args[0] if child.args else [])
            sequence.append(("block", len(blocks) - 1))
        elif child.type == "CONDITION":
            if child.args and not global_cond:
                global_cond = child.args[0].strip()
        elif child.type == "MECHARROW":
            if child.args:
                mech_specs.extend(child.args[0].split(","))
        elif child.type in ("CHARGE", "HBOND") and len(child.args) >= 2:
            ref = child.args[0].strip()
            annotations.setdefault(ref, {})[child.type.lower()] = child.args[1]
        elif child.type == "XH" and len(child.args) >= 2:
            ref = child.args[0].strip()
            try:
                a = int(child.args[1])
                annotations.setdefault(ref, {}).setdefault("xh", []).append(a)
            except ValueError:
                pass
        elif child.type == "BOND" and len(child.args) >= 2:
            ref = child.args[0].strip()
            annotations.setdefault(ref, {}).setdefault("bonds", []).append(
                child.args[1].strip())
    return structs, sequence, mech_specs, global_cond, annotations, blocks


def _pseudo_hbond_lines(info: dict, hbond_toward: dict,
                        hbond_away: dict) -> list:
    """孤立原子（被 HBOND 引用）的假骨架：水/氨画完整结构式。

    O：中心 + V 形 2 个 H（键角 104.5°）；N：三角锥 3 个 H（120° 平面）。
    给体（donor）：第 1 个 H 精确朝受体方向（hbond_toward，X—H···Y 直线）；
    受体（acceptor）：H 在远离给体方向（hbond_away）两侧对称张开。
    返回 TikZ 行（全局坐标，含 shift）；H 位置存入 info["hbond_h_pos"]
    （局部坐标，供 HBOND 点状虚线起点）。
    """
    mol = info["mol"]
    atom = mol.GetAtomWithIdx(0)
    z = atom.GetAtomicNum()
    pseudo = info.get("hbond_pseudo", {})
    role = pseudo.get(0, "donor")
    if role == "donor" and 0 in hbond_toward:
        tx, ty = hbond_toward[0]
        target = math.degrees(math.atan2(ty, tx)) % 360.0
    elif role == "acceptor" and 0 in hbond_away:
        ax, ay = hbond_away[0]
        target = math.degrees(math.atan2(ay, ax)) % 360.0
    else:
        target = 180.0  # 兜底：指向左（row 布局给体在左）
    if z == 8:      # O：2 个 H，V 形（教科书常用 120°）
        n_h, bond_ang = 2, 120.0
    elif z == 7:    # N：3 个 H，平面 120°
        n_h, bond_ang = 3, 120.0
    else:
        n_h, bond_ang = 1, 0.0
    h_len = _covalent_bond_len(atom) * _MOL_SCALE
    if role == "donor":
        # 第 1 个 H 精确朝受体，其余按键角依次绕开
        angles = [(target - k * bond_ang) % 360.0 for k in range(n_h)]
    else:
        # 受体：H 在远离给体方向两侧对称（V 形张开）；N（3H）补第 3 个 H
        # 在角平分线上（沿远离给体方向正中，row 布局下即水平向右）
        half = bond_ang / 2.0
        angles = [(target + half) % 360.0,
                  (target - half) % 360.0]
        if n_h >= 3:
            angles.append(target % 360.0)
        angles = angles[:n_h]
    cx, cy = atom_pos(mol, 0)
    sx, sy = cx + info["shift"][0], cy + info["shift"][1]
    lines = [
        f"  \\node[fill=white, inner sep=1pt] at ({sx:.2f},{sy:.2f}) "
        f"{{{atom.GetSymbol()}}};"
    ]
    # 键与标签的 gap：复用键线式统一留白（label_bond_margin，
    # 单字符标签 0.30；与 label_edge_point 同口径）
    sym_lab = atom.GetSymbol()
    gap_o = label_bond_margin(sym_lab)
    gap_h = label_bond_margin("H")
    h_pos = []
    for ang in angles:
        hx = cx + h_len * math.cos(math.radians(ang))
        hy = cy + h_len * math.sin(math.radians(ang))
        h_pos.append((hx, hy))
        gx, gy = hx + info["shift"][0], hy + info["shift"][1]
        # O 侧：从 O 中心沿 H 方向内缩 gap_o；H 侧：从 H 中心沿 O 方向内缩 gap_h
        d = math.hypot(hx - cx, hy - cy) or 1.0
        ux, uy = (hx - cx) / d, (hy - cy) / d
        ox = sx + ux * gap_o
        oy = sy + uy * gap_o
        hx2 = gx - ux * gap_h
        hy2 = gy - uy * gap_h
        lines.append(f"  \\draw ({ox:.2f},{oy:.2f}) -- ({hx2:.2f},{hy2:.2f});")
        lines.append(
            f"  \\node[fill=white, inner sep=1pt] at ({gx:.2f},{gy:.2f}) {{H}};")
    info["hbond_h_pos"] = {0: h_pos}
    label = info.get("label")
    if label and not is_formula_label(label):
        lines.append(
            f"  \\node[below] at ({sx:.2f},{cy + info['shift'][1] - 0.5:.2f}) "
            f"{{{wrap_format_text(label)}}};")
    return lines


def _molecule_with_annotations_lines(info: dict, *, show_numbers: bool,
                                     show_lone_pairs: bool,
                                     hbond_toward: dict = None,
                                     hbond_away: dict = None) -> list:
    """单个分子组件的完整绘制行：系数 + scope 骨架/标签/电子点 + 组件级注解
    （CHARGE 部分电荷 / HBOND 氢键 / XH 显式氢 / BOND 键突出），全部随
    info["shift"] 移动。主行布局与 energy 布局共用（B3：energy 驻点结构
    不再静默丢弃注解）。

    hbond_toward/hbond_away：HBOND 方向回灌——{原子序号: 目标方向点（本组件
    局部坐标）}。XH 绘制显式 H 时，给体原子（被 HBOND 引用为 a#k）的首个 H
    沿 hbond_toward（朝向受体，X—H···Y 尽量直线）；受体原子（被 HBOND 引用
    为 idB:b）的 H 沿 hbond_away（远离给体，避免遮挡氢键虚线）。
    """
    mol = info["mol"]
    hs = info["explicit_hs"]
    hbond_toward = hbond_toward or {}
    hbond_away = hbond_away or {}
    lines = []
    if info.get("hbond_pseudo"):
        # 孤立原子假骨架（水/氨完整结构式）：不走骨架/标签/XH 通用逻辑
        return _pseudo_hbond_lines(info, hbond_toward, hbond_away)
    # 标签风格统一规则（heavy_atom_count）：≤2 重原子小分子结构简式，
    # 其余键线式（带 XH/BOND/HBOND 标注的分子自然落在键线式——标注
    # 聚焦反应位点，碳不标 CHn）
    labeler = _mech_labeler(info)
    occ = Occupancy()   # R-8 占据注册表（局部坐标）：scope 内元素 + XH 注解共用
    if info.get("coeff", 1.0) != 1.0:
        bbox = info.get("bbox")
        if bbox:
            bx = bbox[0] + info["shift"][0] - 0.15
            lines.append(
                f"  \\node at ({bx:.2f},{info['shift'][1]:.2f}) "
                f"{{{_fmt_coeff(info['coeff'])}}};"
            )
    lines.extend(molecule_scope_lines(mol, info["shift"],
                                      show_numbers=show_numbers,
                                      show_lone_pairs=show_lone_pairs,
                                      explicit_hs=hs,
                                      occupancy=occ))
    for idx, raw_label in info["charges"].items():
        if idx >= mol.GetNumAtoms():
            continue
        # 部分电荷以元素符号中心为基准（与孤对电子同一基准），方向避让
        x, y = partial_charge_pos(mol, idx, info["shift"], hs.get(idx, 0))
        lines.append(
            f"  \\node[font=\\small, red] at ({x:.2f},{y:.2f}) "
            f"{{{format_partial_charge(raw_label)}}};"
        )
    # [XH] 显式氢：标签已按 explicit_hs 扣减，此处画出 X—H 实线 + H 节点
    xh_counts = {}
    for a in info["xh"]:
        xh_counts[a] = xh_counts.get(a, 0) + 1
    xh_points = {}   # {原子序号: [(hx, hy), ...]} 供 MECHARROW "a#k" 端点引用
    mol_has_bond = mol.GetNumBonds() > 0
    for a, count in xh_counts.items():
        if a >= mol.GetNumAtoms():
            continue
        # 孤立原子（分子无键）：h_len 用 2×共价半径并按 _MOL_SCALE 缩放，
        # 使显式 H 键与骨架键等长（问题 2：原 0.75 未缩放，键明显偏短）
        h_len = None
        if not mol_has_bond:
            h_len = _covalent_bond_len(mol.GetAtomWithIdx(a)) * _MOL_SCALE
        # HBOND 方向回灌：给体 H 朝向受体、受体 H 远离给体（place_explicit_hs
        # 的 toward 只影响首个 H——给体常画 1 个 H，正好是 a#1 被引用者）
        toward = None
        if a in hbond_toward:
            toward = hbond_toward[a]
        elif a in hbond_away:
            toward = hbond_away[a]
        if toward is not None:
            pos_list = place_explicit_hs(mol, a, count, toward=toward,
                                         h_len=h_len)
        else:
            pos_list = place_explicit_hs(mol, a, count, h_len=h_len)
        for hx, hy in pos_list:
            # R-8 避障：规则位置撞键/标签/电荷圈/电子点时绕原子旋转取候选
            hx, hy = place_h_avoiding(mol, a, (hx, hy), occ)
            sx, sy = label_edge_point(mol, a, (hx, hy), labeler=labeler)
            # H 端同规则留白（label_bond_margin("H")=0.30）：键线终点停在
            # H 节点占位之外，不画到 H 中心（与假骨架水/氨一致）
            ex, ey = h_label_edge_point(hx, hy, atom_pos(mol, a))
            hx += info["shift"][0]
            hy += info["shift"][1]
            sx += info["shift"][0]
            sy += info["shift"][1]
            ex += info["shift"][0]
            ey += info["shift"][1]
            lines.append(f"  \\draw ({sx:.2f},{sy:.2f}) -- ({ex:.2f},{ey:.2f});")
            lines.append(
                f"  \\node[fill=white, inner sep=1pt] at ({hx:.2f},{hy:.2f}) {{H}};"
            )
            # 存局部坐标（未加 shift）：mech_arrow_origin 的 a#k 分支会加 shift，
            # 否则双重平移导致 b/c 箭头起点偏移（问题 8）
            xh_points.setdefault(a, []).append((hx - info["shift"][0],
                                                hy - info["shift"][1]))
    info["xh_points"] = xh_points
    # [BOND] 反应位点键突出：复用骨架修剪段，红色粗线与原键完全对齐
    for spec in info["bonds"]:
        if "-" not in spec:
            continue
        sa, _, sb = spec.partition("-")
        try:
            a, b = int(sa), int(sb)
        except ValueError:
            continue
        if a >= mol.GetNumAtoms() or b >= mol.GetNumAtoms():
            continue
        segs = bond_segments_for(mol, a, b, labeler=labeler,
                                 margin_fn=label_bond_margin)
        if segs is None:
            continue
        for x1, y1, x2, y2 in segs:
            lines.append(
                f"  \\draw[very thick, red] "
                f"({x1 + info['shift'][0]:.2f},{y1 + info['shift'][1]:.2f}) -- "
                f"({x2 + info['shift'][0]:.2f},{y2 + info['shift'][1]:.2f});"
            )
    return lines


def _render_energy_layout(points_str: str, structs: list, mols: dict,
                          show_numbers: bool) -> str:
    """energy 布局：势能面曲线 + 驻点结构组件（R-3）。

    每个 STRUCT 通过 at= 挂到能量点上（pos=above/below，默认 above），
    分子按视觉包围盒置于驻点正上方/下方；驻点标签优先用 STRUCT 的 label。
    """
    try:
        values = [float(v.strip()) for v in points_str.split(",") if v.strip()]
    except ValueError:
        return f"（COMPOSITE 渲染失败：能量点序列格式错误「{points_str}」）"
    if len(values) < 2:
        return "（COMPOSITE 渲染失败：能量点至少需要 2 个）"

    n = len(values)
    info = energy_point_coords(values)
    x_last = info["x_last"]
    roles = energy_point_roles(values)

    at_map = {}
    for comp in structs:
        if comp["at"] is None:
            return (f"（COMPOSITE 渲染失败：energy 布局中 STRUCT 组件 "
                    f"{comp['id']} 需要 at=点序号）")
        if comp["at"] >= n:
            return (f"（COMPOSITE 渲染失败：组件 {comp['id']} 的 "
                    f"at={comp['at']} 超出能量点范围 0~{n - 1}）")
        at_map[comp["at"]] = comp

    # 先计算全部组件的已占区域，再决定标注框与纵轴高度（避免遮挡）
    # P3 冲突消解：先预置驻点标签区域；结构按驻点序号放置（相邻先检测），
    # 与已占区域重叠时向右错开（最多 8.0），避免相邻驻点结构互相压叠。
    occupied = []
    for i, v, x, y in info["points"]:
        label = (at_map[i]["label"]
                 if (i in at_map and at_map[i]["label"]) else roles.get(i))
        if not label:
            continue
        yoff = 0.35 if roles.get(i) == "过渡态" else -0.3
        occupied.append((x - 0.85, y + yoff - 0.22, x + 0.85, y + yoff + 0.22))

    mol_placements = []
    for comp in sorted(structs, key=lambda c: c["at"]):
        _, _, x, y = info["points"][comp["at"]]
        cinfo = mols[comp["id"]]
        mol = cinfo["mol"]
        bbox = mol_visual_bbox(mol, include_lone_pairs=False)
        shift = place_bbox(bbox, x, y, comp["pos"], margin=0.6)
        for _ in range(20):  # 最多右移 20×0.4 = 8.0
            rect = (bbox[0] + shift[0], bbox[1] + shift[1],
                    bbox[2] + shift[0], bbox[3] + shift[1])
            if not any(_rects_intersect(rect, o) for o in occupied):
                break
            shift = (shift[0] + 0.4, shift[1])
        cinfo["shift"] = shift
        mol_placements.append(cinfo)
        occupied.append(rect)

    box_x, box_y, box_anchor, axis_top = energy_annotation_placement(
        occupied, x_last)

    lines = [r"\begin{tikzpicture}"]
    lines.append(f"  \\draw[->] (0,0) -- ({x_last + 0.8:.1f},0);")
    lines.append(f"  \\draw[->] (0,0) -- (0,{axis_top:.1f});")
    lines.append(f"  \\node[font=\\small] at ({(x_last + 0.8) / 2:.1f},-0.30) {{反应进程}};")
    lines.append(
        f"  \\node[font=\\small, rotate=90, anchor=south] at (-0.10,{axis_top - 0.5:.1f}) "
        "{能量 (kJ/mol)};"
    )
    y0 = info["points"][0][3]
    lines.append(f"  \\draw[gray, dashed] (0,{y0:.2f}) -- ({x_last:.1f},{y0:.2f});")
    coords = " ".join(f"({x:.1f},{y:.2f})" for _, _, x, y in info["points"])
    lines.append(f"  \\draw[thick, blue, smooth] plot coordinates {{{coords}}};")

    for i, v, x, y in info["points"]:
        comp = at_map.get(i)
        label = comp["label"] if (comp and comp["label"]) else roles.get(i)
        lines.append(f"  \\begin{{scope}}[shift={{({x:.1f},{y:.2f})}}]")
        lines.append("    \\fill[blue] (0,0) circle (0.06);")
        if label:
            yoff = 0.35 if roles.get(i) == "过渡态" else -0.3
            lines.append(f"    \\node[font=\\small] at (0,{yoff:.2f}) "
                         f"{{{format_chem_text(label)} ({v:+.0f})}};")
        lines.append("  \\end{scope}")

    for cinfo in mol_placements:
        lines.extend(_molecule_with_annotations_lines(
            cinfo, show_numbers=show_numbers, show_lone_pairs=False))

    ea = max(values) - values[0]
    dh = values[-1] - values[0]
    node_text = f"Ea $\\approx$ {ea:.0f} kJ/mol\\\\$\\Delta$H $\\approx$ {dh:+.0f} kJ/mol"
    lines.append(
        "    \\node[draw, rounded corners, fill=yellow!10, font=\\small, align=left, "
        f"anchor={box_anchor}] at ({box_x:.2f},{box_y:.2f}) {{{node_text}}};"
    )
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def render_composite(layout: str, children: list) -> str:
    r"""[COMPOSITE] 渲染：容器内组件 → 统一坐标系单张 TikZ。

    参数:
        layout: 布局名（reaction_mech / row），可追加逗号分隔的标志
                （如 "reaction_mech,numbering" 打开原子序号标注）。
        children: 容器内子标记 RenderTag 列表（core.tag_parser 解析结果）。

    返回:
        可编译的 TikZ 代码；失败返回可读错误提示。
    """
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return "（COMPOSITE 渲染失败：rdkit 未安装）"

    header = [p.strip() for p in (layout or "").split(",")]
    layout_name = header[0]
    flags = set(header[1:])
    if layout_name not in _SUPPORTED_LAYOUTS:
        return (
            f"（COMPOSITE 渲染失败：未知布局「{layout_name}」，"
            f"支持 {' / '.join(_SUPPORTED_LAYOUTS)}）"
        )
    show_numbers = "numbering" in flags

    structs, sequence, mech_specs, global_cond, annotations, blocks = \
        _collect_components(children)

    # row 布局允许无 [STRUCT]（纯箭头/条件/连接符序列也合法，校验层已同步豁免）；
    # reaction_mech / reaction / energy 仍要求至少一个组件（reaction 允许仅 BLOCK）
    if not structs and not blocks and layout_name != "row":
        return "（COMPOSITE 渲染失败：容器内缺少 [STRUCT] 组件）"
    if layout_name == "reaction_mech" and not any(el[0] == "arrow" for el in sequence):
        return "（COMPOSITE 渲染失败：reaction_mech 布局需要 [RXNARROW] 标记主反应箭头位置）"

    # 含共振箭头时，极限式必须保留显式键级（跳过芳香化），否则不同
    # Kekulé 式会被统一芳香化成同一结构。
    # 其他布局传 None 让 prepare_mol 自动判定（芳香小写→画圈，
    # 凯库勒大写→保留键级），而非一律芳香化。
    allow_aromatic = (
        False
        if (any(el[0] == "resarrow" for el in sequence)
            or blocks)
        else None
    )

    mols = {}
    for comp in structs:
        mol = prepare_mol(comp["smiles"], allow_aromatic=allow_aromatic)
        if mol is None:
            return f"（COMPOSITE 渲染失败：无效 SMILES「{comp['smiles']}」（组件 {comp['id']}）"
        scale_mol_coords(mol, _MOL_SCALE)
        anno = annotations.get(comp["id"], {})
        # 显式 H 对账：仅 [XH] 子标记（氢的显示由 XH 负责，HBOND 只画点）
        explicit_hs = {}
        for a in anno.get("xh", []):
            explicit_hs[a] = explicit_hs.get(a, 0) + 1
        mols[comp["id"]] = {
            "mol": mol,
            "label": comp["label"],
            "coeff": comp.get("coeff", 1.0),
            "shift": (0.0, 0.0),
            "charges": parse_charge_pairs(anno.get("charge", "")),
            "xh": anno.get("xh", []),
            "bonds": anno.get("bonds", []),
            "explicit_hs": explicit_hs,
            # 分子家族重构：容器内绘制模式（skeleton/lewis）
            "mode": comp.get("mode", "skeleton"),
        }

    # 氢键场景构象调整（布局前）：分子内氢键给体/受体折到主链同一侧
    for ida, a, _k, idb, b in _parse_hbond_specs(annotations):
        if ida != idb:
            continue
        mol = mols[ida]["mol"]
        if a < mol.GetNumAtoms() and b < mol.GetNumAtoms():
            adjust_hbond_conformation(mol, a, b)

    # 孤立原子 + HBOND 引用 → 假骨架模式：水/氨等无骨架小分子画完整结构式
    # （O 画 V 形 2H、N 画三角锥 3H），方向由氢键几何决定（给体 H 朝受体、
    # 受体 H 远离给体）。与 XH 的 a#k 机制互斥（假骨架自带 H，不走 XH）。
    hbond_specs = _parse_hbond_specs(annotations)
    for ida, a, _k, idb, b in hbond_specs:
        info_a, info_b = mols.get(ida), mols.get(idb)
        if info_a is None or info_b is None:
            continue
        if info_a["mol"].GetNumAtoms() == 1:
            info_a.setdefault("hbond_pseudo", {})[a] = "donor"
        if info_b["mol"].GetNumAtoms() == 1:
            info_b.setdefault("hbond_pseudo", {})[b] = "acceptor"

    # 受体对齐旋转（分子间单氢键，受体有骨架）：旋转受体分子使受体原子 b
    # 的孤对电子方向指向给体（row 布局中给体在受体左侧 → 目标为水平向左）。
    # 简化：仅单行 row 布局（给体/受体水平相邻）下生效；双氢键/多行不旋转
    # （保守，避免错误旋转）。孤立原子受体（假骨架）不旋转（方向由假骨架处理）。
    if layout_name == "row" and len(set(el[0] for el in sequence)) == 1:
        for ida, a, _k, idb, b in hbond_specs:
            if ida == idb:
                continue
            info_a, info_b = mols.get(ida), mols.get(idb)
            if info_a is None or info_b is None:
                continue
            if info_b.get("hbond_pseudo"):
                continue
            mol_b = info_b["mol"]
            if not 0 <= b < mol_b.GetNumAtoms():
                continue
            lps = lone_pair_angles(mol_b, b)
            if not lps:
                continue
            # 目标：孤对电子方向 → 水平向左（180°，指向左侧给体）；
            # 选与 180° 最接近的孤对电子方向旋转对齐
            lp_ang = min(lps, key=lambda L: _ang_diff(L, 180.0))
            rotate = (180.0 - lp_ang) % 360.0
            if rotate > 180.0:
                rotate -= 360.0
            rotate_mol_coords(mol_b, rotate,
                              center=symbol_center(mol_b, b, 0))

    if layout_name == "energy":
        energy_child = next((c for c in children if c.type == "ENERGY"), None)
        if energy_child is None or not energy_child.args:
            return "（COMPOSITE 渲染失败：energy 布局需要 [ENERGY:点序列] 组件）"
        return _render_energy_layout(energy_child.args[0], structs, mols,
                                     show_numbers)

    # 统一布局引擎：组件序列 → 位置/加号/共振箭头/反应箭头（视觉包围盒防重叠）
    # [BLOCK] 共振块预渲染：块内 STRUCT + 共振箭头 → 内部布局 → lines + bbox
    block_data = {}   # 块序号 → (lines, bbox)
    for bid, bchildren in enumerate(blocks):
        b_items = []
        b_ids = []
        for bc in bchildren:
            if bc.type == "STRUCT":
                bmol = prepare_mol(bc.args[0].strip() if bc.args else "",
                                   allow_aromatic=False)
                if bmol is None:
                    return (f"（COMPOSITE 渲染失败：BLOCK 内无效 SMILES"
                            f"「{bc.args[0] if bc.args else ''}」）")
                scale_mol_coords(bmol, _MOL_SCALE)
                bid_ = bc.attrs.get("id") or f"b{bid}_{len(b_ids)}"
                b_ids.append(bid_)
                b_items.append(("mol", bid_, bmol))
            elif bc.type == "ARROW":
                b_items.append(("resarrow",))
        if not b_items:
            return "（COMPOSITE 渲染失败：BLOCK 内缺少结构组件）"
        blayout = layout_row(b_items, mol_gap=_MOL_GAP, plus_w=_PLUS_W,
                             arrow_w=_ARR_W, arrow_pad=_ARR_PAD, res_w=1.1)
        b_lines = []
        for placed in blayout.mols:
            b_lines.extend(molecule_scope_lines(placed.mol, placed.shift,
                                                show_lone_pairs=False))
        for rx in blayout.resarrows:
            b_lines.append(
                f"  \\node[font=\\large] at ({rx:.2f},0) {{$\\leftrightarrow$}};")
        # 块包围盒：内部布局范围（含共振箭头占位）
        xs, ys = [], []
        for placed in blayout.mols:
            bb = placed.bbox
            xs += [bb[0] + placed.shift[0], bb[2] + placed.shift[0]]
            ys += [bb[1] + placed.shift[1], bb[3] + placed.shift[1]]
        bbox = (min(xs) - 0.2, min(ys) - 0.2, max(xs) + 0.2, max(ys) + 0.2) \
            if xs else (0.0, -0.3, blayout.width, 0.3)
        block_data[bid] = (b_lines, bbox)

    items = []
    for el in sequence:
        if el[0] == "mol":
            cid = structs[el[1]]["id"]
            items.append(("mol", cid, mols[cid]["mol"],
                          mols[cid]["coeff"]))
        elif el[0] == "block":
            lines, bbox = block_data[el[1]]
            items.append(("block", f"block{el[1]}", lines, bbox))
        elif el[0] in ("plus", "resarrow", "newline"):
            items.append((el[0],))
        elif el[0] == "arrow":
            items.append((el[0], el[1], el[2], el[3]))
    # 布局感知 XH 外延：id(mol) → {原子: 显式 H 数}，供 bbox_fn 计入
    # H 节点位置（问题 6：否则孤立碳 CH4 的 H 超出 bbox，与主箭头重叠）
    xh_by_mol = {}
    for cid, info in mols.items():
        if info.get("xh"):
            cnt = {}
            for a in info["xh"]:
                cnt[a] = cnt.get(a, 0) + 1
            xh_by_mol[id(info["mol"])] = cnt
        # 假骨架组件（孤立原子 + HBOND）：H 数按 O=2/N=3 计入 bbox
        if info.get("hbond_pseudo"):
            cnt = {}
            for a in info["hbond_pseudo"]:
                z = info["mol"].GetAtomWithIdx(a).GetAtomicNum()
                cnt[a] = 2 if z == 8 else 3
            xh_by_mol[id(info["mol"])] = cnt

    def _bbox_with_xh(mol):
        cnt = xh_by_mol.get(id(mol))
        if not cnt:
            return mol_visual_bbox(mol, include_lone_pairs=False)
        return mol_visual_bbox_xh(mol, cnt, h_len_scale=_MOL_SCALE,
                                  include_lone_pairs=False)

    rows, y_offsets = layout_rows(items, mol_gap=_MOL_GAP, plus_w=_PLUS_W,
                                  arrow_w=_ARR_W, arrow_pad=_ARR_PAD,
                                  bbox_fn=_bbox_with_xh)
    plus_positions = []      # (x, yoff)
    res_positions = []       # (x, yoff)
    main_arrows = []         # [x1, x2, cond, yoff, kind, sup]
    for row_layout, yoff in zip(rows, y_offsets):
        for placed in row_layout.mols:
            mols[placed.key]["shift"] = (placed.shift[0], placed.shift[1] - yoff)
            mols[placed.key]["bbox"] = placed.bbox
        plus_positions.extend((px, yoff) for px in row_layout.pluses)
        res_positions.extend((rx, yoff) for rx in row_layout.resarrows)
        main_arrows.extend([a.x1, a.x2, a.condition, yoff, a.kind, a.sup]
                           for a in row_layout.arrows)

    if global_cond:
        for arr in main_arrows:
            if not arr[2]:
                arr[2] = global_cond
                break

    # HBOND 方向回灌：给体 H 朝向受体（X—H···Y 尽量直线）、受体 H 远离给体
    # （避免遮挡虚线）。布局 shift 已定，转各组件局部坐标供 XH 绘制使用。
    hbond_toward = {}   # {idA: {原子a: 受体在 idA 局部坐标}}
    hbond_away = {}     # {idB: {原子b: 远离给体方向点（idB 局部坐标）}}
    for ida, a, _k, idb, b in _parse_hbond_specs(annotations):
        info_a, info_b = mols.get(ida), mols.get(idb)
        if info_a is None or info_b is None:
            continue
        mol_a, mol_b = info_a["mol"], info_b["mol"]
        if not (0 <= a < mol_a.GetNumAtoms()
                and 0 <= b < mol_b.GetNumAtoms()):
            continue
        ga = symbol_center(mol_a, a, 0)
        gb = symbol_center(mol_b, b, 0)
        # 受体全局坐标 → idA 局部（给体 H 朝向方向）
        tx = gb[0] + info_b["shift"][0] - info_a["shift"][0]
        ty = gb[1] + info_b["shift"][1] - info_a["shift"][1]
        hbond_toward.setdefault(ida, {})[a] = (tx, ty)
        if ida != idb:
            # 受体 H 远离给体：受体 → 延长线方向（idB 局部）
            gax = ga[0] + info_a["shift"][0]
            gay = ga[1] + info_a["shift"][1]
            dx = gb[0] + info_b["shift"][0] - gax
            dy = gb[1] + info_b["shift"][1] - gay
            hbond_away.setdefault(idb, {})[b] = (gb[0] + dx, gb[1] + dy)

    lines = [r"\begin{tikzpicture}"]

    # 键线式默认不标孤对电子（规范第 3 条）；仅机理场景（弯箭头起点）、
    # 共振场景（孤对电子参与共轭）自动画出；mode=lewis 组件（分子家族
    # 重构：容器内 Lewis 式分子）同样显示孤对
    global_lone_pairs = (
        bool(mech_specs)
        or any(el[0] == "resarrow" for el in sequence)
    )

    # 每个分子一个 scope（布局引擎积木），组件级标注（电荷/氢键）随分子移动。
    # 附件（arrow 令牌：副反应物/副产物）不在此渲染——由各 ARROW 的 sup
    # 渲染在箭头上下（避免箭头两侧重复出现）。
    for comp in structs:
        if comp.get("arrow"):
            continue
        comp_lone_pairs = global_lone_pairs or \
            mols[comp["id"]].get("mode") == "lewis"
        lines.extend(_molecule_with_annotations_lines(
            mols[comp["id"]], show_numbers=show_numbers,
            show_lone_pairs=comp_lone_pairs,
            hbond_toward=hbond_toward.get(comp["id"], {}),
            hbond_away=hbond_away.get(comp["id"], {})))

    # [BLOCK] 共振块：内部行（含分子 scope）整体平移（布局引擎定位）
    for row_layout, yoff in zip(rows, y_offsets):
        for placed in row_layout.blocks:
            lines.append(
                f"  \\begin{{scope}}[shift={{"
                f"({placed.shift[0]:.2f},{placed.shift[1] - yoff:.2f})}}]")
            lines.extend(placed.lines)
            lines.append("  \\end{scope}")

    # 氢键点状虚线（分子内/分子间统一）：H 位置取自 [XH] 画的显式 H
    # （xh_points，MECHARROW a#k 同机制），受体为组件 idB 的原子 b——
    # HBOND 只画 H···Y 点；X—H 实线与 H 节点由 [XH] 负责。
    for ida, a, k, idb, b in _parse_hbond_specs(annotations):
        info_a, info_b = mols.get(ida), mols.get(idb)
        if info_a is None or info_b is None:
            continue
        # 假骨架组件（孤立原子）：H 位置取自假骨架（hbond_h_pos）；否则 XH
        pts = ((info_a.get("hbond_h_pos") if info_a.get("hbond_pseudo")
                else info_a.get("xh_points")) or {}).get(a) or []
        if not (0 <= k - 1 < len(pts)):
            continue  # 对应 XH 未画出（校验已拦截，渲染端兜底跳过）
        mol_b, sh_b = info_b["mol"], info_b["shift"]
        if not 0 <= b < mol_b.GetNumAtoms():
            continue
        hx0, hy0 = pts[k - 1]
        hx = hx0 + info_a["shift"][0]
        hy = hy0 + info_a["shift"][1]
        tx0, ty0 = symbol_center(mol_b, b, 0)
        # 假骨架受体：标签为单字符 O/N 居中（无 H₂O 等结构简式标签的
        # 符号中心修正），受体坐标用原子中心与假骨架节点保持一致
        if info_b.get("hbond_pseudo"):
            tx0, ty0 = atom_pos(mol_b, b)
        tx, ty = tx0 + sh_b[0], ty0 + sh_b[1]
        # 点线两端内缩：半边长 0.13 标签正方形 + 键线式 gap（label_bond_margin，
        # 单字符标签 0.30）——避免首/末点压住给体 H 与受体标签
        inset = _LABEL_SQUARE_HALF + label_bond_margin("H")
        for dot_line in hbond_dots_tikz(hx, hy, tx, ty,
                                        inset_start=inset, inset_end=inset):
            lines.append(f"  {dot_line}")

    for comp in structs:
        if comp.get("arrow"):
            continue   # 附件 label 不显示（其结构式在箭头上下）
        info = mols[comp["id"]]
        label = info["label"]
        # 纯化学式 label（CH3Cl、Cl·、·CH3、OH-）分子本身已展示，不重复；
        # 中文/角色标注（底物、质子化乙醇）显示在分子下方
        if label and not is_formula_label(label):
            min_x, min_y, max_x, _ = info["bbox"]
            shift = info["shift"]
            cx = (min_x + max_x) / 2.0 + shift[0]
            ly = min_y + shift[1] - 0.35
            text = wrap_format_text(label)
            align = "align=center, " if "\\\\" in text else ""
            lines.append(
                f"  \\node[{align}below] at ({cx:.2f},{ly:.2f}) "
                f"{{{text}}};"
            )

    for px, yoff in plus_positions:
        lines.append(f"  \\node at ({px:.2f},{-yoff:.2f}) {{$+$}};")

    for rx, yoff in res_positions:
        lines.append(f"  \\node[font=\\large] at ({rx:.2f},{-yoff:.2f}) {{$\\leftrightarrow$}};")

    # 附件与箭头的通用边距（20260820 布局避让，实测校准）：
    # 附件垂直位置由 bbox 朝向箭头的**真实边**决定（charge_mirror=False——
    # 不带电荷圈镜像补偿，避免 OH⁻ 等"电荷在上"组件的 bbox 下界失真）：
    # 上方：bbox 真实底边距箭头 0.15；下方：bbox 真实顶边距箭头 0.15。
    # 边距推算：标签半高 ≈0.18 + 可逆箭头横线占位 0.05 ≈ 0.23，实测取 0.15
    # 目视合适（_SUP_GAP 可调）。水平以箭头中点为中心；同侧多附件按 bbox
    # 宽横向排布（间距 0.2），避免相互重叠。
    _SUP_GAP = 0.15
    _SUP_GAP_X = 0.2

    def _sup_group(ids):
        """同侧附件按 bbox 宽从中心向两侧排布，返回 [(中心偏移x, mol, bbox)]。"""
        placed = []
        for sid in ids:
            info = mols.get(sid)
            if info is None:
                continue
            bb = mol_visual_bbox(info["mol"], include_lone_pairs=False,
                                 charge_mirror=False)
            placed.append((bb[2] - bb[0], info["mol"], bb))
        if not placed:
            return []
        total_w = sum(w for w, _, _ in placed) + _SUP_GAP_X * (len(placed) - 1)
        x = -total_w / 2.0
        out = []
        for w, amol, bb in placed:
            out.append((x + w / 2.0, amol, bb))   # 中心偏移（相对箭头中点）
            x += w + _SUP_GAP_X
        return out

    for x1, x2, cond, yoff, a_kind, sup in main_arrows:
        # 主反应箭头（→/⇌/↔/⇒ 统一）：共享函数与 reaction/arrow 共用；
        # kind 由 [ARROW:type=...] 显式传入（旧 ⇌ 令牌由函数内部识别）
        lines.extend(main_arrow_lines(x1, x2, cond, y=-yoff, kind=a_kind))
        # 附件结构式：副反应物（+E）画在箭头上方、副产物（-F）下方——
        # 与普通结构式同一绘制管线（可参与机理箭头引用）
        if sup:
            mx = (x1 + x2) / 2.0
            up_items = [s.strip()[1:] for s in sup
                        if s.strip() and not s.strip().startswith("-")]
            dn_items = [s.strip()[1:] for s in sup
                        if s.strip() and s.strip().startswith("-")]
            for amx, amol, bb in _sup_group(up_items):
                cx = bb[0] + (bb[2] - bb[0]) / 2.0
                sy = -yoff + _SUP_GAP - bb[1]      # 真实底边距箭头 0.15
                lines.extend(molecule_scope_lines(
                    amol, (mx + amx - cx, sy), show_lone_pairs=False))
            for amx, amol, bb in _sup_group(dn_items):
                cx = bb[0] + (bb[2] - bb[0]) / 2.0
                sy = -yoff - _SUP_GAP - bb[3]      # 真实顶边距箭头 0.15
                lines.extend(molecule_scope_lines(
                    amol, (mx + amx - cx, sy), show_lone_pairs=False))

    # 加号实际坐标（y 取负：布局 yoff 向下为正，渲染取反）——供成键空位避让
    plus_xy = [(px, -yoff) for px, yoff in plus_positions]
    lines.extend(draw_mech_arrows(mols, _parse_mech_arrows(mech_specs),
                                  plus_positions=plus_xy))

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from core.tag_parser import parse_tags

    demos = [
        (
            "SN2 机理（标准范本风格：孤对电子起点 + 键中点断键箭头）",
            "[COMPOSITE:reaction_mech]"
            "[STRUCT:CCl,label=CH3Cl][PLUS][STRUCT:[OH-],id=nu,label=OH-]"
            "[RXNARROW]"
            "[STRUCT:CO,label=CH3OH][PLUS][STRUCT:[Cl-],label=Cl-]"
            "[MECHARROW:nu:0>r0:0][MECHARROW:r0:0-1>r0:1]"
            "[CONDITION:SN2]"
            "[/COMPOSITE]",
        ),
        (
            "乙醇→乙醚 SN2 机理（质子化物种 + 断键箭头 + 水 H₂O）",
            "[COMPOSITE:reaction_mech]"
            "[STRUCT:CCO,label=乙醇,id=nu][PLUS]"
            "[STRUCT:CC[OH2+],label=乙基氧鎓离子,id=pe]"
            "[RXNARROW:H2SO4,140°C]"
            "[STRUCT:CC[OH+]CC,label=质子化乙醚,id=ps][PLUS]"
            "[STRUCT:O,label=水,id=w]"
            "[MECHARROW:nu:2>pe:1][MECHARROW:pe:1-2>pe:2]"
            "[/COMPOSITE]",
        ),
        (
            "多步序列（row 布局，多个 RXNARROW，内联条件）",
            "[COMPOSITE:row]"
            "[STRUCT:C=C,label=乙烯][RXNARROW:H2O / H+]"
            "[STRUCT:CCO,label=乙醇][RXNARROW:CuO, Δ]"
            "[STRUCT:CC=O,label=乙醛]"
            "[/COMPOSITE]",
        ),
        (
            "鱼钩箭头（自由基加成到 π 键：三个鱼钩写全电子去向）",
            "[COMPOSITE:reaction_mech]"
            "[STRUCT:[Br],id=br][STRUCT:C=C,id=cc]"
            "[RXNARROW]"
            "[STRUCT:BrC[CH2]]"
            "[MECHARROW:br:0>>br:0+cc:0,cc:0-1>>br:0+cc:0,cc:0-1>>cc:1]"
            "[/COMPOSITE]",
        ),
        (
            "错误：reaction_mech 缺少 RXNARROW",
            "[COMPOSITE:reaction_mech][STRUCT:CCl][/COMPOSITE]",
        ),
        (
            "错误：未知布局",
            "[COMPOSITE:grid][STRUCT:CCl][/COMPOSITE]",
        ),
        (
            "容错：机理箭头引用未知 id（应跳过，不崩溃）",
            "[COMPOSITE:reaction_mech]"
            "[STRUCT:CCl][RXNARROW][STRUCT:CO]"
            "[MECHARROW:r9:0>r0:0]"
            "[/COMPOSITE]",
        ),
        (
            "numbering 标志：打开原子序号标注",
            "[COMPOSITE:reaction_mech,numbering]"
            "[STRUCT:CCl,label=CH3Cl][PLUS][STRUCT:[OH-],label=OH-]"
            "[RXNARROW:SN2]"
            "[STRUCT:CO,label=CH3OH][PLUS][STRUCT:[Cl-],label=Cl-]"
            "[MECHARROW:r1:0>r0:0,r0:0-1>r0:1]"
            "[/COMPOSITE]",
        ),
        (
            "energy 布局：SN2 势能面 + 三个驻点结构",
            "[COMPOSITE:energy]"
            "[ENERGY:0,108,-20]"
            "[STRUCT:CCl.[OH-],label=反应物,at=0]"
            "[STRUCT:CCl.[OH-],label=过渡态,at=1]"
            "[STRUCT:CO.[Cl-],label=产物,at=2]"
            "[/COMPOSITE]",
        ),
        (
            "共振式：苯的两个 Kekulé 式（row + 显式 [RESARROW]）",
            "[COMPOSITE:row]"
            "[STRUCT:C1=CC=CC=C1,label=Kekulé 式 I]"
            "[RESARROW]"
            "[STRUCT:C1C=CC=CC=1,label=Kekulé 式 II]"
            "[/COMPOSITE]",
        ),
        (
            "NEWLINE 多行：主结构在上、共振式在下",
            "[COMPOSITE:row]"
            "[STRUCT:CC(=O)[O-],label=羧酸根]"
            "[NEWLINE]"
            "[STRUCT:CC(=O)[O-]][RESARROW][STRUCT:CC([O-])=O]"
            "[/COMPOSITE]",
        ),
    ]

    for title, text in demos:
        print("=" * 60)
        print(title)
        print("=" * 60)
        tags = parse_tags(text)
        composite = next((t for t in tags if t.type == "COMPOSITE"), None)
        if composite is None:
            print("（未解析到 COMPOSITE 标记）")
            continue
        print(render_composite(composite.args[0], composite.args[1]))
        print()