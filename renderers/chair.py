# -*- coding: utf-8 -*-
r"""renderers/chair.py — [CHAIR] 标记渲染器：环己烷椅式构象。

几何规则（Klein《有机化学》SkillBuilder 4.9–4.11，instructions/cyclohexane.pdf）：
- 骨架五步法：宽 V → 60° 下降线 → 平行线 → 第二条 60° 线（底端同高）→ 连成环；
  正确的椅式含三对平行键（浅斜 ±σ 两对 + 陡斜 60° 一对）；
- 竖直键（axial）：严格竖直（90°），绕环交替上/下；
- 平伏键（equatorial）：永远指向环外、与同一碳的竖直键上下相反；
  1/3/4/6 号位与浅斜键平行（±σ），**2/5 号位（左右两侧中间碳）与水平呈
  60°**（2→120°、5→300°，fast_latex_test.tex 参考，20260818 修复）；
- 翻转（ring flip）= 两种镜像画法互换：骨架关于水平轴反射（前碳朝下↔朝上）、
  axial 交替翻转；equatorial 仍由质心判定外指。翻转对比画两张 CHAIR——
  正常一张 + flip 令牌一张（flip 令牌位置不限，如 [CHAIR:SMILES,1:ax,flip]）。

标记格式：
    [CHAIR:SMILES]                纯椅式骨架
    [CHAIR:SMILES,0:ax,2:eq]      取代基：取代基原子序号(SMILES 0 起):ax/eq
                                  （序号指向直接连在环碳上的那个取代基原子，
                                  上下/朝向由所连环碳的顶点规则自动确定；
                                  同一环碳上的两个取代基（偕二取代）可各给一条，
                                  如 [CHAIR:CC1(C)CCCCC1,0:ax,2:eq]——sp³ 环碳
                                  上必为一 ax 一 eq）
    [CHAIR:SMILES,flip,...]       镜像画法（翻转对比第二张；与正常画法同为
                                  合法的椅式，仅朝向不同）
示例：
    [CHAIR:BrC1CCCCC1,0:ax]       溴代环己烷（Br 的 SMILES 原子序号 0，竖直键）
    [CHAIR:CC1CCCCC1,0:eq]        甲基环己烷（甲基 C 序号 0，平伏键，更稳定）
    [CHAIR:CC1CCCCC1,flip,0:eq]   翻转对比第二张（镜像画法）
    [CHAIR:BrC1(Br)CCCCC1,0:ax,2:eq]  1,1-二溴环己烷（两 Br 序号 0/2，一 ax 一 eq）
"""

import math

from .mol_primitives import (_label_symbol_shift_x, atom_main_label,
                             format_chem_text, label_bond_margin, prepare_mol)

_SIGMA = 15.0       # 浅斜键与水平夹角（Klein 约束解出，视觉校准后可调）
_STEEP = 60.0       # 陡斜键与水平夹角（Klein 明文 60°）
_L = 1.5            # 环键长
_SUB_LEN = 1.1      # 取代基键线终点距环碳（20260818 微调：原 1.0 偏短、
                    # 试 1.5 与环键等长后视觉过长，定为 1.1）

# 顶点有向边方向（遍历序）：-σ, +σ, -60°, 180-σ, 180+σ, 180-60°——三对平行、
# 60° 键底端同高、闭环自洽
_EDGE_DIRS = (-_SIGMA, _SIGMA, -_STEEP,
              180.0 - _SIGMA, 180.0 + _SIGMA, 180.0 - _STEEP)

# 镜像画法（flip）：边方向全部取反（等价于顶点 y 反射），前碳朝上、开向不变；
# 与 _EDGE_DIRS 同构——三对平行、闭环自洽
_MIRROR_EDGE_DIRS = (_SIGMA, -_SIGMA, _STEEP,
                     180.0 + _SIGMA, 180.0 - _SIGMA, 180.0 + _STEEP)

# 各顶点竖直键朝上（True）/下（False）：绕环交替
_AXIAL_UP = (True, False, True, False, True, False)
_AXIAL_UP_MIRROR = tuple(not u for u in _AXIAL_UP)


def _chair_vertices(mirror: bool = False) -> list:
    """椅式六顶点坐标（遍历序）；mirror=True 时输出镜像画法。"""
    vs = [(0.0, 0.0)]
    dirs = _MIRROR_EDGE_DIRS if mirror else _EDGE_DIRS
    for d in dirs[:-1]:
        r = math.radians(d)
        x, y = vs[-1]
        vs.append((x + _L * math.cos(r), y + _L * math.sin(r)))
    return vs


def _equatorial_angle(pos: int, vx: float, vy: float, centroid,
                      axial_up: bool, mirror: bool = False) -> float:
    """平伏键方向：水平分量指向环外、竖直分量与竖直键相反。

    2/5 号位（左右两侧中间碳）与水平呈 60°——fast_latex_test.tex 参考：
    5 号位 300°（右下）、2 号位对称 120°（左上）；原"与浅斜骨架键平行
    （±15°）"在这两位视觉上接近水平、画法错误（20260818 修复）。flip
    画法取 y 反射（2: 120→240°、5: 300→60°），仍满足外指与轴向相反。
    其余位保持与浅斜骨架键平行（±σ，Klein 规则）。
    """
    if pos in (2, 5):
        ang = 120.0 if pos == 2 else 300.0
        return (360.0 - ang) % 360.0 if mirror else ang
    out_right = vx >= centroid[0]
    if axial_up:
        return (360.0 - _SIGMA) if out_right else (180.0 + _SIGMA)
    return _SIGMA if out_right else (180.0 - _SIGMA)


def _ring_carbon_of_substituent(mol, atom_idx: int, ring_set: set) -> int:
    """取代基原子所连环碳的原子序号；未连环碳返回 -1。

    从取代基原子出发找它在环中的邻居（该取代基只能是单键连到一个环碳
    上）；环碳是同一结构里的另一个可能邻居。返回该环碳序号。
    """
    for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        if nbr.GetIdx() in ring_set:
            return nbr.GetIdx()
    return -1


def _substituent_label(mol, sub_idx: int, flip: bool = False):
    """环上取代基原子（SMILES 序号 sub_idx）的标签。

    flip：键端在标签右侧时元素符号右移（OH→HO，Drawbacks 第 6 条），
    与 _label_flip_for 同口径（氯/溴等无 H 后缀不受影响）。
    返回标签字符串；该原子是环碳本身则返回 ("", None)（校验层已拦截）。
    """
    atom = mol.GetAtomWithIdx(sub_idx)
    if atom.GetAtomicNum() == 6 and atom.IsInRing():
        return "", None
    return atom_main_label(atom, flip=flip) or atom.GetSymbol(), atom


def _parse_spec(spec: str) -> tuple:
    """"flip,0:ax,2:eq" → (mirror, {0: "ax", 2: "eq"})；畸形项静默跳过（校验层已拦截）。

    键为取代基原子的 SMILES 序号（0 起），值域 ax/eq。
    """
    mirror = False
    out = {}
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() == "flip":
            mirror = True
            continue
        if ":" not in tok:
            continue
        pos, _, kind = tok.partition(":")
        kind = kind.strip().lower()
        kind = {"axial": "ax", "equatorial": "eq"}.get(kind, kind)
        if pos.strip().isdigit() and kind in ("ax", "eq"):
            out[int(pos.strip())] = kind
    return mirror, out


def chair_scope_lines(smiles: str, spec: str = ""):
    """椅式 scope 绘制行 + 视觉包围盒（不含 tikzpicture 包装）。

    供顶层 render_chair 与 COMPOSITE 容器内 mode=chair 组件复用——
    容器负责布局平移（scope shift）。成功返回 (lines, bbox)；
    失败返回 (None, 错误提示串)。
    """
    from utils.rdkit_utils import cyclohexane_ring
    mol = prepare_mol(smiles)
    if mol is None:
        return None, f"（椅式构象渲染失败：无效 SMILES「{smiles}」）"
    ring = cyclohexane_ring(mol)
    if ring is None:
        return None, (f"（椅式构象渲染失败：SMILES 中未找到环己烷六元环"
                      f"「{smiles}」）")
    ring_set = set(ring)
    mirror, subs = _parse_spec(spec)

    vs = _chair_vertices(mirror)
    axial_up_seq = _AXIAL_UP_MIRROR if mirror else _AXIAL_UP
    cx = sum(v[0] for v in vs) / 6.0
    cy = sum(v[1] for v in vs) / 6.0

    lines = []
    xs = [v[0] for v in vs]
    ys = [v[1] for v in vs]
    for i in range(6):
        x1, y1 = vs[i]
        x2, y2 = vs[(i + 1) % 6]
        lines.append(f"  \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")

    for sub_idx, kind in subs.items():
        # subs 键是取代基原子（SMILES 序号 0 起），经它找所连环碳定位顶点。
        # 环碳顶点 = ring.index(环碳)，即环位 1~6（与旧"环位"编号等价——
        # 之前是"按环碳序号排序的 1~6"，现在是"由取代基原子反查其环碳"）。
        ring_at = _ring_carbon_of_substituent(mol, sub_idx, ring_set)
        if ring_at < 0:
            continue            # 校验层已拦截，渲染兜底跳过
        pos = ring.index(ring_at) + 1          # 环位 1~6
        vx, vy = vs[pos - 1]
        up = axial_up_seq[(pos - 1) % 6]
        ang = (90.0 if up else -90.0) if kind == "ax" else \
            _equatorial_angle(pos, vx, vy, (cx, cy), up, mirror)
        r = math.radians(ang)
        # 取代基键线终点距环碳 _SUB_LEN=1.1（短于环键，避免过长遮挡）；
        # 标签中心再沿键方向外移 label_bond_margin（标签不压键线终点；
        # 环碳端是键线式顶点不标 C 无标签不收缩）
        bx, by = vx + _SUB_LEN * math.cos(r), vy + _SUB_LEN * math.sin(r)
        # 元素符号朝向：环碳相对取代基原子在右侧且水平占主导 → 翻转（OH→HO）
        # （与 _label_flip_for 同口径；methylene/carbon 标签不翻转）
        flip = (vx - bx > 0.05 and abs(vx - bx) > abs(vy - by))
        label, atom = _substituent_label(mol, sub_idx, flip)
        if not label:
            continue            # 该引用非取代基（校验层已拦截，渲染兜底跳过）
        m = label_bond_margin(label)
        # 元素符号中心落在键线延长点上（ex,ey = 键终点再外移 m 的沿键方向点），
        # 标签节点再按符号偏移平移——键延伸到元素符号而非整条标签中点
        # （与 STRUCT/COMPOSITE 的 label_node_pos 同一口径）
        ex, ey = bx + math.cos(r) * m, by + math.sin(r) * m
        sym = atom.GetSymbol()
        sym = sym[0].upper() + sym[1:]
        nx, ny = ex - _label_symbol_shift_x(label, sym), ey
        lines.append(f"  \\draw ({vx:.2f},{vy:.2f}) -- ({bx:.2f},{by:.2f});")
        lines.append(
            f"  \\node[fill=white, inner sep=1pt] at ({nx:.2f},{ny:.2f}) "
            f"{{{format_chem_text(label)}}};")
        xs += [bx, ex]
        ys += [by, ey]

    pad = 0.25   # 标签字符半径余量（包围盒用于布局避让）
    bbox = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    return lines, bbox


def render_chair(smiles: str, spec: str = "") -> str:
    """[CHAIR] 渲染：环己烷椅式骨架 + 可选取代基标注。失败返回错误提示。"""
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return "（椅式构象渲染失败：rdkit 未安装）"

    scope, err = chair_scope_lines(smiles, spec)
    if scope is None:
        return err
    return "\n".join(["\\begin{tikzpicture}"] + scope + ["\\end{tikzpicture}"])


if __name__ == "__main__":
    print("[1] 纯椅式骨架: [CHAIR:C1CCCCC1]")
    print(render_chair("C1CCCCC1"))
    print("\n[2] 溴代环己烷（Br 序号 0）竖直键: [CHAIR:BrC1CCCCC1,0:ax]")
    print(render_chair("BrC1CCCCC1", "0:ax"))
    print("\n[3] 甲基环己烷（甲基 C 序号 0）平伏键: [CHAIR:CC1CCCCC1,0:eq]")
    print(render_chair("CC1CCCCC1", "0:eq"))
    print("\n[4] 翻转镜像骨架（对比第二张）: [CHAIR:C1CCCCC1,flip]")
    print(render_chair("C1CCCCC1", "flip"))
    print("\n[5] 非环己烷（应降级提示）:")
    print(render_chair("CCO", "0:ax"))
    print("\n[6] 偕二取代（1,1-二溴，两 Br 序号 0/2，一 ax 一 eq）: "
          "[CHAIR:BrC1(Br)CCCCC1,0:ax,2:eq]")
    print(render_chair("BrC1(Br)CCCCC1", "0:ax,2:eq"))
