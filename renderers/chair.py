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
    [CHAIR:SMILES,1:ax,2:eq]      取代基：环位(1-6):ax/eq（上下由顶点规则自动确定）
    [CHAIR:SMILES,flip,...]       镜像画法（翻转对比第二张；与正常画法同为
                                  合法的椅式，仅朝向不同）
示例：
    [CHAIR:BrC1CCCCC1,1:ax]       溴代环己烷（1 位竖直键 Br）
    [CHAIR:CC1CCCCC1,1:eq]        甲基环己烷（1 位平伏键 CH3，更稳定构象）
    [CHAIR:CC1CCCCC1,flip,1:eq]   翻转对比第二张（镜像画法）
"""

import math

from .mol_primitives import atom_main_label, format_chem_text, label_bond_margin, prepare_mol

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


def _substituent_label(mol, ring_idx: int, ring_set: set) -> str:
    """环位原子的非环取代基标签（Br/CH3/OH 等）；无取代基返回 ""。"""
    for nbr in mol.GetAtomWithIdx(ring_idx).GetNeighbors():
        if nbr.GetIdx() not in ring_set:
            return atom_main_label(nbr) or nbr.GetSymbol()
    return ""


def _parse_spec(spec: str) -> tuple:
    """"flip,1:ax,2:eq" → (mirror, {1: "ax", 2: "eq"})；畸形项静默跳过（校验层已拦截）。"""
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


def render_chair(smiles: str, spec: str = "") -> str:
    """[CHAIR] 渲染：环己烷椅式骨架 + 可选取代基标注。失败返回错误提示。"""
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return "（椅式构象渲染失败：rdkit 未安装）"

    from utils.rdkit_utils import cyclohexane_ring
    mol = prepare_mol(smiles)
    if mol is None:
        return f"（椅式构象渲染失败：无效 SMILES「{smiles}」）"
    ring = cyclohexane_ring(mol)
    if ring is None:
        return f"（椅式构象渲染失败：SMILES 中未找到环己烷六元环「{smiles}」）"
    ring_set = set(ring)
    mirror, subs = _parse_spec(spec)

    vs = _chair_vertices(mirror)
    axial_up_seq = _AXIAL_UP_MIRROR if mirror else _AXIAL_UP
    cx = sum(v[0] for v in vs) / 6.0
    cy = sum(v[1] for v in vs) / 6.0

    lines = ["\\begin{tikzpicture}"]
    for i in range(6):
        x1, y1 = vs[i]
        x2, y2 = vs[(i + 1) % 6]
        lines.append(f"  \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")

    for pos, kind in subs.items():
        if not 1 <= pos <= 6:
            continue
        vx, vy = vs[pos - 1]
        up = axial_up_seq[(pos - 1) % 6]
        ang = (90.0 if up else -90.0) if kind == "ax" else \
            _equatorial_angle(pos, vx, vy, (cx, cy), up, mirror)
        r = math.radians(ang)
        label = _substituent_label(mol, ring[pos - 1], ring_set)
        if not label:
            continue            # 该环位无取代基（校验层已拦截，渲染兜底跳过）
        # 取代基键线终点距环碳 _SUB_LEN=1.1（短于环键，避免过长遮挡）；
        # 标签中心再沿键方向外移 label_bond_margin（标签不压键线终点；
        # 环碳端是键线式顶点不标 C 无标签不收缩）
        bx, by = vx + _SUB_LEN * math.cos(r), vy + _SUB_LEN * math.sin(r)
        m = label_bond_margin(label)
        ex, ey = bx + math.cos(r) * m, by + math.sin(r) * m
        lines.append(f"  \\draw ({vx:.2f},{vy:.2f}) -- ({bx:.2f},{by:.2f});")
        lines.append(
            f"  \\node[fill=white, inner sep=1pt] at ({ex:.2f},{ey:.2f}) "
            f"{{{format_chem_text(label)}}};")

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] 纯椅式骨架: [CHAIR:C1CCCCC1]")
    print(render_chair("C1CCCCC1"))
    print("\n[2] 溴代环己烷 1 位竖直键: [CHAIR:BrC1CCCCC1,1:ax]")
    print(render_chair("BrC1CCCCC1", "1:ax"))
    print("\n[3] 甲基环己烷 1 位平伏键: [CHAIR:CC1CCCCC1,1:eq]")
    print(render_chair("CC1CCCCC1", "1:eq"))
    print("\n[4] 翻转镜像骨架（对比第二张）: [CHAIR:C1CCCCC1,flip]")
    print(render_chair("C1CCCCC1", "flip"))
    print("\n[5] 非环己烷（应降级提示）:")
    print(render_chair("CCO", "1:ax"))
