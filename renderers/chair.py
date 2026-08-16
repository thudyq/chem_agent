# -*- coding: utf-8 -*-
r"""renderers/chair.py — [CHAIR] 标记渲染器：环己烷椅式构象。

几何规则（Klein《有机化学》SkillBuilder 4.9–4.11，instructions/cyclohexane.pdf）：
- 骨架五步法：宽 V → 60° 下降线 → 平行线 → 第二条 60° 线（底端同高）→ 连成环；
  正确的椅式含三对平行键（浅斜 ±σ 两对 + 陡斜 60° 一对）；
- 竖直键（axial）：严格竖直（90°），绕环交替上/下；
- 平伏键（equatorial）：与浅斜键平行（±σ），永远指向环外，与同一碳的
  竖直键上下相反；
- 翻转（ring flip）= 骨架左右镜像 + axial↔equatorial 互换 + 上下不变
  （v1 不做，画两张对比即可）。

标记格式：
    [CHAIR:SMILES]                纯椅式骨架
    [CHAIR:SMILES,1:ax,2:eq]      取代基：环位(1-6):ax/eq（上下由顶点规则自动确定）
示例：
    [CHAIR:BrC1CCCCC1,1:ax]       溴代环己烷（1 位竖直键 Br）
    [CHAIR:CC1CCCCC1,1:eq]        甲基环己烷（1 位平伏键 CH3，更稳定构象）
"""

import math

from .mol_primitives import atom_main_label, format_chem_text, prepare_mol

_SIGMA = 15.0       # 浅斜键与水平夹角（Klein 约束解出，视觉校准后可调）
_STEEP = 60.0       # 陡斜键与水平夹角（Klein 明文 60°）
_L = 1.5            # 环键长
_SUB_LEN = 1.0      # 取代基键长

# 顶点有向边方向（遍历序）：-σ, +σ, -60°, 180-σ, 180+σ, 180-60°——三对平行、
# 60° 键底端同高、闭环自洽
_EDGE_DIRS = (-_SIGMA, _SIGMA, -_STEEP,
              180.0 - _SIGMA, 180.0 + _SIGMA, 180.0 - _STEEP)

# 各顶点竖直键朝上（True）/下（False）：绕环交替
_AXIAL_UP = (True, False, True, False, True, False)


def _chair_vertices() -> list:
    """椅式六顶点坐标（遍历序）。"""
    vs = [(0.0, 0.0)]
    for d in _EDGE_DIRS[:-1]:
        r = math.radians(d)
        x, y = vs[-1]
        vs.append((x + _L * math.cos(r), y + _L * math.sin(r)))
    return vs


def _equatorial_angle(vx: float, vy: float, centroid, axial_up: bool) -> float:
    """平伏键方向：水平分量指向环外、竖直分量与竖直键相反、斜率 ±σ
    （与浅斜骨架键平行，Klein 规则）。"""
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


def _parse_spec(spec: str) -> dict:
    """"1:ax,2:eq" → {1: "ax", 2: "eq"}；畸形项静默跳过（校验层已拦截）。"""
    out = {}
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if ":" not in tok:
            continue
        pos, _, kind = tok.partition(":")
        kind = kind.strip().lower()
        kind = {"axial": "ax", "equatorial": "eq"}.get(kind, kind)
        if pos.strip().isdigit() and kind in ("ax", "eq"):
            out[int(pos.strip())] = kind
    return out


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
    subs = _parse_spec(spec)

    vs = _chair_vertices()
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
        up = _AXIAL_UP[(pos - 1) % 6]
        ang = (90.0 if up else -90.0) if kind == "ax" else \
            _equatorial_angle(vx, vy, (cx, cy), up)
        r = math.radians(ang)
        ex, ey = vx + _SUB_LEN * math.cos(r), vy + _SUB_LEN * math.sin(r)
        label = _substituent_label(mol, ring[pos - 1], ring_set)
        if not label:
            continue            # 该环位无取代基（校验层已拦截，渲染兜底跳过）
        lines.append(f"  \\draw ({vx:.2f},{vy:.2f}) -- ({ex:.2f},{ey:.2f});")
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
    print("\n[4] 非环己烷（应降级提示）:")
    print(render_chair("CCO", "1:ax"))
