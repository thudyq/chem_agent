# -*- coding: utf-8 -*-
"""tests/test_mech_arrows.py — renderers.composite.draw_mech_arrows 单元测试（改进 1 模块化）。

机理箭头几何逻辑（p0/p1 定位、端点吸附避让）提取为独立函数后，
可脱离完整 COMPOSITE 渲染独立测试。需要真实 RDKit。
"""

import pytest

from renderers.composite import _parse_mech_arrows, draw_mech_arrows
from renderers.mol_primitives import prepare_mol, scale_mol_coords

rdkit = pytest.importorskip("rdkit", reason="rdkit 未安装，跳过机理箭头测试")


def _mols(*smiles):
    """SMILES 列表 → {id: {"mol", "shift", ...}} 组件表（r0, r1, ...）。

    字段与 composite._collect_components 填充的组件表一致
    （_mech_labeler 依赖 explicit_hs / xh / bonds / hbonds）。
    """
    mols = {}
    for i, smi in enumerate(smiles):
        m = prepare_mol(smi)
        scale_mol_coords(m, 0.8)
        mols[f"r{i}"] = {
            "mol": m,
            "shift": (0.0, 0.0),
            "explicit_hs": {},
            "xh": [], "bonds": [], "hbonds": [], "charges": [],
        }
    return mols


def test_draw_mech_arrows_basic():
    """合法进攻箭头：输出含红色贝塞尔弯箭头。"""
    mols = _mols("CCl", "CO")
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:0>r1:0"]))
    assert lines, "应产出箭头 TikZ 行"
    assert any("red" in ln and "controls" in ln for ln in lines)


def test_draw_mech_arrows_fishhook():
    """鱼钩箭头（>>）同样绘制。"""
    mols = _mols("CCl", "CO")
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:0>>r1:0"]))
    assert any("red" in ln and "controls" in ln for ln in lines)


def test_carbon_radical_fishhook_starts_at_single_electron():
    """碳自由基（·CH3）的鱼钩起点落在单电子点上（on_electron），
    而非 C 原子中心——回归锚点（mech_arrow_origin 的
    电子点分支原排除全部碳原子，碳自由基起点退化为原子中心）。"""
    from renderers.mol_primitives import (
        lone_pair_dot_groups, mech_arrow_origin,
    )
    mol = prepare_mol("[CH3]")
    scale_mol_coords(mol, 0.8)
    p = mech_arrow_origin(mol, "0", (0.0, 0.0), prefer_single=True)
    assert p[3] is True and p[4] is False      # 落在单电子点，未吸附标签
    # 起点沿 180° 槽位外移（单电子点之外 _ARROW_POINT_GAP）
    sx, sy = lone_pair_dot_groups(mol, 0)[1][0]
    assert abs(p[0] - sx) < 0.01 or abs(p[1] - sy) < 0.01 or p[0] < sx
    # 进攻箭头（非鱼钩）对碳：保持原子中心（碳无孤对，非供体）
    q = mech_arrow_origin(mol, "0", (0.0, 0.0), prefer_single=False)
    assert q[3] is False and abs(q[0]) < 0.01


def test_draw_mech_arrows_bond_break():
    """断键箭头（起点为 a-b 键中点）绘制，向下弯曲。"""
    mols = _mols("CCl", "CO")
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:0-1>r0:1"]))
    assert any("red" in ln and "controls" in ln for ln in lines)


def test_draw_mech_arrows_skip_unknown_id():
    """引用未知组件 id 的箭头跳过，不崩溃。"""
    mols = _mols("CCl")
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["ghost:0>r0:0"]))
    assert lines == []


def test_draw_mech_arrows_skip_invalid_point():
    """端点超出原子数 / 空端点跳过，不崩溃。"""
    mols = _mols("CCl")  # 2 个原子
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:9>r0:0"]))
    assert lines == []
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:0>r0:9"]))
    assert lines == []


def test_draw_mech_arrows_explicit_h_point():
    """显式 H 是真实原子参与编号（a#k 废弃），MECHARROW 直接用
    H 原子序号引用（CC([H])CC 的 2 号原子是 H）。
    """
    mols = _mols("CC([H])CC")   # 5 原子：C0-C1(H2)-C3-C4，2 号是显式 H
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:2>r0:0"]))
    assert lines, "H 原子序号端点应能定位并绘制箭头"


def test_draw_mech_arrows_explicit_h_bond_break():
    """显式 H 断键（夺 H）：起点为 C–H 键中点（a-b，如 1-2），向下弯。"""
    mols = _mols("CC([H])CC")
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:1-2>r0:0"]))
    assert lines, "C–H 键中点端点（1-2）应能绘制断键箭头"


def test_draw_mech_arrows_explicit_h_out_of_range_skips():
    """端点超出原子数 → 跳过，不崩溃。"""
    mols = _mols("CC([H])CC")  # 5 个原子
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:9>r0:0"]))
    assert lines == []


def test_draw_mech_arrows_explicit_h_as_target():
    """显式 H 原子作终点（如碱夺 H）：箭头尖端退让到 H 节点标签正方形
    （半边长 _LABEL_SQUARE_HALF=0.13）边缘外约 0.05（aim_end 切线退让，
    与纯原子终点同一机制），不压 H 占位。

    CC([H])CC 的 2 号是显式 H，MECHARROW 终点用 r0:2——定位到 H 节点方向，
    末端退到 H 节点正方形之外。
    """
    mols = _mols("C", "CC([H])CC")
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:0>r1:2"]))
    assert lines, "H 原子作终点应能定位到 H 节点并绘制箭头"
    # 终点坐标 = H 节点正方形外（半边长 0.13），间隙约 0.05（含斜向切线修正）
    import re
    from renderers.mol_primitives import atom_pos
    m = re.search(r"\.\.\s*\(([-\d.]+),([-\d.]+)\)", lines[0])
    assert m, "箭头应有终点坐标"
    tx, ty = float(m.group(1)), float(m.group(2))
    hx, hy = atom_pos(mols["r1"]["mol"], 2)
    hx += mols["r1"]["shift"][0]
    hy += mols["r1"]["shift"][1]
    dist = ((tx - hx) ** 2 + (ty - hy) ** 2) ** 0.5
    assert dist > 0.13, \
        f"尖端应退到 H 节点正方形（0.13）之外，实际 {dist:.3f}"
    gap = dist - 0.13
    assert 0.03 <= gap <= 0.20, \
        f"正方形外间隙 {gap:.3f} 应在合理范围（约 0.05±斜向修正）"


def test_draw_mech_arrows_bond_form_midpoint():
    """成键空白位终点（id:a+id:b，跨组件两原子中点）：正常绘制。"""
    mols = _mols("[Br]", "C=C")
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:0>>r0:0+r1:0"]))
    assert any("red" in ln and "controls" in ln for ln in lines)


def test_draw_mech_arrows_skip_invalid_midpoint():
    """成键空白位引用未知组件 / 越界原子 / 与键中点混用时跳过，不崩溃。"""
    mols = _mols("CCl")  # 2 个原子
    assert draw_mech_arrows(mols, _parse_mech_arrows(["r0:0>r0:0+ghost:0"])) == []
    assert draw_mech_arrows(mols, _parse_mech_arrows(["r0:0>r0:0+r0:9"])) == []
    assert draw_mech_arrows(mols, _parse_mech_arrows(["r0:0>r0:0-1+r0:0"])) == []


def test_draw_mech_arrows_empty():
    """空箭头列表返回空。"""
    assert draw_mech_arrows(_mols("CCl"), []) == []
