# -*- coding: utf-8 -*-
"""tests/test_composite.py — [COMPOSITE] 容器式复合标记渲染器单元测试。

运行: python -m pytest tests/test_composite.py -v
"""

import math
import re

import pytest

from core.tag_parser import parse_tags
from core.tag_injector import inject_tags_into_text
from renderers.composite import render_composite
from renderers.mol_primitives import format_chem_text
from renderers.registry import RENDERER_REGISTRY

rdkit = pytest.importorskip("rdkit", reason="rdkit 未安装，跳过渲染测试")


SN2_DEMO = (
    "[COMPOSITE:reaction_mech]"
    "[STRUCT:CCl,label=CH3Cl]"
    "[PLUS]"
    "[STRUCT:[OH-],id=nu,label=OH-]"
    "[RXNARROW]"
    "[STRUCT:CO,label=CH3OH]"
    "[PLUS]"
    "[STRUCT:[Cl-],label=Cl-]"
    "[MECHARROW:nu:0>r0:0]"
    "[MECHARROW:r0:0-1>r0:1]"
    "[CONDITION:SN2]"
    "[/COMPOSITE]"
)


def _render(text: str) -> str:
    tags = parse_tags(text)
    composite = next((t for t in tags if t.type == "COMPOSITE"), None)
    assert composite is not None, "未解析到 COMPOSITE 标记"
    return render_composite(composite.args[0], composite.args[1])


def test_sn2_full_scene():
    """正例1：SN2 完整机理场景（4 组分 + 2 机理箭头 + 条件）。"""
    out = _render(SN2_DEMO)
    assert out.startswith("\\begin{tikzpicture}")
    assert out.endswith("\\end{tikzpicture}")
    assert len(re.findall(r"\\node at \([-\d.]+,[-\d.]+\) \{\$\+\$\}", out)) == 2   # 两个 PLUS
    assert "SN$_2$" in out                # CONDITION 标注到主箭头（数字自动下标）
    assert out.count("\\draw[->, very thick]") == 1   # 一个主反应箭头
    assert out.count("\\draw[->, thick, red]") == 2   # 两条双电子弯箭头
    assert "CH$_{3}$" in out              # 非环碳按结构简式写出
    # 电荷为右上角圆圈节点（规范第 3 条），OH- 与 Cl- 各一个
    assert out.count("\\node[draw, circle") == 2
    assert out.count("{OH}") >= 1
    assert "\\node[font=\\tiny, gray" not in out      # 默认不显示原子序号
    assert "\\node[below]" not in out     # 纯化学式 label 不重复显示（分子本身已是简式）
    # 孤对电子点：OH-(3对)+CH3Cl的Cl(3对)+CH3OH的O(2对)+Cl-(4对)=12对=24点
    assert out.count("\\fill") == 24


def test_cjk_label_caption_shown():
    """中文名称/角色标注仍显示在分子下方。"""
    out = _render(
        "[COMPOSITE:reaction_mech]"
        "[STRUCT:CCl,label=底物][RXNARROW][STRUCT:CO,label=产物]"
        "[/COMPOSITE]"
    )
    assert "底物" in out and "产物" in out
    assert out.count("\\node[below]") == 2


def test_radical_label_not_duplicated():
    """自由基 label（Cl·、·CH3）不重复显示在分子下方——含 · 的化学式
    不是角色标注（分子本身已画 Cl/CH3 + 单电子点）。"""
    out = _render(
        "[COMPOSITE:reaction_mech]"
        "[STRUCT:[Cl],id=cl,label=Cl·][RXNARROW][STRUCT:[CH3],id=me,label=·CH3]"
        "[MECHARROW:cl:0>>me:0]"
        "[/COMPOSITE]"
    )
    assert out.count("\\node[below]") == 0          # 无重复 label
    assert "Cl·" not in out or "Cl·" not in [ln for ln in out.splitlines() if "below" in ln]


def test_long_label_wraps_multiline():
    """C2 标签自动换行：长中文标签在渲染输出中分多行 + align=center。"""
    out = _render(
        "[COMPOSITE:reaction_mech]"
        "[STRUCT:CCl,label=质子化乙醇的反应中间体][RXNARROW][STRUCT:CO,label=产物]"
        "[/COMPOSITE]"
    )
    # 长标签拆为多行（含 \\\\ 行分隔）且节点启用 align=center
    assert "质子化乙醇的\\\\反应中间体" in out or "\\\\" in out
    assert "align=center" in out
    # 中文标签仍显示
    assert "质子化乙醇" in out and "反应中间体" in out


def test_short_label_no_align_center():
    """短标签（≤ 3.5 宽）不换行，节点不引入 align=center。"""
    out = _render(
        "[COMPOSITE:reaction_mech]"
        "[STRUCT:CCl,label=底物][RXNARROW][STRUCT:CO,label=产物]"
        "[/COMPOSITE]"
    )
    assert "align=center" not in out


def test_bond_origin_arrow_bends_down():
    """键中点出发的断键箭头向下弯，孤对电子进攻箭头向上弯。"""
    out = _render(SN2_DEMO)
    controls = re.findall(r"controls \(([-\d.]+),([-\d.]+)\)", out)
    assert len(controls) == 2
    cys = [float(cy) for _, cy in controls]
    assert any(cy > 0 for cy in cys)   # nu:0>r0:0 进攻箭头向上
    assert any(cy < 0 for cy in cys)   # r0:0-1>r0:1 断键箭头向下


def test_numbering_flag():
    """正例4：numbering 标志打开原子序号标注。"""
    out = _render(
        "[COMPOSITE:reaction_mech,numbering]"
        "[STRUCT:CCl][RXNARROW][STRUCT:CO]"
        "[/COMPOSITE]"
    )
    assert "\\node[font=\\tiny, gray" in out


def test_lone_pair_origin_offset():
    """进攻箭头起点比孤对电子点高 0.05（点距 0.24 + 空隙 0.05，绕元素符号中心）。"""
    out = _render(SN2_DEMO)
    o_pos = _resolve_node_positions(out, "OH")
    assert o_pos
    ox, oy = o_pos[0]
    # "OH" 后缀宽 1 字符 -> 符号中心左移 0.13；正上方槽位，点距 0.24 + 0.05
    expected = (ox - 0.13, oy + 0.29)
    m = re.search(r"\\draw\[->, thick, red\] \(([-\d.]+),([-\d.]+)\)", out)
    assert m is not None
    start = (float(m.group(1)), float(m.group(2)))
    assert abs(start[0] - expected[0]) < 0.01
    assert abs(start[1] - expected[1]) < 0.01


def test_arrow_aim_end_and_bond_break_inset():
    """字母标签（C）箭头终点退到标签正方形（边长 0.26）边缘外 0.05、沿自然切线
    指向原子中心（终点坐标适配切线方向）；σ 断键起点 inset 加在纵坐标
    （向下 0.05），不再沿箭头方向（向右）。"""
    out = _render(SN2_DEMO)
    curves = re.findall(
        r"\\draw\[->, thick, red\] \(([-\d.]+),([-\d.]+)\) .. controls "
        r"\(([-\d.]+),([-\d.]+)\) .. \(([-\d.]+),([-\d.]+)\);", out)
    assert len(curves) == 2
    atk, brk = curves[0], curves[1]
    # 进攻箭头终点在 CH3 标签正方形（中心=node 中心左移 0.26）边缘外 0.05
    c_pos = _resolve_node_positions(out, "CH$_{3}$")[0]
    c_sym_x, c_sym_y = c_pos[0] - 0.26, c_pos[1]
    atk_edge = max(abs(float(atk[4]) - c_sym_x),
                   abs(float(atk[5]) - c_sym_y)) - 0.13
    assert 0.0 <= atk_edge <= 0.11, \
        f"进攻箭头终点应距 C 标签正方形边缘 0.05: ({atk[4]},{atk[5]})"
    # 断键起点 = C—Cl 键线中点（修剪后，局部 0.075 + shift 0.99 = 1.065）
    # 纵向下 inset 0.05 → (1.065,-0.05)
    assert abs(float(brk[0]) - 1.065) < 0.01
    assert abs(float(brk[1]) - (-0.05)) < 0.01, \
        f"断键起点未纵向下 inset: ({brk[0]},{brk[1]})"


def test_lone_pairs_orthogonal_placement():
    """OH- 的孤对电子点正交摆放（上/左/下，无斜向），且绕 O 符号中心。"""
    out = _render(SN2_DEMO)
    o_pos = _resolve_node_positions(out, "OH")
    assert o_pos
    ox, oy = o_pos[0]
    cx, cy = ox - 0.13, oy          # 元素符号中心（主标签 "OH" 后缀 1 字符）
    scope_with_o = re.search(
        r"\\begin\{scope\}\[shift=\{\(([-\d.]+),([-\d.]+)\)\}\]"
        r"(?:(?!\\end\{scope\}).)*\{OH\}.*?\\end\{scope\}", out, re.DOTALL,
    )
    assert scope_with_o
    sx, sy = float(scope_with_o.group(1)), float(scope_with_o.group(2))
    dots = re.findall(r"\\fill \(([-\d.]+),([-\d.]+)\)", scope_with_o.group(0))
    assert len(dots) == 6            # 3 对
    for dxs, dys in dots:
        dx = sx + float(dxs) - cx
        dy = sy + float(dys) - cy
        # 正交摆放：偏移主轴对齐（|dx| 或 |dy| 小于点对半距）
        assert abs(dx) < 0.1 or abs(dy) < 0.1, f"斜向电子点: ({dx:.2f},{dy:.2f})"
        # 点距 0.24（容差含点对半距 0.055）
        assert abs(abs(dx) + abs(dy) - 0.24) < 0.08


def _resolve_node_positions(out: str, label: str) -> list:
    """解析输出中所有 {label} 节点的全局坐标（局部坐标 + 所属 scope 的 shift）。"""
    positions = []
    for m in re.finditer(
        r"\\begin\{scope\}\[shift=\{\(([-\d.]+),([-\d.]+)\)\}\](.*?)\\end\{scope\}",
        out, re.DOTALL,
    ):
        sx, sy, body = float(m.group(1)), float(m.group(2)), m.group(3)
        for nm in re.finditer(
            r"\\node\[fill=white[^\]]*\] at \(([-\d.]+),([-\d.]+)\) "
            r"\{((?:[^{}]|\{[^{}]*\})*)\}", body
        ):
            if nm.group(3) == label:
                positions.append((sx + float(nm.group(1)), sy + float(nm.group(2))))
    return positions


def test_molecules_wrapped_in_scopes():
    """每个分子组件封装在独立 scope 中（局部坐标）。"""
    out = _render(SN2_DEMO)
    scopes = re.findall(r"\\begin\{scope\}\[shift=", out)
    assert len(scopes) == 4                      # 4 个 STRUCT 组件
    assert out.count("\\end{scope}") == 4
    # scope 内为局部坐标：键/标签不再加全局偏移（CH3Cl 的 C 在局部原点附近）
    first_scope = re.search(
        r"\\begin\{scope\}\[shift=\{\(([-\d.]+),([-\d.]+)\)\}\](.*?)\\end\{scope\}",
        out, re.DOTALL,
    )
    assert "CH$_{3}$" in first_scope.group(3)


def test_arrow_endpoints_near_atoms():
    """箭头终点距"标签所占位置"边缘 0.05（统一 label_gap 规则）：原子标签按
    边长 0.26 正方形（中心=符号中心）计算，终点在正方形边缘外 0.05（浮动
    0~0.10），沿切线指向原子中心。"""
    out = _render(SN2_DEMO)
    # 键中点箭头终点 -> CH3Cl 的 Cl；进攻箭头终点 -> CH3Cl 的 C
    c_positions = _resolve_node_positions(out, "CH$_{3}$")
    cl_positions = _resolve_node_positions(out, "Cl")
    assert c_positions and cl_positions
    c_pos = c_positions[0]
    cl_pos = cl_positions[0]
    ends = re.findall(r"\.\. \(([-\d.]+),([-\d.]+)\);", out)
    assert len(ends) == 2
    attack_end = (float(ends[0][0]), float(ends[0][1]))
    bond_end = (float(ends[1][0]), float(ends[1][1]))
    # 进攻箭头终点：距 C 标签正方形（半 0.13）边缘 0.05（浮动 0~0.10）。
    # 用 max(|dx|,|dy|) 测"到轴对齐正方形边缘"的距离——方向无关
    # （轴向 0.18、对角 0.23 都落在 [0.13, 0.13+0.10] 边缘带外）。
    c_sym = (c_pos[0] - 0.26, c_pos[1])
    atk_edge = max(abs(attack_end[0] - c_sym[0]),
                   abs(attack_end[1] - c_sym[1])) - 0.13
    assert 0.0 <= atk_edge <= 0.11, \
        f"进攻箭头终点应距 C 标签正方形边缘 0.05: {atk_edge:.2f}"
    assert attack_end[1] > c_pos[1] + 0.1, \
        f"进攻箭头终点应在标签上方: {attack_end}"
    # 断键箭头终点（Cl）：起点=键中点距 Cl 0.75 > 正方形 0.13 → 起点在正方形
    # 外，末端退让到正方形边缘外 0.05（距 Cl 符号中心 ≈0.18~0.23）
    bond_dist = math.hypot(bond_end[0] - cl_pos[0], bond_end[1] - cl_pos[1])
    assert 0.13 < bond_dist < 0.30, \
        f"断键箭头终点应距 Cl 标签正方形边缘 0.05: {bond_dist:.2f}"


def test_format_chem_text():
    """化学文本排版：数字下标、尾部电荷上标、已排版文本跳过、加热符号转数学模式。"""
    assert format_chem_text("CH3Cl") == "CH$_3$Cl"
    assert format_chem_text("OH-") == "OH$^{-}$"
    assert format_chem_text("H2SO4, 浓HNO3") == "H$_2$SO$_4$, 浓HNO$_3$"
    assert format_chem_text("SO42-") == "SO$_4$$^{2-}$"
    assert format_chem_text("NH4+") == "NH$_4$$^{+}$"
    assert format_chem_text("OH$^-$") == "OH$^-$"
    assert format_chem_text("") == ""
    # 加热符号（Drawbacks 第 7 条）：△(U+25B3) lmroman 缺字形 → 数学模式
    assert format_chem_text("CuO, △") == "CuO, $\\triangle$"
    assert format_chem_text("CuO, Δ") == "CuO, $\\Delta$"
    assert format_chem_text("△") == "$\\triangle$"
    # 希腊字母统一转数学模式（lmroman 文本字体缺字形）：hν 光照、α/β/γ、
    # π/σ/ω 及有大写命令的大写；无命令大写（Α Ε）保持原样
    assert format_chem_text("hν") == "h$\\nu$"
    assert format_chem_text("α-碳") == "$\\alpha$-碳"
    assert format_chem_text("β-消除") == "$\\beta$-消除"
    assert format_chem_text("δ") == "$\\delta$"
    assert format_chem_text("π 键") == "$\\pi$ 键"
    assert format_chem_text("σ 键") == "$\\sigma$ 键"
    assert format_chem_text("ω-3") == "$\\omega$-3"
    assert format_chem_text("ΔH") == "$\\Delta$H"
    assert format_chem_text("AΑB") == "AΑB"      # 无命令大写不转换


def test_row_layout_multi_step():
    """正例2：row 布局多步序列（多个内联条件 RXNARROW）。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:C=C,label=乙烯][RXNARROW:H2O / H+]"
        "[STRUCT:CCO,label=乙醇][RXNARROW:CuO, Δ]"
        "[STRUCT:CC=O,label=乙醛]"
        "[/COMPOSITE]"
    )
    assert out.count("\\draw[->, very thick]") == 2
    assert "H$_2$O / H$^+$" in out or "H2O / H+" in out or "H$_2$O" in out
    assert "乙烯" in out and "乙醛" in out


def test_row_layout_four_step_sequence():
    """R-5：A→B→C→D 四步序列（3 个带内联条件的主箭头，分子按序排列）。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:C=C,label=乙烯]"
        "[RXNARROW:H2O / H+]"
        "[STRUCT:CCO,label=乙醇]"
        "[RXNARROW:CuO, Δ]"
        "[STRUCT:CC=O,label=乙醛]"
        "[RXNARROW:O2]"
        "[STRUCT:CC(=O)O,label=乙酸]"
        "[/COMPOSITE]"
    )
    assert out.count("\\begin{scope}[shift=") == 4
    assert out.count("\\draw[->, very thick]") == 3
    assert "H$_2$O" in out and "CuO, $\\Delta$" in out and "O$_2$" in out
    scopes = re.findall(r"\\begin\{scope\}\[shift=\{\(([-\d.]+),", out)
    xs = [float(x) for x in scopes]
    assert xs == sorted(xs)                     # 分子按序列从左到右递增


def test_fishhook_arrows():
    """正例3：鱼钩箭头（单电子）生成半边 barb。自由基加成到 π 键（范本三鱼钩）：
    ①Br· 单电子→Br 与近端碳 C0 之间的空白成键位置（上弯）；②π 键一个电子→
    同一空白位置（下弯，与①汇聚、共同形成 C—Br 键，两钩尖留微小间隙）；
    ③π 键另一个电子→远端碳 C1 原子（下弯，生成新自由基）。
    成键钩尖汇聚于两原子间空白中点而非原子标签；③ 退到 C1 的 0.26 正方形
    边缘外 0.05（≈0.18~0.23），不压标签。"""
    out = _render(
        "[COMPOSITE:reaction_mech]"
        "[STRUCT:[Br],id=br][STRUCT:C=C,id=cc]"
        "[RXNARROW]"
        "[STRUCT:BrC[CH2]]"
        "[MECHARROW:br:0>>br:0+cc:0,cc:0-1>>br:0+cc:0,cc:0-1>>cc:1]"
        "[/COMPOSITE]"
    )
    assert "\\draw[thick, red]" in out              # 鱼钩曲线（无 -> 全箭头）
    assert "\\draw[->, thick, red]" not in out
    assert out.count("\\draw[thick, red]") == 6     # 3 钩 ×（曲线 + 半箭头 barb）
    curves = re.findall(
        r"\\draw\[thick, red\] \(([-\d.]+),([-\d.]+)\) .. controls "
        r"\(([-\d.]+),([-\d.]+)\) .. \(([-\d.]+),([-\d.]+)\);", out)
    assert len(curves) == 3                         # 三鱼钩写全电子去向
    # 弯向：Br· 单电子钩上弯（控制点 y>0），两个 π 键钩下弯（y<0）
    cys = [float(c[3]) for c in curves]
    assert sum(1 for cy in cys if cy > 0) == 1, f"Br· 钩应上弯: {cys}"
    assert sum(1 for cy in cys if cy < 0) == 2, f"π 键两钩应下弯: {cys}"
    # 取反应物 C=C 的两个 CH₂ 与 Br·（产物 BrC[CH2] 的 CH₂/Br x≈6.6+ 需排除）
    ch2 = [p for p in _resolve_node_positions(out, "CH$_{2}$")
           if 1.5 < p[0] < 4.0]
    assert len(ch2) == 2
    brs = [p for p in _resolve_node_positions(out, "Br") if p[0] < 1.5]
    assert len(brs) == 1
    c0 = min(ch2, key=lambda p: p[0])               # 近端碳（Br· 加成侧）
    c1 = max(ch2, key=lambda p: p[0])               # 远端碳（新自由基）
    # 源端：两钩同源于 C=C 双键中点（y<0），一钩源于 Br· 单电子点（y≈0）
    mid_x = (c0[0] + c1[0]) / 2
    srcs = [(float(c[0]), float(c[1])) for c in curves]
    bond_srcs = [s for s in srcs if s[1] < -0.01]
    br_srcs = [s for s in srcs if s[1] >= -0.01]
    assert len(bond_srcs) == 2, f"应有两钩同源于 π 键中点: {srcs}"
    for s in bond_srcs:
        assert abs(s[0] - mid_x) < 0.6, f"π 键钩源端不在双键中点: {s}"
    assert len(br_srcs) == 1 and br_srcs[0][0] < 1.5, \
        f"Br· 钩源端不在单电子点: {br_srcs}"
    # 尖端（按 x 排序：左侧两钩汇聚于成键位，最右钩指向 C1）：
    # ①② 钩尖汇聚于 Br 与 C0 之间的空白成键位置（两原子中点，容差 0.45），
    # 不指向 C0 标签（距其符号中心 >0.30），两钩尖留有微小间隙；
    # ③ 钩尖指向 C1 原子：距符号中心（node 中心左移 0.26、含下标上移
    # 0.025）的 0.26 正方形边缘外 0.05（≈0.18~0.23），不压标签。
    bond_mid = ((brs[0][0] + c0[0]) / 2, (brs[0][1] + c0[1]) / 2)
    c0_sym = (c0[0] - 0.26, c0[1] + 0.025)
    c1_sym = (c1[0] - 0.26, c1[1] + 0.025)
    tips = sorted((float(c[4]), float(c[5])) for c in curves)
    conv_tips, atom_tip = tips[:2], tips[2]
    for t in conv_tips:
        d_mid = math.hypot(t[0] - bond_mid[0], t[1] - bond_mid[1])
        d_sym = math.hypot(t[0] - c0_sym[0], t[1] - c0_sym[1])
        assert d_mid < 0.45, f"成键钩尖未汇聚于空白成键位置: {t}"
        assert d_sym > 0.30, f"成键钩尖不应指向 C0 标签: {t}"
    gap = math.hypot(conv_tips[0][0] - conv_tips[1][0],
                     conv_tips[0][1] - conv_tips[1][1])
    assert 0.05 < gap < 0.6, f"汇聚钩尖应留有微小间隙: {conv_tips}"
    d1 = math.hypot(atom_tip[0] - c1_sym[0], atom_tip[1] - c1_sym[1])
    assert 0.13 < d1 < 0.30, f"π 键钩尖端应距 C1 正方形边缘 0.05: {atom_tip}"
    assert atom_tip[1] < c1[1], f"π 键钩尖端应在标签下方: {atom_tip}"


def test_mech_arrow_origin_label_snap():
    """端点定位单元测试（复用 lewis/charge 的 symbol_center）：带标签的原子
    端点返回元素符号中心（on_label，由 aim_end 退让到正方形边缘外 0.05）；
    双键源端在键中点法线方向距外侧杠 0.05；无标签保持原子中心。"""
    import renderers.mol_primitives as mp
    mol = mp.prepare_mol("CCl")
    ax, ay = mp.atom_pos(mol, 1)
    # 目标端（lone_pair_offset=False）：杂原子 Cl 返回元素符号中心
    # （不沿入射偏移；切线退让由 mech_arrow_tikz 的 aim_end 处理）
    p = mp.mech_arrow_origin(mol, "1", lone_pair_offset=False,
                             toward=(3.0, 0.0), labeler=mp.atom_main_label)
    assert p[4] is True
    scx, scy = mp.symbol_center(mol, 1)
    assert abs(p[0] - scx) < 0.01 and abs(p[1] - scy) < 0.01
    # 碳原子（CH₃）：目标端返回元素符号中心（C 字形位置，x 左移 2×0.13=0.26），
    # 供 mech_arrow_tikz 沿末端切线内缩、尖端指向原子中心
    cx, cy = mp.atom_pos(mol, 0)
    scx, scy = mp.symbol_center(mol, 0)
    pc = mp.mech_arrow_origin(mol, "0", lone_pair_offset=False,
                              toward=(3.0, 0.0), labeler=mp.atom_main_label)
    assert pc[4] is True
    assert abs(pc[0] - scx) < 0.01, \
        f"碳端点未与符号中心对齐: {pc[0]:.2f} vs {scx:.2f}"
    assert abs(pc[1] - scy) < 0.01
    # 双键源端按"靠外杠"：键源（bend_side=-1 向下）取 y 最小杠（基准线，
    # 乙烯 CH₂=CH₂ 两端标签对称 → 修剪中点=原子中点），再向下 0.05。
    alkene = mp.prepare_mol("C=C")
    bx, by = mp.atom_pos(alkene, 0)
    ex, ey = mp.atom_pos(alkene, 1)
    pb = mp.mech_arrow_origin(alkene, "0-1", toward=(0.0, 0.0), bend_side=-1.0)
    assert pb[2] is True
    dx, dy = ex - bx, ey - by
    L = math.hypot(dx, dy)
    px, py = -dy / L, dx / L
    assert abs(pb[0] - ((bx + ex) / 2 + px * -0.05)) < 0.01
    assert abs(pb[1] - ((by + ey) / 2 + py * -0.05)) < 0.01
    # 源端默认 lone_pair_offset=True：Cl 有孤对电子，落在电子点上（不吸附）
    e = mp.mech_arrow_origin(mol, "1", toward=(3.0, 0.0))
    assert e[3] is True and e[4] is False
    # 不传 labeler：返回原子中心，on_label=False
    q = mp.mech_arrow_origin(mol, "1", lone_pair_offset=False,
                             toward=(3.0, 0.0))
    assert q[4] is False
    assert abs(q[0] - ax) < 0.01 and abs(q[1] - ay) < 0.01
    # 环上碳原子无标签（atom_main_label 返回 None）：不吸附，保持原子中心
    ring = mp.prepare_mol("c1ccccc1")
    rx, ry = mp.atom_pos(ring, 0)
    w = mp.mech_arrow_origin(ring, "0", lone_pair_offset=False,
                             toward=(5.0, 0.0), labeler=mp.atom_main_label)
    assert w[4] is False
    assert abs(w[0] - rx) < 0.01 and abs(w[1] - ry) < 0.01
    # shift 参与定位基准（toward 是全局坐标，原子位置需加 shift）
    r = mp.mech_arrow_origin(mol, "1", lone_pair_offset=False,
                             shift=(2.0, 0.0), toward=(5.0, 0.0),
                             labeler=mp.atom_main_label)
    assert abs(r[0] - (ax + 2.0)) < 0.01
    # 键中点/电子点返回五元组且不吸附
    b = mp.mech_arrow_origin(mol, "0-1", toward=(3.0, 0.0),
                             labeler=mp.atom_main_label)
    assert b[2] is True and b[3] is False and b[4] is False


def test_error_no_struct():
    """错误处理1：容器内缺少 STRUCT。"""
    out = _render("[COMPOSITE:row][PLUS][/COMPOSITE]")
    assert "缺少 [STRUCT]" in out


def test_error_missing_rxnarrow():
    """错误处理2：reaction_mech 布局缺少 RXNARROW。"""
    out = _render("[COMPOSITE:reaction_mech][STRUCT:CCl][STRUCT:CO][/COMPOSITE]")
    assert "RXNARROW" in out


def test_error_unknown_layout():
    """错误处理3：未知布局名。"""
    out = _render("[COMPOSITE:grid][STRUCT:CCl][/COMPOSITE]")
    assert "未知布局" in out


def test_error_invalid_smiles():
    """错误处理4：无效 SMILES 报出组件 id。"""
    out = _render(
        "[COMPOSITE:row][STRUCT:XYZ_INVALID,id=bad][RXNARROW][STRUCT:CC][/COMPOSITE]"
    )
    assert "无效 SMILES" in out and "bad" in out


def test_unknown_mech_ref_skipped():
    """容错1：机理箭头引用未知 id / 越界原子，跳过不崩溃。"""
    out = _render(
        "[COMPOSITE:reaction_mech]"
        "[STRUCT:CCl][RXNARROW][STRUCT:CO]"
        "[MECHARROW:r9:0>r0:0,r0:99>r2:0]"
        "[/COMPOSITE]"
    )
    assert out.startswith("\\begin{tikzpicture}")
    assert "red" not in out


def test_charge_circle_avoids_bonds():
    """R-8 候选位放置：硝基 N⁺ 的圆圈电荷不压 N=O 双键（Drawbacks 一-8 案例）。

    旧行为：固定 45°/0.34，圈与 60° 方向的双键重合（圈心距键线 ≈0.09 < 半径）；
    新行为：候选位逐个查占据表，落到零冲突角度。
    """
    from renderers.collide import CHARGE_CIRCLE_R, _seg_point_dist
    out = _render("[COMPOSITE:row][STRUCT:O=[N+]([O-])c1ccccc1][/COMPOSITE]")
    m = re.search(
        r"\\begin\{scope\}\[shift=\{\(([-\d.]+),([-\d.]+)\)\}\](.*?)\\end\{scope\}",
        out, re.DOTALL)
    assert m is not None
    body = m.group(3)   # scope 内全部为局部坐标（shift 仅平移，距离不变）
    segs = re.findall(
        r"\\draw \(([-\d.]+),([-\d.]+)\) -- \(([-\d.]+),([-\d.]+)\);", body)
    plus = re.search(
        r"\\node\[draw, circle[^\]]*\] at \(([-\d.]+),([-\d.]+)\) \{\$\+\$\}",
        body)
    assert plus is not None, "未找到 + 电荷圈"
    px, py = float(plus.group(1)), float(plus.group(2))
    dists = [
        _seg_point_dist(float(x1), float(y1), float(x2), float(y2), px, py)
        for x1, y1, x2, y2 in segs
    ]
    assert min(dists) > CHARGE_CIRCLE_R - 0.01, \
        f"+ 圈仍压键：最近键线距离 {min(dists):.3f}"


def test_charge_annotation_child():
    """R-2：CHARGE 子标记在对应组件上标注部分电荷（红色 δ，绕元素符号中心，
    方向避让键与标签氢——Drawbacks 手动测试第 7 条）。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:OCC,label=乙醇,id=et]"
        "[CHARGE:et|0:δ-,1:δ+,2:δ+]"
        "[/COMPOSITE]"
    )
    assert "$\\delta^-$" in out and "$\\delta^+$" in out
    assert "red" in out
    # δ- 标注在 O（标签 "OH"）的元素符号中心：符号中心 = 标签中心左移 0.13
    import renderers.mol_primitives as mp
    mol = mp.prepare_mol("OCC")
    mp.scale_mol_coords(mol, 0.8)   # 与 composite 布局一致（_MOL_SCALE=0.8）
    cx, cy = mp.symbol_center(mol, 0)
    ox, oy = mp.atom_pos(mol, 0)
    assert abs(cx - (ox - 0.13)) < 0.01          # 基准修正存在（绕 O 而非绕 OH）
    # 各 δ 节点应落在 shift + partial_charge_pos（方向避让后的期望坐标）；
    # shift 由 OH 节点反推（row 布局单分子只平移不缩放）
    o_node = _resolve_node_positions(out, "OH")
    assert o_node
    shift_x, shift_y = o_node[0][0] - ox, o_node[0][1] - oy
    nodes = re.findall(
        r"\\node\[font=\\small, red\] at \(([-\d.]+),([-\d.]+)\)", out)
    assert len(nodes) == 3
    for i in range(3):
        ex = shift_x + mp.partial_charge_pos(mol, i)[0]
        ey = shift_y + mp.partial_charge_pos(mol, i)[1]
        assert any(abs(float(nx) - ex) < 0.08 and abs(float(ny) - ey) < 0.08
                   for nx, ny in nodes), f"原子{i} 的电荷节点未落在避让位置"
    # 避让行为锁定：末端 CH₃（原子 2）的 δ+ 在左上 135°（x 偏移为负）——
    # 旧逻辑固定右上 45°，会压住 CH₃ 标签的 H₃ 后缀
    px, py = mp.partial_charge_pos(mol, 2)
    sx, sy = mp.symbol_center(mol, 2)
    assert px < sx and py > sy


def test_hbond_annotation_child():
    """R-2：HBOND 子标记——O—H 实线 + H 标签 + 3~10 个均匀 teal 点。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:OCCO,label=乙二醇,id=diol]"
        "[HBOND:diol|0-3]"
        "[/COMPOSITE]"
    )
    # 供体 O—H 共价键用实线画出并带 H 标签（规范第 4 条）
    assert re.search(r"\\node\[fill=white, inner sep=1pt\] at \([-\d.]+,[-\d.]+\) \{H\}", out)
    dots = re.findall(r"\\fill\[teal\] \(([-\d.]+),([-\d.]+)\) circle", out)
    assert 3 <= len(dots) <= 10                 # 点数 3~10（不过密）
    # 点间距一致（规范：全图点大小、点间距完全一致）
    dists = [math.hypot(float(dots[i+1][0]) - float(dots[i][0]),
                        float(dots[i+1][1]) - float(dots[i][1]))
             for i in range(len(dots) - 1)]
    assert max(dists) - min(dists) < 0.02


def test_hbond_donor_label_no_double_h():
    """氢键给体与受体标签 H 计数 -1：显式 H 不与标签 H 重复（双 H bug 修复）。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:OCCO,label=乙二醇,id=diol]"
        "[HBOND:diol|0-3]"
        "[/COMPOSITE]"
    )
    # 给体 O(0) 与受体 O(3) 的标签均应为 "O"（H 已显式画出），不是 "OH"
    labels = re.findall(r"\\node\[fill=white, inner sep=1pt\] at \([-\d.]+,[-\d.]+\) \{(OH|O)\}", out)
    assert labels.count("OH") == 0               # 无残留 OH 标签
    assert labels.count("O") == 2                # 给体一个 O + 受体一个 O
    # 两个显式 H 节点：给体 H（朝受体）+ 受体 H（远离给体）
    h_nodes = re.findall(r"\\node\[fill=white, inner sep=1pt\] at \(([-\d.]+),([-\d.]+)\) \{H\}", out)
    assert len(h_nodes) == 2


def test_hbond_acceptor_h_away_from_donor():
    """受体羟基同样画成 -O-H：受体 H 朝向远离给体方向（避开氢键点线）。"""
    import renderers.mol_primitives as mp
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:OCCO,label=乙二醇,id=diol]"
        "[HBOND:diol|0-3]"
        "[/COMPOSITE]"
    )
    mol = mp.prepare_mol("OCCO")
    mp.scale_mol_coords(mol, 0.8)
    mp.adjust_hbond_conformation(mol, 0, 3)
    shift = re.search(
        r"\\begin\{scope\}\[shift=\{\(([-\d.]+),([-\d.]+)\)\}\]", out)
    sx, sy = float(shift.group(1)), float(shift.group(2))
    ax, ay = mp.atom_pos(mol, 3)
    dx, dy = mp.atom_pos(mol, 0)[0] - ax, mp.atom_pos(mol, 0)[1] - ay
    # 受体 H 节点 = 离给体 O(0) 最远的 H 节点（给体 H 沿 O(0)→O(3) 方向，
    # 键长化后可能比受体 H 更靠近 O(3)，不能再按"离 O(3) 最近"识别）
    h_nodes = [(float(x), float(y)) for x, y in re.findall(
        r"\\node\[fill=white, inner sep=1pt\] at \(([-\d.]+),([-\d.]+)\) \{H\}", out)]
    assert len(h_nodes) == 2
    ah = max(h_nodes, key=lambda p: math.hypot(
        p[0] - sx - mp.atom_pos(mol, 0)[0], p[1] - sy - mp.atom_pos(mol, 0)[1]))
    v = (ah[0] - sx - ax, ah[1] - sy - ay)
    cos = (v[0] * -dx + v[1] * -dy) / (math.hypot(*v) * math.hypot(dx, dy) or 1.0)
    assert cos > 0.5                      # 受体 H 指向远离给体的一侧
    # O—H 键长 = 普通骨架键长（受体 O(3) 到其邻居 C(2) 的键长）
    bx, by = mp.atom_pos(mol, 2)
    assert abs(math.hypot(*v) - math.hypot(ax - bx, ay - by)) < 0.01


def test_hbond_first_dot_inset_from_h():
    """氢键首点内缩：不落在给体 H 标签中心（起点沿 H→Y 外移 ≥0.15）。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:OCCO,label=乙二醇,id=diol]"
        "[HBOND:diol|0-3]"
        "[/COMPOSITE]"
    )
    h_nodes = [(float(x), float(y)) for x, y in re.findall(
        r"\\node\[fill=white, inner sep=1pt\] at \(([-\d.]+),([-\d.]+)\) \{H\}", out)]
    dots = re.findall(r"\\fill\[teal\] \(([-\d.]+),([-\d.]+)\) circle", out)
    assert h_nodes and len(dots) >= 3
    donor_h = min(h_nodes, key=lambda p: min(
        math.hypot(p[0] - float(d[0]), p[1] - float(d[1])) for d in dots))
    dists = sorted(math.hypot(donor_h[0] - float(d[0]), donor_h[1] - float(d[1]))
                   for d in dots)
    assert dists[0] >= 0.15              # 首点距 H 标签中心至少 0.15


def test_static_composite_no_lone_pairs():
    """静态键线式（XH/BOND/HBOND，无机理箭头）默认不画孤对电子点。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:OCCO,label=乙二醇,id=diol]"
        "[HBOND:diol|0-3]"
        "[XH:diol|0]"
        "[/COMPOSITE]"
    )
    # 孤对电子点是裸 \fill；氢键点是 \fill[teal]，二者需区分
    assert "\\fill (" not in out and "\\fill  (" not in out
    assert out.count("\\fill[teal]") >= 3


def test_xh_explicit_hydrogen():
    """[XH] 显式氢：α-碳按键线式（不标 CHn），画出 1 个 H 节点 + X—H 实线。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:CCC=O,id=pr]"
        "[XH:pr|1]"
        "[/COMPOSITE]"
    )
    # 键线式：α-碳（idx 1）不标 CH/CH2（带 XH 标注的分子碳原子不写标签）
    assert "CH$_{2}$" not in out
    assert "{CH}" not in out
    # 显式 H 节点存在（实线 + H 标签）
    assert re.search(r"\\node\[fill=white, inner sep=1pt\] at \([-\d.]+,[-\d.]+\) \{H\}", out)
    # X—H 实线从 α-碳原子中心起笔（无标签无留白），落在 H 节点上
    h_node = re.search(r"\\node\[fill=white, inner sep=1pt\] at \(([-\d.]+),([-\d.]+)\) \{H\}", out)
    assert h_node is not None
    # 找出终点与 H 节点重合的 \draw（X—H 实线；骨架键不与该 H 节点重合）
    xh_end = f"({float(h_node.group(1)):.2f},{float(h_node.group(2)):.2f})"
    xh_line = re.search(r"\\draw \(([-\d.]+),([-\d.]+)\) -- " + re.escape(xh_end) + ";", out)
    assert xh_line is not None


def test_xh_stacking_multiple():
    """[XH] 多根叠加：α-碳画 2 个 H（扇形不重叠），键线式无碳标签。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:CCC=O,id=pr]"
        "[XH:pr|1]"
        "[XH:pr|1]"
        "[/COMPOSITE]"
    )
    h_nodes = re.findall(r"\\node\[fill=white, inner sep=1pt\] at \(([-\d.]+),([-\d.]+)\) \{H\}", out)
    assert len(h_nodes) == 2
    # 两个 H 位置不同（扇形展开）
    assert h_nodes[0] != h_nodes[1]
    # 键线式：碳原子无 CH/CH2/CH3 标签
    assert "CH$_{2}$" not in out
    assert re.search(r"\\node\[fill=white, inner sep=1pt\] at \([-\d.]+,[-\d.]+\) \{C\}", out) is None


def test_bond_highlight():
    """[BOND] 反应位点键突出：红色粗线从原子中心起笔（键线式无碳标签留白），
    与原键完全重合（全局坐标 = 局部骨架键 + scope shift）。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:CCC=O,id=pr]"
        "[BOND:pr|1-2]"
        "[/COMPOSITE]"
    )
    red = re.search(
        r"\\draw\[very thick, red\] \(([-\d.]+),([-\d.]+)\) -- \(([-\d.]+),([-\d.]+)\);",
        out,
    )
    assert red is not None
    rx1, ry1, rx2, ry2 = map(float, red.groups())
    # 取分子 scope 的 shift，把红色全局坐标换算回局部，再与骨架键线段比对
    import renderers.mol_primitives as mp
    mol = mp.prepare_mol("CCC=O")
    mp.scale_mol_coords(mol, 0.8)
    shift = re.search(
        r"\\begin\{scope\}\[shift=\{\(([-\d.]+),([-\d.]+)\)\}\]", out)
    sx, sy = float(shift.group(1)), float(shift.group(2))
    lx1, ly1, lx2, ly2 = rx1 - sx, ry1 - sy, rx2 - sx, ry2 - sy
    # 局部坐标必须与骨架键线段一致（.2f 精度）
    assert f"\\draw ({lx1:.2f},{ly1:.2f}) -- ({lx2:.2f},{ly2:.2f});" in out
    # 起点 = α-碳原子中心（键线式无标签 → 无留白修剪）
    ax, ay = mp.atom_pos(mol, 1)
    assert abs(lx1 - ax) < 0.01 and abs(ly1 - ay) < 0.01


def test_label_rule_heavy_atom_count():
    """键线式统一规则（heavy_atom_count）：>2 重原子分子无论是否带
    XH 标注都按键线式（碳不标 CHn）；≤2 重原子小分子用结构简式。"""
    annotated = _render(
        "[COMPOSITE:row]"
        "[STRUCT:CCC=O,id=pr]"
        "[XH:pr|1]"
        "[/COMPOSITE]"
    )
    plain = _render("[COMPOSITE:row][STRUCT:CCC=O][/COMPOSITE]")
    small = _render("[COMPOSITE:row][STRUCT:CCl][/COMPOSITE]")
    assert "CH$_{2}$" not in annotated
    assert "CH$_{3}$" not in annotated
    assert "CH$_{2}$" not in plain       # 4 重原子 → 键线式（原结构简式行为已取消）
    assert "CH$_{3}$" not in plain
    assert "CH$_{3}$" in small           # 2 重原子 → 结构简式 CH3Cl


def test_hbond_conformation_folding():
    """HBOND 构象调整：两个羟基折到 C—C 键同一侧。"""
    import renderers.mol_primitives as mp
    mol = mp.prepare_mol("OCCO")
    mp.scale_mol_coords(mol, 0.8)
    mp.adjust_hbond_conformation(mol, 0, 3)

    ax, ay = mp.atom_pos(mol, 1)
    bx, by = mp.atom_pos(mol, 2)
    dx, dy = bx - ax, by - ay

    def side(i):
        px, py = mp.atom_pos(mol, i)
        return dx * (py - ay) - dy * (px - ax)

    assert side(0) * side(3) > 0                # 同侧
    # 键长保持（反射保距）
    assert abs(math.dist(mp.atom_pos(mol, 1), mp.atom_pos(mol, 2)) - 1.2) < 0.1


def test_donor_h_explicit_placement():
    """显式 H：O—H 长度 = 普通骨架键长，方向沿给体→受体（X—H···Y 直线）。"""
    import renderers.mol_primitives as mp
    mol = mp.prepare_mol("OCCO")
    mp.adjust_hbond_conformation(mol, 0, 3)
    hx, hy = mp.place_donor_h(mol, 0, mp.atom_pos(mol, 3))
    x0, y0 = mp.atom_pos(mol, 0)
    x1, y1 = mp.atom_pos(mol, 1)
    bond_len = math.dist((x0, y0), (x1, y1))
    assert abs(math.dist((x0, y0), (hx, hy)) - bond_len) < 0.01
    # H、O0、O3 近似共线（夹角 > 170°）
    v1 = (hx - x0, hy - y0)
    x3, y3 = mp.atom_pos(mol, 3)
    v2 = (x3 - x0, y3 - y0)
    cos = (v1[0]*v2[0] + v1[1]*v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
    assert cos > 0.98


def test_annotation_unknown_ref_ignored():
    """R-2 容错：CHARGE/HBOND 引用未知组件 id，忽略不崩溃。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:CCl]"
        "[CHARGE:nobody|1:δ+]"
        "[HBOND:nobody|0-1]"
        "[/COMPOSITE]"
    )
    assert out.startswith("\\begin{tikzpicture}")
    assert "delta" not in out and "teal" not in out


def test_annotation_bad_bond_ref_ignored():
    """渲染端容错固化：BOND 引用不存在的键（绕过校验层直调渲染器）
    静默跳过不崩溃、无红线；XH 叠加超限（绕过 A2 校验）渲染不崩溃。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:CCC=O,id=pr]"
        "[BOND:pr|0-2]"
        "[/COMPOSITE]"
    )
    assert out.startswith("\\begin{tikzpicture}")
    assert "very thick" not in out             # 不存在的键不画红
    out2 = _render(
        "[COMPOSITE:row]"
        "[STRUCT:CCC=O,id=pr]"
        "[XH:pr|2][XH:pr|2][XH:pr|2]"          # 醛基碳仅 1 H，叠 3 次
        "[/COMPOSITE]"
    )
    assert out2.startswith("\\begin{tikzpicture}")


def test_place_explicit_hs_gap_shortage_warns(capsys):
    """空档不足截断告警（E6）：请求数超过可分配空档时打印告警并少画。"""
    import renderers.mol_primitives as mp
    mol = mp.prepare_mol("CCC=O")
    positions = mp.place_explicit_hs(mol, 1, 10)
    assert len(positions) < 10
    assert "空档不足" in capsys.readouterr().out


ENERGY_DEMO = (
    "[COMPOSITE:energy]"
    "[ENERGY:0,108,-20]"
    "[STRUCT:CCl.[OH-],label=反应物,at=0]"
    "[STRUCT:CCl.[OH-],label=过渡态,at=1]"
    "[STRUCT:CO.[Cl-],label=产物,at=2]"
    "[/COMPOSITE]"
)


def test_energy_layout_annotations_rendered():
    """B3：energy 布局驻点结构支持 XH/BOND/CHARGE 注解（不再静默丢弃）。"""
    out = _render(
        "[COMPOSITE:energy]"
        "[ENERGY:0,108,-20]"
        "[STRUCT:CCl.[OH-],label=反应物,at=0,id=r0][CHARGE:r0|0:δ+]"
        "[STRUCT:CCl.[OH-],label=过渡态,at=1,id=ts]"
        "[STRUCT:CO.[Cl-],label=产物,at=2,id=p0][XH:p0|1][BOND:p0|0-1]"
        "[/COMPOSITE]"
    )
    assert "\\begin{tikzpicture}" in out
    assert "delta" in out                     # CHARGE 部分电荷标注
    assert "very thick, red" in out           # BOND 键突出
    assert re.search(r"\{H\}", out)           # XH 显式 H 节点


def test_energy_layout_full():
    """R-3 正例：势能面曲线 + 3 个驻点结构 + 驻点标签用 STRUCT label。"""
    out = _render(ENERGY_DEMO)
    assert out.startswith("\\begin{tikzpicture}")
    assert "smooth] plot coordinates" in out              # 势能面曲线
    assert "\\draw[gray, dashed]" in out                 # 反应物基线
    assert "Ea $\\approx$ 108" in out                    # Ea 标注框
    assert "anchor=north" in out                         # 标注框 anchor 定位
    # 3 个驻点 scope + 3 个分子 scope（标注框为带 anchor 的直接节点）
    assert out.count("\\begin{scope}[shift=") == 6
    assert "反应物 (+0)" in out and "过渡态 (+108)" in out and "产物 (-20)" in out


def test_energy_layout_struct_positions():
    """R-3 布局：分子在驻点正上方（above）或下方（below），水平居中。"""
    out = _render(ENERGY_DEMO)
    scopes = re.findall(
        r"\\begin\{scope\}\[shift=\{\(([-\d.]+),([-\d.]+)\)\}\]", out)
    assert len(scopes) == 6
    # 前 3 个是驻点 scope（y 单调：低-高-低），后 3 个是分子 scope
    point_ys = [float(y) for _, y in scopes[:3]]
    mol_ys = [float(y) for _, y in scopes[3:6]]
    assert point_ys[1] > point_ys[0] and point_ys[1] > point_ys[2]   # 过渡态最高
    for my, py in zip(mol_ys, point_ys):
        assert my > py - 0.5                                        # 分子在驻点上方


def test_energy_layout_box_above_everything():
    """标注框在最高组件上方净空带（底边高于所有分子与标签），纵轴随之加高。"""
    out = _render(ENERGY_DEMO)
    box = re.search(r"anchor=north (?:east|west)\] at \(([-\d.]+),([-\d.]+)\)", out)
    assert box is not None
    box_y = float(box.group(2))
    scopes = re.findall(
        r"\\begin\{scope\}\[shift=\{\(([-\d.]+),([-\d.]+)\)\}\]", out)
    mol_shift_ys = [float(y) for _, y in scopes[3:6]]
    # 标注框 y（顶边）比最高分子 shift 高出净空（gap 0.3 + 分子自身高度）
    assert box_y > max(mol_shift_ys) + 0.8
    # 纵轴顶端 = 标注框顶 + 0.2
    axis = re.search(r"\\draw\[->\] \(0,0\) -- \(0,([-\d.]+)\);", out)
    assert abs(float(axis.group(1)) - (box_y + 0.2)) < 0.05


def test_energy_layout_pos_below():
    """R-3 pos=below：分子放在驻点下方。"""
    out = _render(
        "[COMPOSITE:energy]"
        "[ENERGY:0,108,-20]"
        "[STRUCT:CCl,at=1,pos=below]"
        "[/COMPOSITE]"
    )
    scopes = re.findall(
        r"\\begin\{scope\}\[shift=\{\(([-\d.]+),([-\d.]+)\)\}\]", out)
    point_y = float(scopes[1][1])        # 过渡态驻点（最高点）
    mol_y = float(scopes[3][1])          # 分子 scope（顶部应低于驻点 margin 0.6）
    assert mol_y < point_y - 0.5


def test_energy_layout_errors():
    """R-3 错误路径：缺 ENERGY / STRUCT 缺 at / at 越界。"""
    assert "需要 [ENERGY" in _render(
        "[COMPOSITE:energy][STRUCT:CCl,at=0][/COMPOSITE]")
    assert "需要 at=" in _render(
        "[COMPOSITE:energy][ENERGY:0,108,-20][STRUCT:CCl][/COMPOSITE]")
    assert "超出能量点范围" in _render(
        "[COMPOSITE:energy][ENERGY:0,108,-20][STRUCT:CCl,at=5][/COMPOSITE]")


def test_resonance_arrow_explicit():
    """共振式（重构后）：row 布局 + 显式 [RESARROW] 手动插 ↔。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:C1=CC=CC=C1,label=式 I]"
        "[RESARROW]"
        "[STRUCT:C1C=CC=CC=1,label=式 II]"
        "[/COMPOSITE]"
    )
    assert out.count("$\\leftrightarrow$") == 1
    assert out.count("\\begin{scope}[shift=") == 2
    assert "式 I" in out and "式 II" in out


def test_resonance_three_forms_two_arrows():
    """三个共振极限式两个 ↔（显式 [RESARROW]）。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:C1=CC=CC=C1][RESARROW]"
        "[STRUCT:C1C=CC=CC=1][RESARROW]"
        "[STRUCT:c1ccccc1]"
        "[/COMPOSITE]"
    )
    assert out.count("$\\leftrightarrow$") == 2
    assert out.count("\\begin{scope}[shift=") == 3


def test_resonance_layout_deprecated_rejected():
    """旧 resonance 布局已废弃：校验拦截（未知布局），须用 row + 显式 [RESARROW]。"""
    import core.tag_validator as tv
    from core.tag_parser import parse_tags
    tags = parse_tags(
        "[COMPOSITE:resonance]"
        "[STRUCT:C1=CC=CC=C1]"
        "[STRUCT:C1C=CC=CC=1]"
        "[/COMPOSITE]"
    )
    ok, bad = tv.validate_tags(tags)
    assert len(bad) == 1
    assert "未知布局" in bad[0].reason


def test_newline_vertical_stacking():
    """R-6：NEWLINE 换行，主结构在上、共振式在下（上下排列）。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:CC(=O)[O-],label=羧酸根]"
        "[NEWLINE]"
        "[STRUCT:CC(=O)[O-]][RESARROW][STRUCT:CC([O-])=O]"
        "[/COMPOSITE]"
    )
    assert out.count("$\\leftrightarrow$") == 1
    scopes = re.findall(
        r"\\begin\{scope\}\[shift=\{\(([-\d.]+),([-\d.]+)\)\}\]", out)
    assert len(scopes) == 3
    ys = [float(y) for _, y in scopes]
    assert ys[0] > ys[1] and ys[0] > ys[2]   # 第一行在第二行上方
    assert abs(ys[1] - ys[2]) < abs(ys[0] - ys[1])  # 第二行两个分子同高（或接近）


def test_registry_dispatch_and_injection():
    """集成：注册表分派 + 注入器整串替换。"""
    text = f"SN2 反应机理如下：\n{SN2_DEMO}\n以上。"
    tags = parse_tags(text)
    rendered = {}
    for tag in tags:
        renderer = RENDERER_REGISTRY.get(tag.type)
        assert renderer is not None
        rendered[tag.raw] = renderer(*tag.args)
    out = inject_tags_into_text(text, tags, rendered)
    assert "[COMPOSITE" not in out
    assert "[STRUCT" not in out
    assert "[/COMPOSITE]" not in out
    assert "\\begin{tikzpicture}" in out
    assert out.startswith("SN2 反应机理如下：")
    assert out.endswith("以上。")


# ---------- 跨组件（分子间）氢键 ----------

_INTER_HB_BASE = (
    "[COMPOSITE:row]"
    "[STRUCT:CC(=O)O,id=a,label=乙酸]"
    "[STRUCT:CC(=O)O,id=b,label=乙酸]"
)


def test_hbond_inter_parser_both_forms():
    """跨组件 HBOND 两种写法（`id|from>idB:to` 与紧凑 `id:from>idB:to`）
    解析结果一致：args = [给体组件 id, "from>idB:to"]。"""
    for spec in ("[HBOND:a|3>b:2]", "[HBOND:a:3>b:2]"):
        tags = parse_tags(_INTER_HB_BASE + spec + "[/COMPOSITE]")
        comp = tags[0]
        hbond = [c for c in comp.args[1] if c.type == "HBOND"][0]
        assert hbond.args == ["a", "3>b:2"]


def test_hbond_inter_validation():
    """跨组件 HBOND 校验：合法通过；未知组件/越界/格式错拦截。"""
    from core.tag_validator import validate_tags

    def check(spec):
        tags = parse_tags(_INTER_HB_BASE + spec + "[/COMPOSITE]")
        _, invalid = validate_tags(tags)
        return invalid

    assert not check("[HBOND:a|3>b:2]")
    r = check("[HBOND:a|3>ghost:2]")
    assert r and "引用未知组件" in r[0].reason
    r = check("[HBOND:a|99>b:2]")
    assert r and "超出组件 a" in r[0].reason
    r = check("[HBOND:a|3>b:99]")
    assert r and "超出组件 b" in r[0].reason
    r = check("[HBOND:a|bad>]")
    assert r and "标注格式错误" in r[0].reason


def test_hbond_inter_renders_dots():
    """跨组件氢键渲染：给体 X—H 实线 + H 节点 + teal 点状虚线 + 两分子。"""
    text = _INTER_HB_BASE + "[HBOND:a|3>b:2][/COMPOSITE]"
    tags = parse_tags(text)
    out = render_composite(*tags[0].args)
    assert out.startswith("\\begin{tikzpicture}")
    assert "\\fill[teal]" in out          # H···Y 点状虚线（圆点）
    assert out.count("\\fill[teal]") >= 3
    assert out.count("{H}") == 1          # 给体显式 H（受体不画 H）
    assert out.count("{乙酸}") == 2       # 两分子 label 都在


def test_hbond_intra_unchanged():
    """单组件（分子内）HBOND 行为不变：`id|from-to` 仍只画该组件内氢键。"""
    text = ("[COMPOSITE:row][STRUCT:OCCO,id=g,label=乙二醇]"
            "[HBOND:g|0-3][/COMPOSITE]")
    tags = parse_tags(text)
    out = render_composite(*tags[0].args)
    assert out.startswith("\\begin{tikzpicture}")
    assert "\\fill[teal]" in out
    assert out.count("{H}") == 2   # 给体 H + 受体显式 H（分子内受体有 H）
