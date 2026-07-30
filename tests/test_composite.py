# -*- coding: utf-8 -*-
"""tests/test_composite.py — [COMPOSITE] 容器式复合标记渲染器单元测试。

运行: python -m pytest tests/test_composite.py -v
"""

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
    """进攻箭头起点落在孤对电子点上（点距 0.30，绕元素符号中心）。"""
    out = _render(SN2_DEMO)
    o_pos = _resolve_node_positions(out, "OH")
    assert o_pos
    ox, oy = o_pos[0]
    # "OH" 后缀宽 1 字符 -> 符号中心左移 0.13；正上方槽位，点距 0.30
    expected = (ox - 0.13, oy + 0.30)
    m = re.search(r"\\draw\[->, thick, red\] \(([-\d.]+),([-\d.]+)\)", out)
    assert m is not None
    start = (float(m.group(1)), float(m.group(2)))
    assert abs(start[0] - expected[0]) < 0.1
    assert abs(start[1] - expected[1]) < 0.1


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
        # 点距 0.30（容差含点对半距 0.055）
        assert abs(abs(dx) + abs(dy) - 0.30) < 0.08


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
    """箭头终点贴近目标原子（内缩 0.10），起点在键中点的箭头贴近键。"""
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
    assert abs(attack_end[0] - c_pos[0]) < 0.2 and abs(attack_end[1] - c_pos[1]) < 0.2
    assert abs(bond_end[0] - cl_pos[0]) < 0.2 and abs(bond_end[1] - cl_pos[1]) < 0.2


def test_format_chem_text():
    """化学文本排版：数字下标、尾部电荷上标、已排版文本跳过。"""
    assert format_chem_text("CH3Cl") == "CH$_3$Cl"
    assert format_chem_text("OH-") == "OH$^{-}$"
    assert format_chem_text("H2SO4, 浓HNO3") == "H$_2$SO$_4$, 浓HNO$_3$"
    assert format_chem_text("SO42-") == "SO$_4$$^{2-}$"
    assert format_chem_text("NH4+") == "NH$_4$$^{+}$"
    assert format_chem_text("OH$^-$") == "OH$^-$"
    assert format_chem_text("") == ""


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
    assert "H$_2$O" in out and "CuO, Δ" in out and "O$_2$" in out
    scopes = re.findall(r"\\begin\{scope\}\[shift=\{\(([-\d.]+),", out)
    xs = [float(x) for x in scopes]
    assert xs == sorted(xs)                     # 分子按序列从左到右递增


def test_fishhook_arrows():
    """正例3：鱼钩箭头（单电子）生成半边 barb。"""
    out = _render(
        "[COMPOSITE:reaction_mech]"
        "[STRUCT:C=C][PLUS][STRUCT:[Br],id=br]"
        "[RXNARROW:hv]"
        "[STRUCT:[CH2]CBr]"
        "[MECHARROW:br:0>>r0:0]"
        "[/COMPOSITE]"
    )
    assert "\\draw[thick, red]" in out              # 鱼钩曲线（无 -> 全箭头）
    assert "\\draw[->, thick, red]" not in out
    assert "hv" in out                              # RXNARROW 内联条件生效


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


def test_charge_annotation_child():
    """R-2：CHARGE 子标记在对应组件上标注部分电荷（红色 δ，绕元素符号中心）。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:OCC,label=乙醇,id=et]"
        "[CHARGE:et|0:δ-,1:δ+]"
        "[/COMPOSITE]"
    )
    assert "$\\delta^-$" in out and "$\\delta^+$" in out
    assert "red" in out
    # δ- 标注在 O（标签 "OH"）的元素符号中心右上：符号中心 = 标签中心左移 0.13
    import renderers.mol_primitives as mp
    mol = mp.prepare_mol("OCC")
    cx, cy = mp.symbol_center(mol, 0)
    ox, oy = mp.atom_pos(mol, 0)
    assert abs(cx - (ox - 0.13)) < 0.01          # 基准修正存在（绕 O 而非绕 OH）
    # 输出中 δ- 节点的 x 应接近"符号中心+shift+0.30"而非"标签中心+shift+0.30"
    dnode = re.search(r"\\node\[font=\\small, red\] at \(([-\d.]+),([-\d.]+)\) \{\$\\delta\^-\$\}", out)
    assert dnode is not None
    dx = float(dnode.group(1))
    o_node = _resolve_node_positions(out, "OH")
    assert o_node
    assert abs(dx - (o_node[0][0] - 0.13 + 0.30)) < 0.15


def test_hbond_annotation_child():
    """R-2：HBOND 子标记在对应组件内画氢键虚线（teal dashed）。"""
    out = _render(
        "[COMPOSITE:row]"
        "[STRUCT:OCO,id=diol]"
        "[HBOND:diol|0-2]"
        "[/COMPOSITE]"
    )
    assert "\\draw[dashed, teal, thick]" in out


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


ENERGY_DEMO = (
    "[COMPOSITE:energy]"
    "[ENERGY:0,108,-20]"
    "[STRUCT:CCl.[OH-],label=反应物,at=0]"
    "[STRUCT:CCl.[OH-],label=过渡态,at=1]"
    "[STRUCT:CO.[Cl-],label=产物,at=2]"
    "[/COMPOSITE]"
)


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


def test_resonance_layout_auto_arrow():
    """R-6：resonance 布局连续 STRUCT 之间自动插入 ↔（无需手写连接符）。"""
    out = _render(
        "[COMPOSITE:resonance]"
        "[STRUCT:C1=CC=CC=C1,label=式 I]"
        "[STRUCT:C1C=CC=CC=1,label=式 II]"
        "[/COMPOSITE]"
    )
    assert out.count("$\\leftrightarrow$") == 1
    assert out.count("\\begin{scope}[shift=") == 2
    assert "式 I" in out and "式 II" in out


def test_resonance_layout_three_forms():
    """R-6：三个共振极限式两个 ↔。"""
    out = _render(
        "[COMPOSITE:resonance]"
        "[STRUCT:C1=CC=CC=C1][STRUCT:C1C=CC=CC=1][STRUCT:c1ccccc1]"
        "[/COMPOSITE]"
    )
    assert out.count("$\\leftrightarrow$") == 2
    assert out.count("\\begin{scope}[shift=") == 3


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
