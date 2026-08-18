# -*- coding: utf-8 -*-
"""tests/test_parsing.py — 标记解析器单元测试。

运行: python -m pytest tests/test_parsing.py -v
"""

from core.tag_parser import parse_tags, RenderTag


def test_single_struct():
    """用例1：单标记 STRUCT。"""
    tags = parse_tags("[STRUCT:c1ccccc1]")
    assert len(tags) == 1
    t = tags[0]
    assert t.type == "STRUCT"
    assert t.args[0] == "c1ccccc1"
    assert t.args[1] is None  # 无 label
    assert t.raw == "[STRUCT:c1ccccc1]"
    assert t.start_pos == 0


def test_multiple_struct_mixed_text():
    """用例2：多标记与文字混合。"""
    text = "苯 [STRUCT:c1ccccc1] 硝化生成 [STRUCT:c1ccccc1N]"
    tags = parse_tags(text)
    assert len(tags) == 2
    assert tags[0].type == "STRUCT"
    assert tags[0].args[0] == "c1ccccc1"
    assert tags[1].type == "STRUCT"
    assert tags[1].args[0] == "c1ccccc1N"
    # 按 start_pos 升序
    assert tags[0].start_pos < tags[1].start_pos


def test_struct_with_label():
    """用例3：带 label 的 STRUCT。"""
    tags = parse_tags("[STRUCT:CC(=O)O,label=乙酸]")
    assert len(tags) == 1
    t = tags[0]
    assert t.type == "STRUCT"
    assert t.args[0] == "CC(=O)O"
    assert t.args[1] == "乙酸"  # label 正确提取（验证正则 bug 修正）


def test_struct_trailing_comma_normalized():
    """STRUCT 的 SMILES 尾逗号（LLM 笔误）归一化去掉。"""
    tags = parse_tags("[STRUCT:CC[OH2+],]")
    assert len(tags) == 1
    assert tags[0].args[0] == "CC[OH2+]"


def test_stereo_with_label():
    """STEREO 归一化为 STRUCT+mode（分子家族重构），label 保留。"""
    tags = parse_tags("[STEREO:C[C@@H](O)C(=O)O,label=(R)-乳酸]")
    assert len(tags) == 1
    t = tags[0]
    assert t.type == "STRUCT"
    assert t.attrs["mode"] == "stereo"
    assert t.attrs["orig_type"] == "STEREO"
    assert t.args[0] == "C[C@@H](O)C(=O)O"
    assert t.args[1] == "(R)-乳酸"


def test_lewis_with_label():
    """LEWIS 归一化为 STRUCT+mode，label 保留。"""
    tags = parse_tags("[LEWIS:O,label=水]")
    assert len(tags) == 1
    t = tags[0]
    assert t.type == "STRUCT"
    assert t.attrs["mode"] == "lewis"
    assert t.args[0] == "O"
    assert t.args[1] == "水"


def test_struct_mode_named_params():
    """[STRUCT:SMILES, mode=...] 命名参数解析（分子家族统一写法，容忍逗号后空格）。"""
    t = parse_tags("[STRUCT:O, mode=lewis, label=水]")[0]
    assert t.type == "STRUCT"
    assert t.args == ["O", "水"]
    assert t.attrs["mode"] == "lewis"
    t2 = parse_tags("[STRUCT:CC, mode=newman, bond=0-1, angle=60]")[0]
    assert t2.args[0] == "CC"
    assert t2.attrs["mode"] == "newman"
    assert t2.attrs["bond"] == "0-1"
    assert t2.attrs["angle"] == "60"
    t3 = parse_tags("[STRUCT:BrC1CCCCC1, mode=chair, subs=1:ax]")[0]
    assert t3.args[0] == "BrC1CCCCC1"
    assert t3.attrs["mode"] == "chair"
    assert t3.attrs["subs"] == "1:ax"
    # 缺省 mode = skeleton；与 label/id/at/pos 混用
    t4 = parse_tags("[STRUCT:CCl,label=底物,id=s0]")[0]
    assert t4.attrs["mode"] == "skeleton"
    assert t4.attrs["id"] == "s0"
    # 20260821：bond=/charge= 参数化标注（bond 与 newman 语义分派）
    t5 = parse_tags("[STRUCT:CCC=O, bond=1-2, charge=0:+,3:-]")[0]
    assert t5.args[0] == "CCC=O"
    assert t5.attrs["mode"] == "skeleton"
    assert t5.attrs["bond"] == "1-2"
    assert t5.attrs["charge"] == "0:+,3:-"
    # charge 值内逗号不污染 SMILES 截止（与 label 混用）
    t6 = parse_tags("[STRUCT:CCC=O, charge=0:+, label=丙醛]")[0]
    assert t6.args[0] == "CCC=O"
    assert t6.attrs["charge"] == "0:+"
    assert t6.args[1] == "丙醛"


def test_composite_arrow_new_syntax():
    """大一统架构：容器内 [ARROW:type=..., sup=..., 条件] 解析。"""
    tag = parse_tags("[COMPOSITE:reaction]"
                     "[STRUCT:CCl,id=A]"
                     "[ARROW:type=reversible,sup=+E,-F,条件]"
                     "[/COMPOSITE]")[0]
    arrow = next(c for c in tag.args[1] if c.type == "ARROW")
    assert arrow.args[0] == "reversible"
    assert arrow.args[1] == ["+E", "-F"]
    assert arrow.args[2] == "条件"
    # 无附件、无条件的箭头
    tag2 = parse_tags("[COMPOSITE:reaction][STRUCT:CC,id=A]"
                      "[ARROW:type=single][/COMPOSITE]")[0]
    a2 = next(c for c in tag2.args[1] if c.type == "ARROW")
    assert a2.args == ["single", [], ""]


def test_composite_arrow_toplevel_old_syntax_kept():
    """顶层 [ARROW:反应物,产物,类型] 旧语法不受容器内新语法影响。"""
    tags = parse_tags("[ARROW:CCO,CC=O,Cu, Δ]")
    assert tags[0].type == "ARROW"
    assert tags[0].args == ["CCO", "CC=O", "Cu, Δ"]


def test_composite_block_parsing():
    """[BLOCK]...[/BLOCK] 共振块配对解析（块内子标记递归）。"""
    tag = parse_tags("[COMPOSITE:reaction]"
                     "[STRUCT:c1ccccc1,id=A]"
                     "[BLOCK][STRUCT:C1=CC=CC=C1,id=B1]"
                     "[ARROW:type=resonance]"
                     "[STRUCT:C1C=CC=CC=1,id=B2][/BLOCK]"
                     "[/COMPOSITE]")[0]
    block = next(c for c in tag.args[1] if c.type == "BLOCK")
    inner = block.args[0]
    assert [c.type for c in inner] == ["STRUCT", "ARROW", "STRUCT"]
    assert inner[1].args[0] == "resonance"


def test_composite_struct_arrow_token():
    """STRUCT 的 arrow 裸令牌：附件标记进 attrs，id 不被污染。"""
    tag = parse_tags("[COMPOSITE:reaction]"
                     "[STRUCT:[OH-],id=E,arrow]"
                     "[/COMPOSITE]")[0]
    s = next(c for c in tag.args[1] if c.type == "STRUCT")
    assert s.attrs["arrow"] is True
    assert s.attrs["id"] == "E"
    assert s.args[0] == "[OH-]"


def test_composite_prose_mention_not_swallowed():
    """正文文字提及 [COMPOSITE:...]（如"用 [COMPOSITE:row] 展示"）时不应
    与后面的真容器贪婪配对——真容器必须正常解析，不被幻影容器吞掉。"""
    text = ("下面用 [COMPOSITE:reaction_mech,numbering] 展示序号：\n"
            "[COMPOSITE:reaction_mech]"
            "[STRUCT:CCl][RXNARROW][STRUCT:CO]"
            "[/COMPOSITE]")
    tags = parse_tags(text)
    assert len(tags) == 1
    t = tags[0]
    assert t.type == "COMPOSITE"
    assert t.args[0] == "reaction_mech"
    assert "使用" not in t.raw and "下面" not in t.raw
    assert len(t.args[1]) == 3  # 2 个 STRUCT + 1 个 RXNARROW


def test_xh_toplevel_parse():
    """顶层 [XH:SMILES|序号]（单分子位点标注，无需 COMPOSITE）。"""
    tags = parse_tags("[XH:CC(=O)O|3]")
    assert len(tags) == 1
    assert tags[0].type == "XH"
    assert tags[0].args == ["CC(=O)O", "3"]


def test_bond_toplevel_parse():
    """顶层 [BOND:SMILES|a-b]。"""
    tags = parse_tags("[BOND:CCC=O|1-2]")
    assert len(tags) == 1
    assert tags[0].type == "BOND"
    assert tags[0].args == ["CCC=O", "1-2"]


def test_xh_bond_label_miswrite_stripped():
    """Drawbacks 九 C-2：非 STRUCT 标记（XH/BOND/CHARGE/HBOND）误写
    ,label=（LLM 以为所有标记都支持 label）时，SMILES/ref 字段必须剥离
    label 尾随，否则整串 'O,label=苯酚' 进 RDKit 报 SMILES Parse Error。"""
    for raw, expect in [
        ("[XH:O,label=苯酚|0]", ["O", "0"]),
        ("[BOND:O,label=苯酚|0-1]", ["O", "0-1"]),
        ("[HBOND:O,label=苯酚|0-1]", ["O", "0-1"]),
        ("[CHARGE:O,label=苯酚|0:δ-]", ["O", "0:δ-"]),
        ("[XH:CC(=O)O|3]", ["CC(=O)O", "3"]),   # 无 label 正常不受影响
        ("[BOND:CCC=O|1-2]", ["CCC=O", "1-2"]),
    ]:
        tags = parse_tags(raw)
        assert len(tags) == 1, raw
        assert tags[0].args == expect, f"{raw} → {tags[0].args}"


def test_reasoning_paired():
    """用例4：REASONING 配对标记。"""
    tags = parse_tags("[REASONING]亲电取代[/REASONING]")
    assert len(tags) == 1
    t = tags[0]
    assert t.type == "REASONING"
    assert t.args[0] == "亲电取代"
    assert t.raw == "[REASONING]亲电取代[/REASONING]"


def test_no_tags_plain_text():
    """用例5：无标记的纯文本。"""
    tags = parse_tags("你好")
    assert tags == []


def test_arrow():
    """补充：ARROW 三参数。"""
    tags = parse_tags("[ARROW:c1ccccc1,c1ccccc1N,amination]")
    assert len(tags) == 1
    t = tags[0]
    assert t.type == "ARROW"
    assert t.args == ["c1ccccc1", "c1ccccc1N", "amination"]


def test_newman():
    """NEWMAN 归一化为 STRUCT+mode（旧格式 [SMILES,角度] → bond 空、angle 生效）。"""
    tags = parse_tags("[NEWMAN:CC,60]")
    assert len(tags) == 1
    t = tags[0]
    assert t.type == "STRUCT"
    assert t.attrs["mode"] == "newman"
    assert t.attrs["bond"] == ""
    assert t.attrs["angle"] == "60"


def test_newman_with_bond():
    """NEWMAN 三参数：[SMILES, a-b, 角度]（投影观察键 + 二面角）。"""
    tags = parse_tags("[NEWMAN:CC,0-1,60]")
    assert len(tags) == 1
    t = tags[0]
    assert t.type == "STRUCT"
    assert t.attrs["mode"] == "newman"
    assert t.attrs["bond"] == "0-1"
    assert t.attrs["angle"] == "60"


def test_energy():
    """补充：ENERGY 点序列。"""
    tags = parse_tags("[ENERGY:0,15,25,5,10]")
    assert len(tags) == 1
    assert tags[0].type == "ENERGY"
    assert tags[0].args[0] == "0,15,25,5,10"


def test_mixed_types_sorted():
    """补充：不同类型混合，按位置排序。"""
    text = "[STRUCT:c1ccccc1] 和 [ARROW:c1ccccc1,c1ccccc1N,amination] 然后 [ENERGY:0,80,-20]"
    tags = parse_tags(text)
    assert len(tags) == 3
    assert [t.type for t in tags] == ["STRUCT", "ARROW", "ENERGY"]
    # 严格升序
    positions = [t.start_pos for t in tags]
    assert positions == sorted(positions)


def test_end_pos_correctness():
    """补充：end_pos 指向 raw 之后。"""
    text = "前[STRUCT:C]后"
    tags = parse_tags(text)
    assert len(tags) == 1
    t = tags[0]
    assert t.start_pos == 1
    assert t.end_pos == 1 + len(t.raw)
    assert text[t.start_pos:t.end_pos] == t.raw


def test_struct_bracket_atoms():
    """补充：SMILES 含括号原子（[N+]/[O-]），验证括号平衡扫描器。"""
    tags = parse_tags("[STRUCT:O=[N+]([O-])c1ccccc1]")
    assert len(tags) == 1
    assert tags[0].args[0] == "O=[N+]([O-])c1ccccc1"


def test_struct_chiral_brackets():
    """补充：手性 SMILES [C@@H] 等含括号。"""
    tags = parse_tags("[STRUCT:[C@@H](N)(C)O]")
    assert len(tags) == 1
    assert tags[0].args[0] == "[C@@H](N)(C)O"


def test_arrow_bracket_smiles():
    """补充：ARROW 反应物/产物含括号原子。"""
    tags = parse_tags("[ARROW:O=[N+]([O-])c1ccccc1,c1ccccc1N,a]")
    assert tags[0].args[0] == "O=[N+]([O-])c1ccccc1"
    assert tags[0].args[1] == "c1ccccc1N"


def test_reaction_basic():
    """[REACTION] 基础解析：反应物 | 产物 | 条件。"""
    tags = parse_tags("[REACTION:c1ccccc1;[O-][N+](=O)[O-]|O=[N+]([O-])c1ccccc1;O|H2SO4]")
    assert len(tags) == 1
    t = tags[0]
    assert t.type == "REACTION"
    assert t.args[0] == "c1ccccc1;[O-][N+](=O)[O-]"
    assert t.args[1] == "O=[N+]([O-])c1ccccc1;O"
    assert t.args[2] == "H2SO4"


def test_reaction_no_conditions():
    """[REACTION] 省略条件时，args[2] 为空串。"""
    tags = parse_tags("[REACTION:C;O2|CO2]")
    assert len(tags) == 1
    t = tags[0]
    assert t.args[0] == "C;O2"
    assert t.args[1] == "CO2"
    assert t.args[2] == ""


def test_reaction_bracket_atoms():
    """[REACTION] 反应物/产物含 [N+]/[O-] 括号原子，验证括号平衡扫描器。"""
    tags = parse_tags("[REACTION:[O-][N+](=O)[O-]|O=[N+]([O-])c1ccccc1|]")
    assert len(tags) == 1
    t = tags[0]
    assert t.args[0] == "[O-][N+](=O)[O-]"
    assert t.args[1] == "O=[N+]([O-])c1ccccc1"
    assert t.args[2] == ""


def test_composite_basic():
    """[COMPOSITE] 基础解析：布局名 + 子标记列表。"""
    text = (
        "[COMPOSITE:reaction_mech]"
        "[STRUCT:CCl,label=CH3Cl]"
        "[PLUS]"
        "[STRUCT:[OH-],label=OH-]"
        "[RXNARROW]"
        "[STRUCT:[Cl-]][PLUS][STRUCT:CO]"
        "[MECHARROW:r1:0>r0:0]"
        "[CONDITION:SN2]"
        "[/COMPOSITE]"
    )
    tags = parse_tags(text)
    assert len(tags) == 1
    t = tags[0]
    assert t.type == "COMPOSITE"
    assert t.args[0] == "reaction_mech"
    children = t.args[1]
    assert [c.type for c in children] == [
        "STRUCT", "PLUS", "STRUCT", "RXNARROW",
        "STRUCT", "PLUS", "STRUCT", "MECHARROW", "CONDITION",
    ]
    assert children[0].args == ["CCl", "CH3Cl"]
    assert children[2].args == ["[OH-]", "OH-"]
    assert children[7].args == ["r1:0>r0:0"]
    assert children[8].args == ["SN2"]
    assert text[t.start_pos:t.end_pos] == t.raw


def test_composite_inner_tags_not_toplevel():
    """[COMPOSITE] 容器内的 STRUCT 不会作为顶层标记重复出现。"""
    text = "前 [STRUCT:C] [COMPOSITE:row][STRUCT:CC][/COMPOSITE] 后"
    tags = parse_tags(text)
    assert [t.type for t in tags] == ["STRUCT", "COMPOSITE"]
    assert tags[0].args[0] == "C"
    children = tags[1].args[1]
    assert len(children) == 1
    assert children[0].args[0] == "CC"


def test_composite_rxnarrow_inline_condition():
    """[COMPOSITE] 内 [RXNARROW:条件] 内联条件解析。"""
    tags = parse_tags(
        "[COMPOSITE:reaction_mech][STRUCT:C=C][RXNARROW:H2SO4][STRUCT:CCO][/COMPOSITE]"
    )
    children = tags[0].args[1]
    assert [c.type for c in children] == ["STRUCT", "RXNARROW", "STRUCT"]
    assert children[1].args == ["H2SO4"]


def test_composite_struct_with_id():
    """[COMPOSITE] 内 STRUCT 的 id= 不污染 SMILES 与 label。"""
    tags = parse_tags("[COMPOSITE:row][STRUCT:CCl,label=CH3Cl,id=sub][/COMPOSITE]")
    child = tags[0].args[1][0]
    assert child.args == ["CCl", "CH3Cl"]
    assert ",id=sub" in child.raw


def test_composite_struct_id_before_label():
    """[COMPOSITE] 内 STRUCT 的 id= 写在 label= 之前同样解析正确。"""
    tags = parse_tags("[COMPOSITE:row][STRUCT:CCl,id=sub,label=CH3Cl][/COMPOSITE]")
    child = tags[0].args[1][0]
    assert child.args == ["CCl", "CH3Cl"]


def test_composite_unclosed_degrades():
    """[COMPOSITE] 缺少 [/COMPOSITE] 时不产生 COMPOSITE 标记，内部标记按顶层处理。"""
    text = "[COMPOSITE:row][STRUCT:CCl]"
    tags = parse_tags(text)
    assert [t.type for t in tags] == ["STRUCT"]
    assert tags[0].args[0] == "CCl"


def test_composite_reasoning_inside_filtered():
    """[COMPOSITE] 内的 REASONING 不作为顶层标记返回。"""
    text = "[COMPOSITE:row][REASONING]x[/REASONING][STRUCT:C][/COMPOSITE]"
    tags = parse_tags(text)
    assert [t.type for t in tags] == ["COMPOSITE"]


def test_composite_charge_hbond_children():
    """[COMPOSITE] 内 CHARGE/HBOND 子标记解析（R-2），不作为顶层标记。"""
    text = (
        "[COMPOSITE:row]"
        "[STRUCT:OCC,id=et]"
        "[CHARGE:et|0:δ-,1:δ+]"
        "[HBOND:et|0-2]"
        "[/COMPOSITE]"
    )
    tags = parse_tags(text)
    assert [t.type for t in tags] == ["COMPOSITE"]
    children = tags[0].args[1]
    assert [c.type for c in children] == ["STRUCT", "CHARGE", "HBOND"]
    assert children[1].args == ["et", "0:δ-,1:δ+"]
    assert children[2].args == ["et", "0-2"]


def test_composite_struct_at_pos_attrs():
    """[COMPOSITE] energy 布局 STRUCT 的 at=/pos= 不污染 SMILES 与 label（R-3）。"""
    text = "[COMPOSITE:energy][STRUCT:CCl,label=底物,at=1,pos=below][/COMPOSITE]"
    child = parse_tags(text)[0].args[1][0]
    assert child.args == ["CCl", "底物"]
    assert ",at=1" in child.raw and ",pos=below" in child.raw


def test_composite_xh_bond_children():
    """[COMPOSITE] 内 XH/BOND 子标记解析（容器内 id 引用形式），不作为顶层标记。"""
    text = (
        "[COMPOSITE:row]"
        "[STRUCT:CCC=O,id=pr]"
        "[XH:pr|1]"
        "[BOND:pr|1-2]"
        "[/COMPOSITE]"
    )
    tags = parse_tags(text)
    assert [t.type for t in tags] == ["COMPOSITE"]
    children = tags[0].args[1]
    assert [c.type for c in children] == ["STRUCT", "XH", "BOND"]
    assert children[1].args == ["pr", "1"]
    assert children[2].args == ["pr", "1-2"]

