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
    """补充：NEWMAN。"""
    tags = parse_tags("[NEWMAN:CC,60]")
    assert len(tags) == 1
    assert tags[0].type == "NEWMAN"
    assert tags[0].args == ["CC", "60"]


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


def test_reactionmech_basic():
    """[REACTIONMECH] 基础解析：反应物 | 产物 | 条件 | 机理箭头。"""
    tags = parse_tags("[REACTIONMECH:CCl;[OH-]|[Cl-];CO|SN2|1:0>0:0,0:1>2:0]")
    assert len(tags) == 1
    t = tags[0]
    assert t.type == "REACTIONMECH"
    assert t.args[0] == "CCl;[OH-]"
    assert t.args[1] == "[Cl-];CO"
    assert t.args[2] == "SN2"
    assert t.args[3] == "1:0>0:0,0:1>2:0"


def test_reactionmech_no_conditions():
    """[REACTIONMECH] 省略条件时，args[2] 为空串。"""
    tags = parse_tags("[REACTIONMECH:CCl;[OH-]|[Cl-];CO||1:0>0:0]")
    assert len(tags) == 1
    t = tags[0]
    assert t.args[0] == "CCl;[OH-]"
    assert t.args[1] == "[Cl-];CO"
    assert t.args[2] == ""
    assert t.args[3] == "1:0>0:0"


def test_reactionmech_fishhook():
    """[REACTIONMECH] 鱼钩箭头（单电子转移）解析。"""
    tags = parse_tags("[REACTIONMECH:C=C;[Br]|[CH2]CBr|hv|1:0>>0:0,0:0>>0:1]")
    assert len(tags) == 1
    t = tags[0]
    assert t.type == "REACTIONMECH"
    assert t.args[3] == "1:0>>0:0,0:0>>0:1"


def test_reactionmech_numbering_flag():
    """[REACTIONMECH] 第五段 numbering 标志解析，缺省为空串。"""
    tags = parse_tags("[REACTIONMECH:CCl;[OH-]|[Cl-];CO|SN2|1:0>0:0|numbering]")
    t = tags[0]
    assert len(t.args) == 5
    assert t.args[4] == "numbering"

    tags = parse_tags("[REACTIONMECH:CCl;[OH-]|[Cl-];CO|SN2|1:0>0:0]")
    assert tags[0].args[4] == ""


def test_mech_numbering_flag():
    """[MECH] 第三段 numbering 标志解析，缺省为空串。"""
    tags = parse_tags("[MECH:CCl.[OH-]|2>0,0>1|numbering]")
    t = tags[0]
    assert t.type == "MECH"
    assert t.args == ["CCl.[OH-]", "2>0,0>1", "numbering"]

    tags = parse_tags("[MECH:CCl.[OH-]|2>0,0>1]")
    assert tags[0].args == ["CCl.[OH-]", "2>0,0>1", ""]


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

