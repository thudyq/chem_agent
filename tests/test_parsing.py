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
