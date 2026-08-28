# -*- coding: utf-8 -*-
"""tests/test_attachments.py — core/attachments.py 附件构建单元测试。

运行: python -m pytest tests/test_attachments.py -v
"""

import pytest

import core.attachments as att


FAKE_PNG = b"\x89PNG\r\n\x1a\nfake"
ANSWER_WITH_TIKZ = (
    "苯的结构式：\n"
    "\\begin{tikzpicture}\n  \\draw (0,0) -- (1,0);\n\\end{tikzpicture}\n"
    "以上为苯环。"
)
ANSWER_NO_TIKZ = "苯的分子式是 C6H6。"


def test_extract_code_blocks():
    blocks = att.extract_code_blocks(ANSWER_WITH_TIKZ)
    assert len(blocks) == 1
    assert blocks[0].startswith("\\begin{tikzpicture}")
    assert att.extract_code_blocks(ANSWER_NO_TIKZ) == []
    assert att.extract_code_blocks("") == []


def test_build_attachments_writes_files(monkeypatch, tmp_path):
    """编译成功：写 PNG 文件 + 生成 4 必填 + fileSize 的条目。"""
    monkeypatch.setattr(att, "compile_tikz_to_png", lambda code: FAKE_PNG)
    result = att.build_attachments(ANSWER_WITH_TIKZ, "https://host",
                                   dir_path=tmp_path)
    assert len(result) == 1
    a = result[0]
    assert a["fileType"] == "image"
    assert a["mimeType"] == "image/png"
    assert a["fileSize"] == len(FAKE_PNG)
    name = a["fileUrl"].rsplit("/files/", 1)[-1]
    assert a["fileUrl"] == f"https://host/files/{name}"
    assert (tmp_path / name).read_bytes() == FAKE_PNG
    assert a["fileName"].endswith(".png")


def test_build_attachments_no_tikz(tmp_path):
    """无代码块：空列表，不建目录。"""
    assert att.build_attachments(ANSWER_NO_TIKZ, "https://host",
                                 dir_path=tmp_path / "none") == []


def test_build_attachments_compile_failure(monkeypatch, tmp_path):
    """全部编译失败：返回与块对齐的 [None]（占位，无图、不写文件）。"""
    monkeypatch.setattr(att, "compile_tikz_to_png", lambda code: None)
    result = att.build_attachments(ANSWER_WITH_TIKZ, "https://host",
                                   dir_path=tmp_path)
    assert len(result) == 1
    assert result[0] is None


_MULTI_BLOCK = (
    "前置。\n"
    "\\begin{tikzpicture}\\draw(0,0)--(1,0);\\end{tikzpicture}\n"   # 块1 成功
    "\\begin{tikzpicture}\\draw(0,0)--(2,0);\\end{tikzpicture}\n"   # 块2 失败
    "\\begin{tikzpicture}\\draw(0,0)--(3,0);\\end{tikzpicture}\n"   # 块3 成功
    "后置。"
)


def test_partial_failure_keeps_position(monkeypatch, tmp_path):
    """多块部分编译失败：返回与块一一对应（失败块 None 占位），
    不写该失败块文件；success 块 url 按原始位置对齐（修复 20260827：
    曾因只按成功块序号填 url，导致中间失败块被赋予下一张图的 url、
    后面的真实成功图被挤掉）。"""
    monkeypatch.setattr(att, "compile_tikz_to_png", lambda code: (
        b"\x89PNG" + b"\x00" * 20 if "1,0" in code or "3,0" in code else None))
    result = att.build_attachments(_MULTI_BLOCK, "https://host",
                                   dir_path=tmp_path, max_files=100)
    assert len(result) == 3            # 与 3 个块对齐
    assert result[0] is not None and result[2] is not None
    assert result[0]["fileUrl"].startswith("https://host/files/")
    assert result[1] is None           # 失败块占位
    # 只写成功块文件
    assert len(list(tmp_path.glob("*.png"))) == 2
    # 正文替换：块1→图1、块2→提示、块3→图3，不错位
    urls = [a["fileUrl"] if a else None for a in result]
    display = att.replace_code_blocks_with_images(_MULTI_BLOCK, urls)
    assert "![化学图示-1](" in display
    assert "图示未能渲染" in display
    assert "![化学图示-3](" in display
    assert "![化学图示-3](" not in display.replace("![化学图示-3](", "", 1)
    assert "tikzpicture" not in display
    # 顺序正确：图1 在提示之前、图3 在提示之后
    assert display.index("![化学图示-1](") < display.index("图示未能渲染") \
        < display.index("![化学图示-3](")


def test_prune_by_quota_files(monkeypatch, tmp_path):
    """配额滚动：超出 max_files 时删最旧，保留最新（含本批新图）。"""
    monkeypatch.setattr(att, "compile_tikz_to_png", lambda code: FAKE_PNG)
    # 先造 3 张旧图（mtime 递增：old1 最旧，old3 最新——必须错开，
    # 相同 mtime 的删除顺序依赖文件系统 glob 序，不确定）
    import os
    for i, name in enumerate(("old1.png", "old2.png", "old3.png"), 1):
        p = tmp_path / name
        p.write_bytes(FAKE_PNG)
        os.utime(p, (i * 10, i * 10))
    result = att.build_attachments(ANSWER_WITH_TIKZ, "https://host",
                                   dir_path=tmp_path, max_files=2)
    assert len(result) == 1
    # 目录里只剩 2 张：最旧的 old1/old2 被删，old3 + 新生成留着
    remaining = sorted(p.name for p in tmp_path.glob("*.png"))
    assert remaining == sorted(
        ["old3.png", result[0]["fileUrl"].rsplit("/files/", 1)[-1]])


def test_prune_by_quota_bytes(monkeypatch, tmp_path):
    """超出 max_bytes 时删最旧，直到总大小达标。

    注意：_prune_attachments 在新文件写入**之后**执行（本批新图列入 keep
    保护不删）；max_bytes 设得比已有 old 图还小，old 即被清。
    """
    monkeypatch.setattr(att, "compile_tikz_to_png", lambda code: FAKE_PNG)
    old = tmp_path / "old.png"
    old.write_bytes(b"\x89PNG" * 10)   # 40 字节
    import os
    os.utime(old, (10, 10))
    result = att.build_attachments(ANSWER_WITH_TIKZ, "https://host",
                                   dir_path=tmp_path,
                                   max_bytes=len(FAKE_PNG))  # < 40 → 清 old
    assert len(result) == 1
    assert not old.exists()
