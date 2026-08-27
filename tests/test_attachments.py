# -*- coding: utf-8 -*-
"""tests/test_attachments.py — core/attachments.py 附件构建单元测试。

运行: python -m pytest tests/test_attachments.py -v
"""

import time

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
    """全部编译失败：空列表。"""
    monkeypatch.setattr(att, "compile_tikz_to_png", lambda code: None)
    assert att.build_attachments(ANSWER_WITH_TIKZ, "https://host",
                                 dir_path=tmp_path) == []


def test_prune_by_quota_files(monkeypatch, tmp_path):
    """配额滚动：超出 max_files 时删最旧，保留最新。"""
    monkeypatch.setattr(att, "compile_tikz_to_png", lambda code: FAKE_PNG)
    # 先造 3 张旧图（mtime 递增：old1 最旧，old3 最新）
    for name in ("old1.png", "old2.png", "old3.png"):
        p = tmp_path / name
        p.write_bytes(FAKE_PNG)
        import os
        os.utime(p, (10, 10))
    result = att.build_attachments(ANSWER_WITH_TIKZ, "https://host",
                                   dir_path=tmp_path, max_files=2)
    assert len(result) == 1
    # 目录里只剩 2 张：最旧的 old1/old2 被删，old3 + 新生成留着
    remaining = sorted(p.name for p in tmp_path.glob("*.png"))
    assert remaining == ["old3.png", result[0]["fileUrl"].rsplit("/files/", 1)[-1]]


def test_prune_by_quota_bytes(monkeypatch, tmp_path):
    """超出 max_bytes 时删最旧，直到总大小达标。

    注意：_prune_attachments 在新文件写入**之前**执行，只约束已有文件；
    max_bytes 设得比已有 old 图还小，old 即被清。
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
