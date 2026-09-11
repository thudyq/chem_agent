# -*- coding: utf-8 -*-
"""tests/test_diaglog.py — 完整诊断落盘（core/diaglog.py）测试。

背景（20260828）：journald 打印是截断摘要（raw[:60]），长 COMPOSITE 出错
看不到 LLM 具体写了什么；完整记录落盘 JSONL 供离线 replay 重放。

运行: python -m pytest tests/test_diaglog.py -v
"""

import json
from pathlib import Path

import pytest

from core import diaglog

_ROOT = Path(__file__).resolve().parents[1]

_LONG_RAW = ("[COMPOSITE:reaction]"
             "[STRUCT:BrC([H])1C=CC=C[CH+]1,label=σ 络合物,id=sigma]"
             "[ARROW:type=single]"
             "[STRUCT:BrC1=CC=CC=C1,label=溴苯][PLUS][STRUCT:[H+],label=H+]"
             "[MECHARROW:sigma:1-2>sigma:1][MECHARROW:sigma:7-1>sigma:7]"
             "[/COMPOSITE]")     # >60 字符，journald 摘要里会被截断
_LONG_REASON = "方向错误：" + "很长的错误说明" * 30  # >120 字符


def _diag():
    return [{"round": 0, "stage": "upgrade", "type": "COMPOSITE",
             "raw": _LONG_RAW, "reason": _LONG_REASON, "resolved": False}]


def _read_lines(path):
    return [json.loads(ln) for ln in
            path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_full_raw_and_reason_not_truncated(tmp_path):
    """失败标记的完整原文/完整原因落盘（不截断）。"""
    p = tmp_path / "diag.jsonl"
    diaglog.log_request("介绍苯的溴代反应机理", "cid-1", _diag(), path=p)
    lines = _read_lines(p)
    assert len(lines) == 1
    rec = lines[0]
    assert rec["type"] == "failure" and rec["cid"] == "cid-1"
    assert rec["raw"] == _LONG_RAW                    # 完整，未截断
    assert rec["reason"] == _LONG_REASON
    assert rec["question"] == "介绍苯的溴代反应机理"
    assert rec["resolved"] is False


def test_answer_line_written(tmp_path):
    """每条回答附一行最终原始标记文本（供离线 replay）。"""
    p = tmp_path / "diag.jsonl"
    diaglog.log_request("问", "cid-2", [], raw_answer="原始标记文本", path=p)
    lines = _read_lines(p)
    assert len(lines) == 1 and lines[0]["type"] == "answer"
    assert lines[0]["raw"] == "原始标记文本"


def test_empty_diag_no_answer_writes_nothing(tmp_path):
    p = tmp_path / "diag.jsonl"
    diaglog.log_request("问", "cid-3", [], None, path=p)
    assert not p.exists() or not _read_lines(p)


def test_rotation_when_over_max(tmp_path):
    """超过大小上限轮转为 .1（旧 .1 被覆盖），新文件只含新记录。"""
    p = tmp_path / "diag.jsonl"
    diaglog.log_request("q", "cid-old", _diag(), path=p, max_bytes=10)
    diaglog.log_request("q", "cid-new", _diag(), path=p, max_bytes=10)
    assert p.with_suffix(".1.jsonl").exists()
    lines = _read_lines(p)
    assert len(lines) == 1 and lines[0]["cid"] == "cid-new"


def test_write_failure_does_not_raise(tmp_path):
    """写盘异常静默降级（不拖垮主流程）。"""
    bad_dir = tmp_path / "a_dir"
    bad_dir.mkdir()
    # path 指向一个目录 → open 必失败 → 应被吞掉
    diaglog.log_request("q", "cid-x", _diag(), path=bad_dir)


def test_api_non_stream_writes_full_diag(monkeypatch, tmp_path):
    """API 集成：非流式回答的失败标记完整落盘（端到端）。"""
    import api
    from fastapi.testclient import TestClient

    def fake_pipeline(question, history=None, diagnostics=None,
                      responses=None, **k):
        if diagnostics is not None:
            diagnostics.append({"round": 0, "stage": "upgrade",
                                "type": "COMPOSITE", "raw": _LONG_RAW,
                                "reason": _LONG_REASON, "resolved": False})
        if responses is not None:
            responses.append("最终原始标记文本")
        return "回答文本。"

    monkeypatch.setattr(api, "process_question", fake_pipeline)
    monkeypatch.setattr(api, "SERVICE_KEY", "test-key")
    monkeypatch.setenv("DIAG_LOG_PATH", str(tmp_path / "d.jsonl"))
    client = TestClient(api.app)
    resp = client.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "问"}]},
                       headers={"Authorization": "Bearer test-key"})
    assert resp.status_code == 200
    lines = _read_lines(tmp_path / "d.jsonl")
    assert any(r["type"] == "failure" and r["raw"] == _LONG_RAW
               for r in lines)                       # 完整原文落盘
    assert any(r["type"] == "answer"
               and r["raw"] == "最终原始标记文本" for r in lines)


# ------------------------------------------- 隐私与隔离（安全审查 R12，20260911）

def test_log_file_is_owner_only(monkeypatch, tmp_path):
    """★ 日志里有真人提问原文 → 必须 **0600**（只属主可读写）。

    默认 umask 下新建文件是 0644（同机任何用户可读），所以这里既要"建的时候
    就是 0600"，也要对**已存在的**文件补一次 chmod。
    """
    calls = []
    real_chmod = diaglog.os.chmod

    def spy_chmod(path, mode):
        calls.append((str(path), mode))
        return real_chmod(path, mode)

    monkeypatch.setattr(diaglog.os, "chmod", spy_chmod)
    p = tmp_path / "diag.jsonl"
    diaglog.log_request("问", "cid-perm", _diag(), path=p)
    assert (str(p), 0o600) in calls

    import os as _os
    import stat
    if _os.name != "nt":            # Windows 只有只读位，不做 POSIX 权限断言
        assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_existing_loose_file_gets_tightened(monkeypatch, tmp_path):
    """已经存在的 0644 老文件，写入时会被纠正成 0600（无需人工 chmod）。"""
    import os as _os
    import stat
    if _os.name == "nt":
        pytest.skip("Windows 不区分 POSIX 权限位")
    p = tmp_path / "diag.jsonl"
    p.write_text("old\n", encoding="utf-8")
    _os.chmod(p, 0o644)
    diaglog.log_request("问", "cid-fix", _diag(), path=p)
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_streamlit_uses_a_separate_log_file():
    """★ Streamlit **不得**再写线上那份 `data/diagnostics.jsonl`。

    它有"启动时清空"的逻辑（`_flush_diagnostics_file`），两者共用一个文件时，
    跑一次 Streamlit 就会把清小搭/网页的诊断记录抹掉。
    """
    src = (_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert '"streamlit_diagnostics.jsonl"' in src
    assert '/ "diagnostics.jsonl"' not in src, "Streamlit 不能再用线上那个文件"
    # 清空判断必须是**进程级**的模块变量，而不是 st.session_state（那是每会话级）
    assert "_DIAG_FLUSHED" in src
    assert "st.session_state.get(_DIAG_CLEARED_KEY)" not in src


def test_streamlit_flush_is_process_level(monkeypatch, tmp_path):
    """清空只做一次（进程级）：第二次调用不再截断，已有内容保留。"""
    import streamlit_app as sa

    monkeypatch.setattr(sa, "_DIAGNOSTICS_FILE", tmp_path / "sl.jsonl")
    monkeypatch.setattr(sa, "_DIAG_FLUSHED", False)
    sa._flush_diagnostics_file()
    (tmp_path / "sl.jsonl").write_text("keep-me\n", encoding="utf-8")
    sa._flush_diagnostics_file()                      # 第二次不该再清
    assert (tmp_path / "sl.jsonl").read_text(encoding="utf-8") == "keep-me\n"


def test_page_carries_the_privacy_notice():
    """页面上必须有隐私说明（否则"记录提问原文"就是没告知）。"""
    html = (_ROOT / "web/index.html").read_text(encoding="utf-8")
    assert "隐私说明" in html
    assert "提问原文" in html and "不包含" in html and "API Key" in html
    assert "会记录到服务端日志" in html          # 输入区的常驻提示那条
