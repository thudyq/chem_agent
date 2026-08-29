# -*- coding: utf-8 -*-
"""tests/test_diaglog.py — 完整诊断落盘（core/diaglog.py）测试。

背景（20260828）：journald 打印是截断摘要（raw[:60]），长 COMPOSITE 出错
看不到 LLM 具体写了什么；完整记录落盘 JSONL 供离线 replay 重放。

运行: python -m pytest tests/test_diaglog.py -v
"""

import json

from core import diaglog

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
