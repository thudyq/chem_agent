# -*- coding: utf-8 -*-
"""tests/test_metrics.py — core/metrics.py 按标记类型统计（第 4 项）单元测试。

mock ask_llm 返回固定输出，验证 evaluate_compliance 的 by_type 累计与
format_by_type 表格输出（不调用真实 LLM）。
"""

import pytest

import core.metrics as metrics


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    """ask_llm 返回含多种标记的固定输出（STRUCT 合法、REACTION 非法）。

    evaluate_compliance 内部从 core.llm_client 延迟导入 ask_llm，
    故 monkeypatch 目标为 llm_client 而非 metrics 模块。
    """
    answers = [
        "苯：[STRUCT:c1ccccc1] 硝化：[REACTION:c1ccccc1|CC|H2SO4]",
        "乙醇：[STRUCT:CCO]",
    ]

    def fake_ask(q, **kw):
        return answers.pop(0)

    import core.llm_client as llm
    monkeypatch.setattr(llm, "ask_llm", fake_ask)


def test_by_type_accumulates(monkeypatch):
    """by_type 按标记类型累计 总数/合法/非法/化学失败/渲染。"""
    stats = metrics.evaluate_compliance(["q1", "q2"])
    bt = stats["by_type"]
    # STRUCT×2 全合法可渲染
    assert bt["STRUCT"]["tags"] == 2
    assert bt["STRUCT"]["valid"] == 2
    assert bt["STRUCT"]["invalid"] == 0
    assert bt["STRUCT"]["renderable"] == 2
    assert bt["STRUCT"]["render_ok"] == 2
    # REACTION×1 不守恒 → 非法 + 化学失败
    assert bt["REACTION"]["tags"] == 1
    assert bt["REACTION"]["valid"] == 0
    assert bt["REACTION"]["invalid"] == 1
    assert bt["REACTION"]["chem_invalid"] == 1
    assert bt["REACTION"]["renderable"] == 0
    # 合计与顶层统计一致
    assert stats["tags"] == 3
    assert stats["valid"] == 2
    assert stats["invalid"] == 1
    assert stats["chem_invalid"] == 1


def test_format_by_type_table():
    """format_by_type 输出含表头与各类型行、遵循率列。"""
    by_type = {
        "STRUCT": {"tags": 2, "valid": 2, "invalid": 0, "chem_invalid": 0,
                   "renderable": 2, "render_ok": 2, "render_fail": 0},
        "REACTION": {"tags": 1, "valid": 0, "invalid": 1, "chem_invalid": 1,
                     "renderable": 0, "render_ok": 0, "render_fail": 0},
    }
    out = metrics.format_by_type(by_type)
    assert "按标记类型统计" in out
    assert "类型" in out and "遵循率" in out
    assert "结构式" in out and "100.0%" in out      # STRUCT 中文名 + 100%
    assert "反应方程式" in out and "0.0%" in out    # REACTION 中文名 + 0%
    # 对齐：表头在首行、数据行随后
    lines = out.splitlines()
    assert lines[1] == "=============="
    assert "类型" in lines[2]


def test_format_report_includes_by_type():
    """format_report 在 stats 含 by_type 时追加类型统计表。"""
    stats = metrics.evaluate_compliance(["q1", "q2"])
    out = metrics.format_report(stats)
    assert "标记遵循率报告" in out
    assert "按标记类型统计" in out
    assert "标记总数: 3" in out


def test_format_report_without_by_type_ok():
    """旧 stats（无 by_type 字段）不崩溃——向后兼容。"""
    stats = {"total": 1, "llm_fail": 0, "truncated": 0, "tags": 1,
             "valid": 1, "invalid": 0, "chem_invalid": 0, "renderable": 1,
             "render_ok": 1, "render_fail": 0, "needs_correction": 0}
    out = metrics.format_report(stats)
    assert "标记遵循率报告" in out
    assert "按标记类型统计" not in out
