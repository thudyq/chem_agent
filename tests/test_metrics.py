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
        "苯：[STRUCT:c1ccccc1] 硝化：[COMPOSITE:reaction][STRUCT:CCO,id=a][PLUS][STRUCT:O,id=w][ARROW:type=single][STRUCT:CC=O,id=b][/COMPOSITE]",
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
    assert bt["COMPOSITE"]["tags"] == 1
    assert bt["COMPOSITE"]["valid"] == 0
    assert bt["COMPOSITE"]["invalid"] == 1
    assert bt["COMPOSITE"]["chem_invalid"] == 1
    assert bt["COMPOSITE"]["renderable"] == 0
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
        "COMPOSITE": {"tags": 1, "valid": 0, "invalid": 1, "chem_invalid": 1,
                     "renderable": 0, "render_ok": 0, "render_fail": 0},
    }
    out = metrics.format_by_type(by_type)
    assert "按标记类型统计" in out
    assert "类型" in out and "遵循率" in out
    assert "结构式" in out and "100.0%" in out      # STRUCT 中文名 + 100%
    assert "复合图" in out and "0.0%" in out        # COMPOSITE 中文名 + 0%
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


# ---------- 端到端路由评估（evaluate_route） ----------

def _enable_route_config(monkeypatch):
    """启用路由配置（core.config.settings，非 app.settings——evaluate_route
    内部从 core.config 延迟导入）。"""
    import types
    import core.config as cfg
    monkeypatch.setattr(
        cfg, "settings",
        types.SimpleNamespace(
            llm=types.SimpleNamespace(
                upgrade_model_name="deepseek-v4-pro",
                upgrade_keywords=())))  # 空 → 用 app 内置默认关键词


def test_evaluate_route_counts(monkeypatch):
    """路由三种结局统计：flash 一遍过 / 关键词直 pro 降级 / flash 失败升级救回。"""
    _enable_route_config(monkeypatch)

    def fake_pq(q, max_corrections=2, history=None, progress_callback=None,
                correction_callback=None, diagnostics=None, responses=None):
        d = diagnostics
        if "机理" in q:   # 命中关键词 → 直 pro → 失败降级
            if responses is not None:
                responses.append("机理标记文本：[COMPOSITE:reaction][STRUCT:CCO,id=a][ARROW][STRUCT:CC=O,id=b][/COMPOSITE]")
            d.append({"round": 0, "stage": "upgrade", "type": "REACTION",
                      "raw": "[COMPOSITE:reaction][STRUCT:CCO,id=a][ARROW][STRUCT:CC=O,id=b][/COMPOSITE]", "reason": "化学校验：不守恒",
                      "friendly": "（反应方程式图示无法渲染，已省略）",
                      "resolved": False})
            return "机理回答（反应方程式图示无法渲染，已省略）"
        if "氧化" in q:   # flash 失败 → 升级 pro 救回
            if responses is not None:
                responses.append("flash 标记文本：[STRUCT:bad]")
                responses.append("pro 标记文本：[STRUCT:c1ccccc1]")
            d.append({"round": 0, "stage": "main", "type": "STRUCT",
                      "raw": "[STRUCT:bad]", "reason": "无效 SMILES",
                      "friendly": "（结构式图示无法渲染，已省略）",
                      "resolved": True})
            return "氧化回答 [STRUCT:c1ccccc1]"
        if responses is not None:
            responses.append("苯标记文本：[STRUCT:c1ccccc1]")
        return "苯是 [STRUCT:c1ccccc1]。"   # flash 一遍过

    import app
    monkeypatch.setattr(app, "process_question", fake_pq)

    stats = metrics.evaluate_route(
        ["苯的结构式", "介绍苯的硝化反应机理", "乙醇氧化方程式"])
    assert stats["main_pass"] == 1
    assert stats["upgrade_triggered"] == 2
    assert stats["keyword_direct"] == 1
    assert stats["unresolved_tags"] == 1
    assert stats["degraded_answers"] == 1
    # 原始 LLM 输出记录：每题 1+ 条（氧化题为 flash + pro 两条）
    by_q = {r["question"]: r for r in stats["responses"]}
    assert by_q["苯的结构式"]["llm_outputs"] == ["苯标记文本：[STRUCT:c1ccccc1]"]
    assert len(by_q["乙醇氧化方程式"]["llm_outputs"]) == 2


def test_format_route_report_and_detail():
    """路由报告与逐题详情输出格式。"""
    stats = {
        "total": 3, "main_pass": 1, "upgrade_triggered": 2,
        "keyword_direct": 1, "unresolved_tags": 1, "degraded_answers": 1,
        "corrections_after_upgrade": 0,
        "responses": [
            {"question": "q1", "keyword_hit": False,
             "upgrade_triggered": False, "degraded": False,
             "corrections_failed_after": False, "unresolved": 0, "diag": [],
             "text": "苯是 [STRUCT:c1ccccc1]。",
             "llm_outputs": ["苯标记文本：[STRUCT:c1ccccc1]"]},
            {"question": "q2", "keyword_hit": True,
             "upgrade_triggered": True, "degraded": True,
             "corrections_failed_after": False, "unresolved": 1,
             "text": "机理回答（反应方程式图示无法渲染，已省略）",
             "llm_outputs": ["机理标记文本：[COMPOSITE:reaction][STRUCT:CCO,id=a][ARROW][STRUCT:CC=O,id=b][/COMPOSITE]"],
             "diag": [{"round": 0, "stage": "upgrade", "resolved": False,
                       "reason": "化学校验：不守恒"}]},
        ],
    }
    rep = metrics.format_route_report(stats)
    assert "路由评估报告" in rep
    assert "主模型（flash）一遍过: 1" in rep
    assert "升级触发: 2" in rep and "关键词直 pro: 1" in rep
    det = metrics.format_route_detail(stats)
    assert "flash 一遍过" in det
    assert "关键词直 pro" in det and "未解决 1" in det
    assert "最终回答" in det and "苯是 [STRUCT:c1ccccc1]" in det
    assert "原始输出（主模型，渲染前）" in det
    assert "苯标记文本：[STRUCT:c1ccccc1]" in det
