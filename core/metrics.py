# -*- coding: utf-8 -*-
"""core/metrics.py — 标记遵循率指标工具（P4）。

对一组问题跑真实 LLM 调用，统计标记遵循率与渲染健康度：
- 标记总数 / 合法数 / 非法数（契约校验层判定）
- 渲染成功 / 失败数
- 输出截断数（内容完整性自检）
- 需要修正的回答数（含非法标记或渲染失败）

用途：量化 prompt / 渲染器改动的效果——改进前后各跑一次对比遵循率。

运行：
    python -m core.metrics "问题1" "问题2" ...
    python -m core.metrics --questions-file questions.txt
"""

import sys


def evaluate_compliance(questions: list, *, max_corrections: int = 1) -> dict:
    """对问题列表跑一次 LLM 输出，统计标记遵循指标。

    注意：只评估"首次 LLM 输出"的标记质量（不做 P2 修正重试），
    因此指标反映的是 prompt 引导质量本身。
    """
    from .llm_client import _is_truncated, ask_llm
    from .tag_parser import parse_tags
    from .tag_validator import validate_tags
    from renderers.registry import RENDERER_REGISTRY

    stats = {
        "total": len(questions),
        "llm_fail": 0,
        "truncated": 0,
        "tags": 0,
        "valid": 0,
        "invalid": 0,
        "render_ok": 0,
        "render_fail": 0,
        "needs_correction": 0,
        "responses": [],
    }
    for q in questions:
        resp = ask_llm(q)
        if not resp:
            stats["llm_fail"] += 1
            stats["responses"].append({"question": q, "status": "llm_fail"})
            continue
        trunc = _is_truncated(resp)
        if trunc:
            stats["truncated"] += 1

        tags = parse_tags(resp)
        valid, invalid = validate_tags(tags)
        stats["tags"] += len(tags)
        stats["valid"] += len(valid)
        stats["invalid"] += len(invalid)

        render_ok = render_fail = 0
        for tag in valid:
            if tag.type == "REASONING":
                continue
            renderer = RENDERER_REGISTRY.get(tag.type)
            if renderer is None:
                continue
            try:
                out = renderer(*tag.args)
            except Exception:
                out = None
            if out and not out.startswith("（"):
                render_ok += 1
            else:
                render_fail += 1
        stats["render_ok"] += render_ok
        stats["render_fail"] += render_fail

        if invalid or render_fail:
            stats["needs_correction"] += 1
        stats["responses"].append({
            "question": q,
            "status": "ok",
            "tags": len(tags),
            "invalid": len(invalid),
            "truncated": bool(trunc),
        })
    return stats


def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "—"


def format_report(stats: dict) -> str:
    total = stats["total"]
    tags = stats["tags"]
    lines = [
        "标记遵循率报告",
        "==============",
        f"问题数: {total}",
        f"LLM 调用失败: {stats['llm_fail']}",
        f"输出截断: {stats['truncated']}",
        "",
        f"标记总数: {tags}",
        f"  合法: {stats['valid']}（{_pct(stats['valid'], tags)}）",
        f"  非法: {stats['invalid']}（{_pct(stats['invalid'], tags)}）",
        f"渲染成功: {stats['render_ok']}（{_pct(stats['render_ok'], tags)}）",
        f"渲染失败: {stats['render_fail']}",
        "",
        f"需要修正的回答数: {stats['needs_correction']}（{_pct(stats['needs_correction'], total)}）",
        f"标记遵循率: {_pct(stats['valid'], tags)}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    questions = []
    if "--questions-file" in sys.argv:
        i = sys.argv.index("--questions-file")
        with open(sys.argv[i + 1], encoding="utf-8") as f:
            questions = [ln.strip() for ln in f if ln.strip()]
    else:
        questions = sys.argv[1:]
    if not questions:
        print("用法: python -m core.metrics \"问题1\" \"问题2\" ...")
        print("      python -m core.metrics --questions-file questions.txt")
        sys.exit(1)
    print(format_report(evaluate_compliance(questions)))
