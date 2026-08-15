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
        "chem_invalid": 0,
        "renderable": 0,
        "render_ok": 0,
        "render_fail": 0,
        "needs_correction": 0,
        "by_type": {},      # 按标记类型统计（第 4 项）
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
        # 化学正确率维度（T2-5）：原因带「化学校验：」前缀的非法标记子集
        stats["chem_invalid"] += sum(
            1 for r in invalid if r.reason.startswith("化学校验："))

        # 按标记类型累计（tags/合法/非法/化学失败）
        def _acc(bt: dict, ttype: str, **kw):
            slot = bt.setdefault(ttype, {"tags": 0, "valid": 0, "invalid": 0,
                                         "chem_invalid": 0, "renderable": 0,
                                         "render_ok": 0, "render_fail": 0})
            for k, v in kw.items():
                slot[k] += v
            return slot

        for tag in tags:
            _acc(stats["by_type"], tag.type, tags=1)
        for tag in valid:
            _acc(stats["by_type"], tag.type, valid=1)
        for r in invalid:
            _acc(stats["by_type"], r.tag.type, invalid=1)
            if r.reason.startswith("化学校验："):
                _acc(stats["by_type"], r.tag.type, chem_invalid=1)

        render_ok = render_fail = 0
        renderable = 0
        for tag in valid:
            if tag.type == "REASONING":
                continue
            renderer = RENDERER_REGISTRY.get(tag.type)
            if renderer is None:
                continue
            renderable += 1
            try:
                out = renderer(*tag.args)
            except Exception:
                out = None
            ok = bool(out) and not out.startswith("（")
            if ok:
                render_ok += 1
            else:
                render_fail += 1
            _acc(stats["by_type"], tag.type, renderable=1,
                 render_ok=1 if ok else 0, render_fail=0 if ok else 1)
        stats["render_ok"] += render_ok
        stats["render_fail"] += render_fail
        stats["renderable"] += renderable

        if invalid or render_fail:
            stats["needs_correction"] += 1
        stats["responses"].append({
            "question": q,
            "status": "ok",
            "tags": len(tags),
            "invalid": len(invalid),
            "truncated": bool(trunc),
            "llm_output": resp,
            "invalid_details": [(r.tag.raw, r.reason) for r in invalid],
            "render_fail": render_fail,
        })
    return stats


def evaluate_route(questions: list, max_corrections: int = 2) -> dict:
    """端到端路由评估：每问题走 process_question（含关键词预判直 pro /
    flash 首跑失败升级 / 部分修正闭环），统计路由效果。

    与 evaluate_compliance 的区别：后者只测单模型（MODEL_NAME）首次输出
    质量（prompt 基线）；本函数测真实管线（路由 + 修正 + 降级）的最终结果，
    回答的是"路由把哪些题救回来了、还剩多少降级"。

    统计口径：
    - main_pass: flash 一遍过（无失败，未升级）
    - upgrade_triggered: 命中难题关键词直 pro，或 flash 首跑失败升级
    - keyword_direct: 其中命中关键词直接走 pro 的题数
    - unresolved_tags: 最终未解决（resolved=False）标记总数
    - degraded_answers: 输出文本含"图示无法渲染"（降级）的回答数
    - corrections_after_upgrade: 升级后修正仍失败的题数（diag 有 round>=1）
    """
    from .config import settings
    import app as _app
    from app import process_question

    keywords = getattr(settings.llm, "upgrade_keywords", None) \
        or _app._DEFAULT_UPGRADE_KEYWORDS
    has_route = bool((settings.llm.upgrade_model_name or "").strip())

    stats = {
        "total": len(questions),
        "main_pass": 0,
        "upgrade_triggered": 0,
        "keyword_direct": 0,
        "unresolved_tags": 0,
        "degraded_answers": 0,
        "corrections_after_upgrade": 0,
        "responses": [],
    }
    for q in questions:
        diag = []
        text = process_question(q, max_corrections=max_corrections,
                                diagnostics=diag)
        keyword_hit = has_route and any(k and k in q for k in keywords)
        # 升级判定：关键词直 pro / 有 main 失败（flash 失败必升级）/
        # 有 upgrade 阶段失败记录
        upgrade_triggered = (keyword_hit
                             or any(d["stage"] == "main" for d in diag)
                             or any(d["stage"] == "upgrade" for d in diag))
        unresolved = [d for d in diag if d.get("resolved") is False]
        degraded = bool(text) and "图示无法渲染" in text
        corrections_failed_after = any(d.get("round", 0) >= 1 for d in diag)

        if keyword_hit:
            stats["keyword_direct"] += 1
        if upgrade_triggered:
            stats["upgrade_triggered"] += 1
        else:
            stats["main_pass"] += 1
        stats["unresolved_tags"] += len(unresolved)
        if degraded:
            stats["degraded_answers"] += 1
        if corrections_failed_after:
            stats["corrections_after_upgrade"] += 1
        stats["responses"].append({
            "question": q,
            "keyword_hit": keyword_hit,
            "upgrade_triggered": upgrade_triggered,
            "degraded": degraded,
            "corrections_failed_after": corrections_failed_after,
            "unresolved": len(unresolved),
            "text": text or "",          # 最终回答全文（含渲染后 TikZ/降级提示）
            "diag": [
                {"round": d.get("round"), "stage": d.get("stage"),
                 "resolved": d.get("resolved"),
                 "reason": (d.get("reason") or "")[:120]}
                for d in diag
            ],
        })
    return stats


def format_route_report(stats: dict) -> str:
    total = stats["total"]
    mp = stats["main_pass"]
    up = stats["upgrade_triggered"]
    return "\n".join([
        "路由评估报告",
        "============",
        f"问题数: {total}",
        f"主模型（flash）一遍过: {mp}（{_pct(mp, total)}）",
        f"升级触发: {up}（{_pct(up, total)}）"
        f"，其中关键词直 pro: {stats['keyword_direct']}",
        f"升级后修正仍失败: {stats['corrections_after_upgrade']}",
        f"最终未解决标记: {stats['unresolved_tags']}",
        f"降级回答数: {stats['degraded_answers']}"
        f"（{_pct(stats['degraded_answers'], total)}）",
    ])


def format_route_detail(stats: dict, output_limit: int = 300) -> str:
    lines = ["逐问题路由详情", "=============="]
    for i, r in enumerate(stats["responses"], 1):
        flags = []
        if r["keyword_hit"]:
            flags.append("关键词直 pro")
        elif r["upgrade_triggered"]:
            flags.append("flash 失败升级")
        if r["corrections_failed_after"]:
            flags.append("升级后修正仍失败")
        if r["degraded"]:
            flags.append("降级")
        lines.append(f"\n[{i}] 问题：{r['question']}")
        lines.append(f"    状态：{'、'.join(flags) if flags else 'flash 一遍过'}"
                     f"（未解决 {r['unresolved']}）")
        for d in r["diag"]:
            lines.append(f"    ✗ round {d['round']}[{d['stage']}]"
                         f"{'✗未解决' if d['resolved'] is False else '✓已解决'}"
                         f" → {d['reason']}")
        text = r.get("text") or ""
        if text:
            shown = text if len(text) <= output_limit \
                else text[:output_limit] + "…"
            lines.append(f"    最终回答：\n{_indent(shown)}")
    return "\n".join(lines)


def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "—"


def format_detail(stats: dict, output_limit: int = 300) -> str:
    """逐问题详情：每个问题的状态、标记情况、非法标记原因与 LLM 输出摘要。

    output_limit: 每条 LLM 输出的最大显示字符数（完整版用 --detail-file 写文件）。
    """
    lines = ["逐问题详情", "=========="]
    for i, r in enumerate(stats["responses"], 1):
        lines.append(f"\n[{i}] 问题：{r['question']}")
        if r["status"] == "llm_fail":
            lines.append("    状态：LLM 调用失败（无输出）")
            continue
        flags = []
        if r["truncated"]:
            flags.append("截断")
        if r["invalid"]:
            flags.append(f"{r['invalid']} 非法")
        if r["render_fail"]:
            flags.append(f"{r['render_fail']} 渲染失败")
        lines.append(f"    状态：ok（标记 {r['tags']} 个"
                     + (f"，{'、'.join(flags)}" if flags else "") + "）")
        for raw, reason in r["invalid_details"]:
            lines.append(f"    ✗ {raw} → {reason}")
        out = r["llm_output"]
        shown = out if len(out) <= output_limit else out[:output_limit] + "…"
        lines.append(f"    LLM 输出：\n{_indent(shown)}")
    return "\n".join(lines)


def _indent(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + ln for ln in text.splitlines())


def format_report(stats: dict) -> str:
    total = stats["total"]
    tags = stats["tags"]
    renderable = stats["renderable"]
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
        f"  其中化学校验失败: {stats['chem_invalid']}"
        f"（化学正确率 {_pct(tags - stats['chem_invalid'], tags)}）",
        f"可渲染标记: {renderable}",
        f"渲染成功: {stats['render_ok']}（{_pct(stats['render_ok'], renderable)}）",
        f"渲染失败: {stats['render_fail']}",
        "",
        f"需要修正的回答数: {stats['needs_correction']}（{_pct(stats['needs_correction'], total)}）",
        f"标记遵循率: {_pct(stats['valid'], tags)}",
    ]
    if stats.get("by_type"):
        lines += ["", format_by_type(stats["by_type"])]
    return "\n".join(lines)


def format_by_type(by_type: dict) -> str:
    """按标记类型统计表：每种标记的 总数/合法/非法/化学失败/渲染/遵循率。

    第 4 项：定位 LLM 最容易写错的标记类型（比只看总数更可操作）。
    """
    from .tag_validator import tag_name

    def disp_w(s: str) -> int:
        # 显示宽度：CJK/全角按 2 列（对齐用）
        return sum(2 if ord(c) > 0x2E7F else 1 for c in s)

    def lj(s: str, w: int) -> str:
        return s + " " * max(0, w - disp_w(s))

    header = ("类型", "标记", "合法", "非法", "化学失败", "可渲染",
              "渲染成功", "渲染失败", "遵循率")
    widths = [disp_w(h) for h in header]
    rows = []
    for ttype, s in sorted(by_type.items()):
        row = (tag_name(ttype), str(s["tags"]), str(s["valid"]),
               str(s["invalid"]), str(s["chem_invalid"]), str(s["renderable"]),
               str(s["render_ok"]), str(s["render_fail"]),
               _pct(s["valid"], s["tags"]))
        rows.append(row)
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], disp_w(cell))
    lines = ["按标记类型统计", "=============="]
    lines.append("  ".join(lj(h, widths[i]) for i, h in enumerate(header)))
    for row in rows:
        lines.append("  ".join(lj(cell, widths[i])
                               for i, cell in enumerate(row)))
    return "\n".join(lines)


if __name__ == "__main__":
    # Windows GBK 控制台打印含 ₆/CJK 的 LLM 输出会 UnicodeEncodeError，
    # 与 composite.py 同款处理：强制 UTF-8、不可编码字符替换
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # 解析选项：--questions-file / --detail-file 各带 1 个文件参数，
    # --report-only / --route 为标志；其余位置参数才是问题（避免选项被当作问题）
    args = sys.argv[1:]
    questions = []
    detail_file = None
    report_only = False
    route_mode = False
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--questions-file" and i + 1 < len(args):
            with open(args[i + 1], encoding="utf-8") as f:
                questions = [ln.strip() for ln in f
                             if ln.strip() and not ln.strip().startswith("#")]
            i += 2
        elif a == "--detail-file" and i + 1 < len(args):
            detail_file = args[i + 1]
            i += 2
        elif a == "--report-only":
            report_only = True
            i += 1
        elif a == "--route":
            route_mode = True
            i += 1
        elif a.startswith("--"):
            print(f"未知选项: {a}")
            i += 1
        else:
            questions.append(a)
            i += 1
    if not questions:
        print("用法: python -m core.metrics \"问题1\" \"问题2\" ...")
        print("      python -m core.metrics --questions-file questions.txt")
        print("      --detail-file FILE 把每条 LLM 完整输出写入文件（终端仍打印摘要）")
        print("      --report-only 终端只打印统计报告，不打印逐问题详情")
        print("      --route 端到端路由评估（process_question 含关键词直 pro /")
        print("              flash 失败升级 / 部分修正；默认模式为单模型首次输出基线）")
        sys.exit(1)
    if route_mode:
        stats = evaluate_route(questions)
        print(format_route_report(stats))
        if not report_only:
            print()
            print(format_route_detail(stats))  # 终端截断版
        if detail_file:
            with open(detail_file, "w", encoding="utf-8") as f:
                f.write(format_route_detail(stats, output_limit=1 << 30))
        sys.exit(0)
    stats = evaluate_compliance(questions)
    print(format_report(stats))
    if not report_only:
        print()
        print(format_detail(stats))
    if detail_file:
        with open(detail_file, "w", encoding="utf-8") as f:
            f.write(format_detail(stats, output_limit=1 << 30))  # 完整输出
