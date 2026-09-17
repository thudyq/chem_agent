# -*- coding: utf-8 -*-
"""core/rule_stats.py — 校验规则归因统计。

输入 core.metrics --json-out 导出的 JSON 语料（compliance 首次输出基线，
或 route 端到端——route 只取每题首轮输出做归因），对语料中每个被拦截
标记回答"哪条规则开的枪"：

- 首失败归因：格式/引用类按失败原因前缀映射（校验器早退，首失败即全部）；
- 语义类直接重跑各检查函数，得到全部"开火"规则（含被首失败掩盖的并发
  失败）——STRUCT 六项标签检查、COMPOSITE 六项语义检查；
- autofix 覆盖：按 app.py 顺序试三个确定性自动修复钩子，统计各规则
  拦截量中有多少本可不经 LLM 修复（回读协议的成本估算依据）。

用法：
    python -m core.metrics --questions-file Q.txt --json-out corpus.json --report-only
    python -m core.rule_stats corpus.json [--out report.txt]
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict

from core.tag_parser import parse_tags
from core.tag_validator import (
    _RDKIT_OK, _check_chinese_label, _check_cip_label,
    _check_cistrans_label, _check_radical_charge_conflict,
    _check_radical_label, _parse_coeff, _parse_mol,
    autofix_balance_gap, autofix_mech_bond_endpoint, autofix_stereo_label,
    check_protonated_label, tag_name, validate_tags,
)

# ---------------------------------------------------------------- 首失败归因

_COMP_PREFIX_RE = re.compile(r"^组件 [^:：]+: ")
# BLOCK 内组件的失败（"BLOCK: 块内组件 s2: 无效 SMILES…"）剥到内层原因，
# 按本质归因（如 struct.smiles_invalid）而非笼统归 block 格式
_BLOCK_COMP_PREFIX_RE = re.compile(r"^BLOCK: 块内组件 [^:：]+: ")

# (rule_id, 判定) 顺序敏感：先匹配先得。判定为前缀串或谓词函数。
_FORMAT_RULES = [
    ("struct.smiles_empty", ("SMILES 为空",)),
    ("struct.smiles_invalid", ("无效 SMILES",)),
    ("struct.label_len", ("label 过长",)),
    ("struct.formula_param", ("化学式文本组件",)),
    ("struct.mode", ("未知 STRUCT 模式",)),
    ("struct.chair", ("CHAIR", "SMILES 中未找到环己烷六元环")),
    ("struct.newman", ("键参数", "角度", "缺少角度")),
    ("struct.bond_mark", ("bond 键",)),
    ("struct.charge_mark", ("charge 标注", "charge 原子编号")),
    ("comp.layout", ("未知布局",)),
    ("comp.dup_id", ("组件 id 重复",)),
    ("comp.coeff", ("系数格式错误",)),
    ("comp.mode_layout", ("reaction 布局不支持 mode=newman",)),
    ("comp.arrow_token_mode", ("箭头附件（arrow 令牌）仅支持",)),
    ("comp.formula_mode", ("化学式组件（文本轨道）不支持",
                            "化学式组件不支持")),
    ("comp.empty", ("容器内缺少",)),
    ("comp.energy_layout", ("energy 布局需要", "energy 布局中组件")),
    ("comp.energy_at", ("超出能量点范围", "的 at=")),
    ("energy.points", ("至少需要 3 个能量点", "能量点个数必须为奇数",
                        "能量值", "点序列为空")),
    ("comp.arrow_type", ("ARROW 类型",)),
    ("comp.sup_token", ("ARROW 附件", "箭头附件组件")),
    ("ref.opaque", ("引用立体画法组件",)),
    ("ref.formula", ("引用化学式组件",)),
    ("ref.unknown", ("引用未知组件",)),
    ("ref.charge", ("CHARGE 标注", "CHARGE 原子编号")),
    ("ref.hbond", ("HBOND",)),
    ("ref.xh", ("XH 原子编号", "XH 引用")),
    ("ref.bond", ("BOND 键", "BOND 引用")),
    ("mech.empty", ("MECHARROW 为空",)),
    ("mech.format", ("MECHARROW 格式错误",)),
    ("mech.self_loop", ("MECHARROW 起点与终点相同",)),
    ("mech.cross_step", ("跨越主反应箭头",)),
    ("mech.blank_format", ("MECHARROW 成键空白位端点格式错误",)),
    ("mech.blank_pair", ("MECHARROW 成键空白位配对错误",)),
    ("mech.endpoint", ("MECHARROW 源端点", "MECHARROW 目标端点")),
    ("mech.polar_semantics", ("MECHARROW「",)),  # R1–R4 语义（前缀靠后）
    ("xh.ghost_h", ("可用隐含 H 为",)),
    ("block.empty", ("BLOCK 内缺少",)),
    ("block.arrow_type", ("BLOCK 内箭头类型",)),
    ("block.content", ("BLOCK 内不支持",)),
    ("block.resonance_balance", ("BLOCK",)),  # 其余 BLOCK 失败均为守恒
]


def _first_fail_rule(reason: str) -> str | None:
    """失败原因 → 格式/引用/带前缀语义规则；语义自由文本返回 None（走重跑）。"""
    r = reason or ""
    r = _BLOCK_COMP_PREFIX_RE.sub("", r)
    r = _COMP_PREFIX_RE.sub("", r)
    for rule, prefixes in _FORMAT_RULES:
        if any(r.startswith(p) for p in prefixes):
            return rule
    return None


# ---------------------------------------------------------------- 语义重跑

_STRUCT_LABEL_CHECKS = [
    ("struct.radical_charge_conflict", _check_radical_charge_conflict),
    ("struct.protonated_label", check_protonated_label),
    ("struct.radical_label", _check_radical_label),
    ("struct.chinese_label", _check_chinese_label),
    ("struct.cip_label", _check_cip_label),
    ("struct.cistrans_label", _check_cistrans_label),
]


def _struct_firing(tag) -> list:
    """STRUCT 语义规则全量重跑，返回开火规则列表（执行顺序）。"""
    if not _RDKIT_OK:
        return []
    smi = (tag.args[0] or "").strip() if tag.args else ""
    label = tag.args[1] if len(tag.args) > 1 else ""
    mol = _parse_mol(smi)
    if mol is None:
        return [("struct.smiles_invalid", "SMILES 无法解析")]
    out = []
    for rule, fn in _STRUCT_LABEL_CHECKS:
        reason = fn(mol, label) if fn is not _check_radical_charge_conflict \
            else fn(mol)
        if reason:
            out.append((rule, reason))
    return out


_COMPOSITE_TRACE_RULES = {
    "质子转移配对": "comp.proton_transfer",
    "SN2 进攻位点": "comp.sn2_site",
    "EAS σ 脱质子方向": "comp.eas_rearom",
    "消除成 π 键方向": "comp.elim_pi_target",
    "reaction 守恒": "comp.reaction_conservation",
    "电子流模拟": "comp.electron_flow",
}


def _composite_firing(tag) -> list:
    """COMPOSITE 语义规则全量重跑（复用 replay 的 trace），返回开火规则。"""
    if not _RDKIT_OK:
        return []
    from core.replay import _composite_trace
    checks, _sim_trace = _composite_trace(tag)
    out = []
    for name, reason in checks:
        if reason and not reason.startswith("SKIP"):
            out.append((_COMPOSITE_TRACE_RULES.get(name, f"comp.{name}"),
                        reason))
    return out


# ---------------------------------------------------------------- autofix

def _autofix_hook(tag, reason: str):
    """按 app.py 顺序试确定性自动修复，返回命中的钩子名或 None。"""
    if autofix_mech_bond_endpoint(tag):
        return "autofix_mech_bond_endpoint"
    if autofix_stereo_label(tag):
        return "autofix_stereo_label"
    if autofix_balance_gap(tag, reason):
        return "autofix_balance_gap"
    return None


# ---------------------------------------------------------------- 主流程

def _iter_responses(corpus: dict):
    """统一两种语料模式为 (question, first_pass_text) 迭代器。"""
    mode = corpus.get("mode", "compliance")
    for r in corpus.get("responses", []):
        if r.get("status") == "llm_fail":
            continue
        if mode == "route":
            outs = r.get("llm_outputs") or []
            text = outs[0] if outs else ""
        else:
            text = r.get("llm_output") or ""
        if text:
            yield r.get("question", ""), text


def analyze(corpus: dict) -> dict:
    """对语料逐题重放校验，返回归因统计字典。"""
    stats = {
        "mode": corpus.get("mode", "compliance"),
        "questions": 0, "tags": 0, "valid": 0, "invalid": 0,
        "first_rules": Counter(),       # 首失败归因
        "firing": Counter(),            # 语义规则开火（含并发）
        "by_type": defaultdict(lambda: {"tags": 0, "invalid": 0}),
        "autofix": Counter(),           # 钩子名 → 可修标记数
        "autofix_by_rule": Counter(),   # 首失败规则 → 可修数
        "unattributed": Counter(),      # 归因失败的原因（维护前缀表用）
        "details": [],                  # 逐题明细
    }
    for question, text in _iter_responses(corpus):
        stats["questions"] += 1
        tags = parse_tags(text)
        valid, invalid = validate_tags(tags)
        q_det = {"question": question, "tags": len(tags), "failures": []}
        for t in tags:
            stats["by_type"][t.type]["tags"] += 1
        stats["tags"] += len(tags)
        stats["valid"] += len(valid)
        stats["invalid"] += len(invalid)
        for vr in invalid:
            tag = vr.tag
            stats["by_type"][tag.type]["invalid"] += 1
            reason = vr.reason or ""
            rule = _first_fail_rule(reason)
            firing = []
            if rule is None:
                if tag.type == "STRUCT":
                    firing = _struct_firing(tag)
                elif tag.type == "COMPOSITE":
                    firing = _composite_firing(tag)
                if firing:
                    rule = firing[0][0]   # 执行顺序首个即首失败
                else:
                    rule = "other"
                    stats["unattributed"][reason[:80]] += 1
            else:
                # 格式失败掩盖语义失败时仍重跑语义（并发开火统计）
                if tag.type == "STRUCT":
                    firing = _struct_firing(tag)
                elif tag.type == "COMPOSITE":
                    firing = _composite_firing(tag)
            stats["first_rules"][rule] += 1
            for frule, _ in firing:
                stats["firing"][frule] += 1
            hook = _autofix_hook(tag, reason)
            if hook:
                stats["autofix"][hook] += 1
                stats["autofix_by_rule"][rule] += 1
            q_det["failures"].append({
                "type": tag.type, "raw": tag.raw[:120], "rule": rule,
                "reason": reason, "firing": [f for f, _ in firing],
                "autofix": hook,
            })
        stats["details"].append(q_det)
    return stats


# ---------------------------------------------------------------- 报告

def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "—"


def format_report(stats: dict, corpus_path: str) -> str:
    lines = [
        "校验规则归因统计报告（P0）",
        "==========================",
        f"语料: {corpus_path}（mode={stats['mode']}，{stats['questions']} 题）",
        f"标记: {stats['tags']} 个，通过 {stats['valid']}，"
        f"拦截 {stats['invalid']}（{_pct(stats['invalid'], stats['tags'])}）",
        "",
        "一、首失败归因（哪条规则拦下的）",
        "    规则                          拦截    占比   autofix可修",
    ]
    for rule, n in stats["first_rules"].most_common():
        fix = stats["autofix_by_rule"].get(rule, 0)
        lines.append(f"    {rule:<28} {n:>4}  {_pct(n, stats['invalid']):>6}"
                     f"   {fix if fix else ''}")
    lines += ["", "二、语义规则开火（含被首失败掩盖的并发失败）"]
    if stats["firing"]:
        for rule, n in stats["firing"].most_common():
            lines.append(f"    {rule:<28} {n:>4}")
    else:
        lines.append("    （无）")
    lines += ["", "三、按标记类型", "    类型          总数   拦截"]
    for ttype, s in sorted(stats["by_type"].items()):
        lines.append(f"    {tag_name(ttype):<12} {s['tags']:>4}  {s['invalid']:>4}")
    lines += ["", "四、确定性 autofix 覆盖（本可不经 LLM 修复的拦截）"]
    total_fix = sum(stats["autofix"].values())
    lines.append(f"    合计: {total_fix}/{stats['invalid']}"
                 f"（{_pct(total_fix, stats['invalid'])}）")
    for hook, n in stats["autofix"].most_common():
        lines.append(f"    {hook:<28} {n:>4}")
    if stats["unattributed"]:
        lines += ["", "五、未归因原因（前缀表缺口，需维护）"]
        for reason, n in stats["unattributed"].most_common():
            lines.append(f"    [{n}] {reason}")
    lines += ["", "六、逐题明细"]
    for i, det in enumerate(stats["details"], 1):
        if not det["failures"]:
            continue
        lines.append(f"\n[{i}] {det['question']}")
        for f in det["failures"]:
            lines.append(f"    ✗ {f['type']} → {f['rule']}"
                         + (f"（autofix: {f['autofix']}）" if f["autofix"] else ""))
            lines.append(f"      原因: {f['reason'][:160]}")
            if len(f["firing"]) > 1:
                lines.append(f"      并发: {', '.join(f['firing'][1:])}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="校验规则归因统计（P0）")
    ap.add_argument("corpus", help="core.metrics --json-out 导出的 JSON 语料")
    ap.add_argument("--out", help="报告写入文件（终端同时打印）")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with open(args.corpus, encoding="utf-8") as f:
        corpus = json.load(f)
    report = format_report(analyze(corpus), args.corpus)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
