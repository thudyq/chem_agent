# -*- coding: utf-8 -*-
"""core/replay.py — 离线重放：标记文本的解析/校验/渲染判定与规则 trace。

用法：
    python -m core.replay "含标记的文本"
    python -m core.replay path/to/file.txt
    python -m core.replay -            # 从 stdin 读
    python -m core.replay --full ...   # 附完整 TikZ 输出

用途（归因方法论，Drawbacks §16.4）：测试中发现错误时快速定位"哪条规则
开枪、判断对不对"——逐标记显示校验判定与原因；COMPOSITE 附语义规则
逐项 trace（质子转移配对 / SN2 位点 / EAS σ 脱质子 / 消除成 π 键 /
reaction 守恒）；STRUCT 附标签检查逐项 trace（质子化 / 自由基 /
中文名一致性含拓扑）。退出码：全部通过 0，任一拦截 1。
"""

import argparse
import sys

from core.tag_parser import parse_tags
from core.tag_validator import (
    _RDKIT_OK, _check_chinese_label, _check_eas_rearomatization,
    _check_elimination_pi_target, _check_proton_transfer_pairing,
    _check_radical_label, _check_reaction_sequence, _check_sn2_attack_site,
    _parse_coeff, _parse_mol, check_protonated_label,
    iter_struct_components, validate_tag,
)
from renderers.registry import RENDERER_REGISTRY, render_tag

_RENDER_ERROR_PREFIX = "（"


def _composite_trace(tag) -> tuple:
    """COMPOSITE 语义规则逐项判定（复用校验器的独立检查函数，只读不改）。
    返回 (检查项列表, 电子流模拟操作日志)。"""
    if not _RDKIT_OK:
        return [("语义规则", "SKIP（RDKit 不可用，跳过）")], []
    children = tag.args[1] if len(tag.args) > 1 else []
    comps, comp_mols, mech = {}, {}, []
    for cid, child in iter_struct_components(children):
        raw = (child.args[0] or "").strip() if child.args else ""
        parsed = _parse_coeff(raw)
        bare = parsed[1] if parsed else raw
        comps[cid] = {"smiles": bare, "at": child.attrs.get("at"),
                      "arrow": bool(child.attrs.get("arrow"))}
        if bare:
            try:
                comp_mols[cid] = _parse_mol(bare)
            except Exception:
                comp_mols[cid] = None
    for c in children:
        if c.type == "MECHARROW":
            mech.append(c)
        elif c.type == "BLOCK":
            mech.extend(bc for bc in (c.args[0] if c.args else [])
                        if bc.type == "MECHARROW")
    layout = (tag.args[0] or "").split(",")[0].strip()
    checks = [
        ("质子转移配对", _check_proton_transfer_pairing(mech, comp_mols, comps)),
        ("SN2 进攻位点", _check_sn2_attack_site(mech, comp_mols)),
        ("EAS σ 脱质子方向", _check_eas_rearomatization(mech, comp_mols)),
        ("消除成 π 键方向", _check_elimination_pi_target(mech, comp_mols)),
    ]
    if layout == "reaction":
        checks.append(("reaction 守恒",
                       _check_reaction_sequence(children, comps)))
    # 电子流模拟（P1）：机理箭头能否推出声明产物；失败时附操作日志
    from core.electron_sim import verify_composite_electron_flow
    sim_reason, sim_trace = verify_composite_electron_flow(
        children, comps, comp_mols)
    checks.append(("电子流模拟", sim_reason))
    return checks, sim_trace


def _struct_trace(tag) -> list:
    """STRUCT 标签检查逐项判定。"""
    if not _RDKIT_OK:
        return [("标签检查", "SKIP（RDKit 不可用，跳过）")]
    smi = (tag.args[0] or "").strip() if tag.args else ""
    label = tag.args[1] if len(tag.args) > 1 else ""
    mol = _parse_mol(smi)
    if mol is None:
        return [("SMILES 解析", "FAIL（无法解析）")]
    return [
        ("质子化一致性", check_protonated_label(mol, label)),
        ("自由基一致性", _check_radical_label(mol, label)),
        ("中文名一致性（含拓扑）", _check_chinese_label(mol, label)),
    ]


def _print_trace(checks: list) -> None:
    width = max((len(name) for name, _ in checks), default=0)
    for name, reason in checks:
        status = "PASS" if not reason else "FAIL"
        print(f"    {name.ljust(width)}  {status}"
              + (f" — {reason}" if reason else ""))


def replay(text: str, full: bool = False) -> int:
    """重放一段文本的标记管线判定，打印报告；返回退出码（0 全过 / 1 有拦截）。"""
    tags = parse_tags(text)
    if not tags:
        print("（纯文本：未解析到任何标记）")
        return 0
    n_bad = 0
    for i, tag in enumerate(tags, 1):
        print(f"\n== 标记 {i}/{len(tags)}: {tag.type} ==")
        raw_preview = tag.raw if len(tag.raw) <= 200 else tag.raw[:200] + " …"
        print(f"原文: {raw_preview}")
        if tag.type == "REASONING":
            print("判定: （思考块，不校验不渲染）")
            continue
        vr = validate_tag(tag)
        if vr.ok:
            print("判定: ✓ 通过")
        else:
            n_bad += 1
            print(f"判定: ✗ 拦截 — {vr.reason}")
        if tag.type == "COMPOSITE":
            print("  语义规则 trace:")
            checks, sim_trace = _composite_trace(tag)
            _print_trace(checks)
            if sim_trace:
                print("  电子流操作日志:")
                for t in sim_trace:
                    print(f"    | {t}")
        elif tag.type == "STRUCT":
            print("  标签检查 trace:")
            _print_trace(_struct_trace(tag))
        if not vr.ok:
            print("渲染: （跳过——校验未过，管线中不会进渲染器）")
            continue
        if tag.type not in RENDERER_REGISTRY:
            print(f"渲染: （{tag.type} 无注册渲染器）")
            continue
        try:
            out = render_tag(tag)
        except Exception as e:  # noqa: BLE001 — 渲染器异常即失败串展示
            out = f"（{tag.type} 渲染异常：{e}）"
        if out is None:
            print("渲染: （返回 None）")
        elif out.startswith(_RENDER_ERROR_PREFIX):
            print(f"渲染: ✗ {out}")
        else:
            print(f"渲染: ✓ 成功（{len(out)} 字符 TikZ）")
            if full:
                print(out)
    print(f"\n== 汇总: {len(tags)} 个标记，通过 {len(tags) - n_bad}，拦截 {n_bad} ==")
    return 1 if n_bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="离线重放标记文本的校验/渲染判定")
    ap.add_argument("source", nargs="?", default="-",
                    help="文本内容 / 文件路径 / '-' 或省略从 stdin 读")
    ap.add_argument("--full", action="store_true", help="附完整 TikZ 输出")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    src = args.source
    if src == "-":
        text = sys.stdin.read()
    elif src.endswith(".txt") and __import__("os").path.exists(src):
        with open(src, encoding="utf-8") as f:
            text = f.read()
    else:
        text = src
    return replay(text, full=args.full)


if __name__ == "__main__":
    raise SystemExit(main())
