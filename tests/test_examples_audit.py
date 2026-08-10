# -*- coding: utf-8 -*-
"""tests/test_examples_audit.py — 示例一致性长期回归（审计）。

把 prompts / README / 文档 / 源码 docstring 与 __main__ demo 中的全部标记示例
提取出来，过真实 P1 + 化学校验（core.tag_validator），防止"自带示例写错"
（历史案例：20260808 硝酸的 prompt 示例误写为硝酸根 [O-][N+](=O)[O-]、
README 的 c1ccccc1NO2 非法 SMILES，均由该校验口径抓获）。

过滤规则（宁可漏报、不可误报）：
- 占位格式说明（如 [HBOND:id|from-to,...]、[ENERGY:点序列]）按占位符正则跳过；
- 故意错误示例（反面教材 / 容错 demo / 校验拦截演示）按上下文关键词跳过。

另含渲染器/解析器 __main__ 演示运行检查（子进程，rc=0 且无意外渲染失败串）。

运行: python -m pytest tests/test_examples_audit.py -v
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit", reason="rdkit 未安装，跳过示例审计")

from core.tag_parser import parse_tags
from core.tag_validator import validate_tags

_ROOT = Path(__file__).resolve().parent.parent

# 审计的文本文件（不存在时自动跳过——instructions/ 与 fast_test/ 不入库）
_TEXT_FILES = [
    "prompts/system_prompt.txt",
    "README.md",
    "instructions/Test-Method.md",
    "instructions/Chemical-Notation.md",
    "instructions/AGENT.md",
    "fast_test/llm_output.md",
]
# 审计的源码目录（docstring 与 __main__ demo 中的字面标记）
_SRC_DIRS = ["renderers", "core", "utils"]

# 含 __main__ 离线演示的模块（子进程运行检查；联网/编译型入口不在此列）
_DEMO_MODULES = [
    "renderers.structure", "renderers.arrow", "renderers.reaction",
    "renderers.composite", "renderers.energy", "renderers.newman",
    "renderers.lewis", "renderers.stereo", "renderers.charge",
    "renderers.hbond", "renderers.retro", "renderers.layout",
    "core.tag_parser", "core.tag_validator",
]

# 占位格式说明（文档中的语法示意，非真实示例）
_PLACEHOLDER_RE = re.compile(
    r"SMILES|点序列|角度|名称|转化名|反应类型|布局|条件|文本|引用名|"
    r"点序号|端点|序号|原子序号|反应物|产物|转化|类型|组件|ref|\bid\b|"
    r"式\d|目标|\.\.\.|[A-Za-z]+\\")

# 故意错误示例的上下文关键词（该行及前 3 行命中即跳过）
_SKIP_CTX_RE = re.compile(
    r"错误|反例|反面教材|无效|畸形|故意|乱写|容错|应提示|错误示范|避坑|"
    r"失败|违规|半截|不符|拦截|拒绝|占位|bad|invalid", re.IGNORECASE)


def _audit_files() -> list:
    files = [_ROOT / f for f in _TEXT_FILES]
    for d in _SRC_DIRS:
        files += sorted((_ROOT / d).glob("*.py"))
    return [p for p in files if p.exists()]


def _unexpected_invalids(path: Path) -> list:
    """文件中未通过校验且不属于占位/故意示例的标记 [(raw, reason), ...]。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    _, invalid = validate_tags(parse_tags(text))
    problems = []
    for r in invalid:
        raw = r.tag.raw
        # 单行短标记 + 占位词 → 占位格式说明，跳过
        if "\n" not in raw and len(raw) < 80 and _PLACEHOLDER_RE.search(raw):
            continue
        # 失败原因的「引用片段」全部是占位词（式1、布局、组件id 等）
        # 或无实义字符（源码字面量幻影标签，如 "[ENERGY:" 匹配到远处引号）
        # → 跳过；多行/长 raw 不做整体占位过滤——幻影/跨段容器不得静默豁免真实错误
        quoted = re.findall(r"「([^」]*)」", r.reason or "")
        if quoted and all(_PLACEHOLDER_RE.search(q)
                          or not re.search(r"[\w一-鿿]", q)
                          for q in quoted):
            continue
        # 上下文（该行及前 3 行）含故意错误示例关键词 → 跳过
        line_no = text[: r.tag.start_pos].count("\n")
        ctx = "\n".join(lines[max(0, line_no - 3): line_no + 1])
        if _SKIP_CTX_RE.search(ctx):
            continue
        problems.append((raw.replace("\n", " ⏎ "), r.reason))
    return problems


@pytest.mark.parametrize("path", _audit_files(),
                         ids=lambda p: str(p.relative_to(_ROOT)))
def test_examples_valid(path: Path):
    """文件中的标记示例全部通过真实 P1 + 化学校验（占位/故意示例除外）。"""
    problems = _unexpected_invalids(path)
    assert not problems, (
        f"{path.name} 中 {len(problems)} 个示例标记未通过校验：\n"
        + "\n".join(f"  ✗ {raw[:80]} → {reason}" for raw, reason in problems))


@pytest.mark.parametrize("module", _DEMO_MODULES)
def test_renderer_main_demos(module: str):
    """__main__ 离线演示可正常运行（rc=0），且无意外渲染失败串。"""
    proc = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(_ROOT), timeout=120,
    )
    assert proc.returncode == 0, f"{module} 演示异常退出:\n{proc.stderr[-500:]}"
    prev = ""
    for ln in proc.stdout.splitlines():
        if "渲染失败" in ln and not _SKIP_CTX_RE.search(prev + ln):
            pytest.fail(f"{module} 演示出现意外渲染失败: {ln.strip()[:100]}")
        prev = ln
