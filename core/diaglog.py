# -*- coding: utf-8 -*-
"""core/diaglog.py — 完整诊断落盘（journald 摘要之外的完整记录）。

背景（20260828 用户反馈）：journald 里的 [api]/[process_question] 打印是
**截断摘要**（raw[:60] / reason[:120]）——长标记（如 COMPOSITE）出错时
看不到 LLM 具体写了什么、哪里错了。本模块把完整诊断落盘到 JSONL 文件
（默认 data/diagnostics.jsonl，已被 .gitignore 覆盖）：

- 每个失败标记一行：{"type":"failure", ... 完整 raw + 完整 reason + round/
  stage/resolved}；
- 每条回答一行：{"type":"answer", ... 最终采用的原始标记文本（渲染前，
  供本地 python -m core.replay 精确重放）}。

journald 摘要打印保持不变（健康检查不能刷屏）。文件超上限（默认 8MB）
轮转为 .1（覆盖旧 .1）。任何写盘异常静默降级（不拖垮回答主流程）。
"""

import json
import os
import threading
import time
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" \
    / "diagnostics.jsonl"
_MAX_BYTES = 8 * 1024 * 1024   # 轮转阈值（超过即重命名为 .1）

_LOCK = threading.Lock()


def _resolve_path(path=None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get("DIAG_LOG_PATH", "") or _DEFAULT_PATH)


def _rotate_if_needed(p: Path, max_bytes: int) -> None:
    if p.exists() and p.stat().st_size > max_bytes:
        p.replace(p.with_suffix(".1.jsonl"))


def log_request(question: str, cid: str, diag: list = None,
                raw_answer: str = None, path=None,
                max_bytes: int = _MAX_BYTES) -> None:
    """落盘一次请求的完整诊断。diag 为 process_question 的 diagnostics
    列表（可为空）；raw_answer 为最终采用的原始标记文本（可无）。"""
    try:
        p = _resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        lines = []
        for d in diag or []:
            lines.append(json.dumps({
                "ts": ts, "cid": cid, "type": "failure",
                "question": question,
                "round": d.get("round"), "stage": d.get("stage"),
                "tag_type": d.get("type"), "resolved": d.get("resolved"),
                "raw": d.get("raw", ""),          # 完整原文，不截断
                "reason": d.get("reason", ""),    # 完整原因，不截断
            }, ensure_ascii=False))
        if raw_answer:
            lines.append(json.dumps({
                "ts": ts, "cid": cid, "type": "answer",
                "question": question, "raw": raw_answer,
            }, ensure_ascii=False))
        if not lines:
            return
        with _LOCK:
            _rotate_if_needed(p, max_bytes)
            with p.open("a", encoding="utf-8") as f:
                for ln in lines:
                    f.write(ln + "\n")
    except Exception as e:  # 诊断落盘是旁路，任何异常不影响主流程
        print(f"[diaglog] 诊断落盘失败（已忽略）: {e}")
