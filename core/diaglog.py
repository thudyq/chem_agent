# -*- coding: utf-8 -*-
"""core/diaglog.py — 完整诊断落盘（journald 摘要之外的完整记录）。

背景（用户反馈）：journald 里的 [api]/[process_question] 打印是
**截断摘要**（raw[:60] / reason[:120]）——长标记（如 COMPOSITE）出错时
看不到 LLM 具体写了什么、哪里错了。本模块把完整诊断落盘到 JSONL 文件
（默认 data/diagnostics.jsonl，已被 .gitignore 覆盖）：

- 每个失败标记一行：{"type":"failure", ... 完整 raw + 完整 reason + round/
  stage/resolved}；
- 每条回答一行：{"type":"answer", ... 最终采用的原始标记文本（渲染前，
  供本地 python -m core.replay 精确重放）}。

★ 隐私与权限（安全审查 R12）
------------------------------------
这个文件里有**真人的提问原文**（`question`，带图时还含视觉模型对图的描述）
与模型原始输出（`raw`）；只有凭证是**指纹**（不含密钥）。因此：
* 文件以 **0600** 创建/纠正（见 `_open_owner_only`），别让同机其他用户读到；
* **清小搭（/v1）与网页写的是同一个文件**，靠 `cid` 前缀区分
  （`chatcmpl-*` = 清小搭，`web-*` = 网页）；
* Streamlit **不得**再往这个文件写（它会截断），它用自己的
  `data/streamlit_diagnostics.jsonl`；
* 页面上有对应的一句话告知（`web/index.html` 的设置面板与输入区提示）。
要"只留元信息、不留原文"时，改这里：`question` 置空 / 换哈希，`raw` 不写。

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


def _open_owner_only(p: Path):
    """以 **0600（只属主可读写）** 打开日志文件（安全审查 R12）。

    为什么：这个文件里有**真人的提问原文**与模型原始输出（只不含密钥）。
    默认 umask 下新建文件是 0644 —— 本机任何用户都能读。这里显式建 0600，
    并对**已存在的**文件补一次 chmod（把它从 0644 纠正过来，无需人工操作）。
    """
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.chmod(p, 0o600)          # 尽力而为（Windows 上只影响只读位）
    except OSError:
        pass
    return os.fdopen(fd, "a", encoding="utf-8")


def log_request(question: str, cid: str, diag: list = None,
                raw_answer: str = None, path=None,
                max_bytes: int = _MAX_BYTES,
                credential: str = "default") -> None:
    """落盘一次请求的完整诊断。diag 为 process_question 的 diagnostics
    列表（可为空）；raw_answer 为最终采用的原始标记文本（可无）。

    credential: 凭证指纹（`core.credentials.fingerprint()`，**不含密钥本身**）
    ——BYOK 网页路径用它区分不同用户来源，便于按 key 归因失败模式。
    """
    try:
        p = _resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        lines = []
        for d in diag or []:
            lines.append(json.dumps({
                "ts": ts, "cid": cid, "type": "failure",
                "credential": credential,
                "question": question,
                "round": d.get("round"), "stage": d.get("stage"),
                "tag_type": d.get("type"), "resolved": d.get("resolved"),
                "raw": d.get("raw", ""),          # 完整原文，不截断
                "reason": d.get("reason", ""),    # 完整原因，不截断
            }, ensure_ascii=False))
        if raw_answer:
            lines.append(json.dumps({
                "ts": ts, "cid": cid, "type": "answer",
                "credential": credential,
                "question": question, "raw": raw_answer,
            }, ensure_ascii=False))
        if not lines:
            return
        with _LOCK:
            _rotate_if_needed(p, max_bytes)
            with _open_owner_only(p) as f:
                for ln in lines:
                    f.write(ln + "\n")
    except Exception as e:  # 诊断落盘是旁路，任何异常不影响主流程
        print(f"[diaglog] 诊断落盘失败（已忽略）: {e}")
