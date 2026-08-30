# -*- coding: utf-8 -*-
"""core/answer_cache.py — 回答缓存：发出的最终文本 → 原始标记文本。

用途（20260828，多轮对话标记恢复）：清小搭平台会在后续请求的历史消息里
原样带回我们发出的 assistant 回答（渲染后文本）。本模块缓存
"sha256(发出的 content) → 原始标记文本"（含 [COMPOSITE] 等渲染标记、
剥离 [REASONING]），使下一轮 LLM 能看到上一轮"画了什么"（标记语法
层面），而不是只剩散文。

设计约束：
- 服务多线程（uvicorn 事件循环 + 流式 work 线程）——所有访问加锁；
- 容量上限（LRU 淘汰）+ TTL 双保险；重启丢失属正常，调用方回退到
  原有剥离逻辑，不影响功能。
"""

import hashlib
import re
import threading
import time
from collections import OrderedDict

_TTL_SECONDS = 24 * 3600     # 历史标记恢复的有效期（跨天对话足够）
_MAX_ENTRIES = 1000          # LRU 容量上限（每条约几 KB，内存可控）

_LOCK = threading.Lock()
_CACHE: "OrderedDict[str, tuple[float, str, dict]]" = OrderedDict()

_REASONING_RE = re.compile(r"\s*\[REASONING\].*?\[/REASONING\]\s*",
                           re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """剥离 [REASONING] 思考块（过程而非结论，不进历史）并收敛空行。"""
    text = _REASONING_RE.sub("\n\n", text or "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def store(content: str, raw_markup: str, meta: dict = None) -> None:
    """登记一条回答映射。content = 实际发给客户端的文本（渲染后）；
    raw_markup = 渲染前的标记文本；meta = 可选处理结果（如
    {"failed": bool, "reason": str}，供下轮"修正/去锚定"时判断并回传失败原因）。
    纯文本回答（二者一致）不缓存。"""
    if not content or not raw_markup or content == raw_markup:
        return
    key = hashlib.sha256(content.encode("utf-8")).hexdigest()
    with _LOCK:
        _CACHE[key] = (time.time(), _strip_reasoning(raw_markup), meta or {})
        _CACHE.move_to_end(key)
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)


def lookup(content: str) -> str | None:
    """按发出的 content 找回原始标记文本；未登记/过期返回 None。"""
    raw, _meta = lookup_full(content)
    return raw


def lookup_full(content: str) -> tuple:
    """按发出的 content 找回 (原始标记文本, meta)；未登记/过期返回 (None, None)。"""
    if not content:
        return None, None
    key = hashlib.sha256(content.encode("utf-8")).hexdigest()
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None, None
        ts, raw, meta = entry
        if time.time() - ts > _TTL_SECONDS:
            _CACHE.pop(key, None)
            return None, None
        _CACHE.move_to_end(key)
        return raw, meta or {}


def clear() -> None:
    """清空缓存（测试用）。"""
    with _LOCK:
        _CACHE.clear()
