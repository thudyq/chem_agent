# -*- coding: utf-8 -*-
"""core/capabilities.py — 端点能力表（学习 + 探测）。

背景（端点能力表：思考参数的学习与记忆）
--------------------------------------------------------------
不同 OpenAI 兼容端点在"思考参数"上的行为差异很大，且**无法预先枚举**：

* 强制思考、不允许关闭（GLM 报错原文："该模型始终思考，不支持关闭思考；请使用
  low、high 或 max。"——**中文、不含英文字段名**，任何按字段名匹配的规则都会漏判）；
* 不接受 `thinking` 字段（Gemini 的 OpenAI 兼容层）；
* 接受字段但**只认部分取值**（Google 官方论坛实测：Gemini 3 Preview 拒
  `reasoning_effort=medium`，接受 `low`/`high`）；
* 完全纯文本、没有思考概念。

因此本模块**不做任何厂商/模型映射表**，只做两件事：

1. **行为判定**：调用失败（HTTP 400）时，判断"摘掉某个字段后能否成功"——
   成则说明该字段是原因，记住；不成则说明是真实错误，原样报错（不掩盖）。
2. **记忆**：把结论按 `(端点 host, 模型, Key 指纹)` 分键缓存，之后不再重复试错。

设计要点
--------
* 键含 **Key 指纹**：同一 host 同一模型名，在不同 Key 下可能被中转平台路由到
  不同上游（能力不同），去掉指纹会串味。键存短哈希，**不存密钥明文**。
* **LRU 2048 + TTL 7 天**：多用户 BYOK 下每个用户一份条目，容量不足时会淘汰；
  淘汰的后果只是"重新学一次 400"，**自愈、无正确性风险**。
* 并发安全：多线程（SSE 工作线程）可能同时学同一个键，各多付一次 400，无正确性影响。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .config import EFFORT_ORDER

# ---------------------------------------------------------------- 常量

LRU_LIMIT = 2048
TTL_SECONDS = 7 * 24 * 3600

# toggle 取值
TOGGLE_UNKNOWN = "unknown"
TOGGLE_BOTH = "both"            # 可开可关
TOGGLE_FORCE_ON = "force_on"    # 只能开（拒绝 disabled）
TOGGLE_UNSUPPORTED = "unsupported"   # 没有思考概念（拒绝 enabled）

# effort 取值集合的特殊标记
EFFORT_ALL = "all"          # 未观察到拒绝，按全部支持对待（乐观：先发再学）
EFFORT_NONE = "none"        # 全部取值都被拒 → 不支持强度


@dataclass
class Capability:
    """一个 (host, model, key) 组合的能力事实。"""

    toggle: str = TOGGLE_UNKNOWN
    # 被**接受**的强度取值集合；EFFORT_ALL 表示"尚无拒绝记录"
    effort_values: set = field(default_factory=lambda: set(EFFORT_ORDER))
    effort_all: bool = True                 # True = 还没观察到拒绝，按全都支持试
    max_tokens_ok: bool | None = None       # None = 未知
    observed_thinking: bool | None = None   # 响应里是否出现过思考内容
    learned_at: float = field(default_factory=time.time)

    # ---- 查询 ----

    def allows_off(self) -> bool | None:
        """能否关闭思考。None = 未知。"""
        if self.toggle == TOGGLE_BOTH:
            return True
        if self.toggle in (TOGGLE_FORCE_ON, TOGGLE_UNSUPPORTED):
            return False
        return None

    def allows_on(self) -> bool | None:
        if self.toggle == TOGGLE_BOTH:
            return True
        if self.toggle == TOGGLE_UNSUPPORTED:
            return False
        if self.toggle == TOGGLE_FORCE_ON:
            return True
        return None

    def supports_effort(self, value: str) -> bool:
        """该强度取值是否可用（未知时乐观放行，让请求去试）。"""
        if self.effort_all:
            return True
        return value in self.effort_values

    def best_effort_at_most(self, value: str) -> str | None:
        """在"不超过 value"的范围内取该端点允许的最高档；都不允许返回 None。

        用于"用户选了某个档位但端点不支持"时就近换档（如 Gemini 拒 medium →
        退到 low；拒 high → 退到 low）。
        """
        try:
            idx = EFFORT_ORDER.index(value)
        except ValueError:
            idx = len(EFFORT_ORDER) - 1
        for cand in reversed(EFFORT_ORDER[: idx + 1]):
            if self.supports_effort(cand):
                return cand
        # 范围内全被拒：尝试更高的（总比不发参数更接近用户意图）
        for cand in EFFORT_ORDER[idx + 1:]:
            if self.supports_effort(cand):
                return cand
        return None

    def describe(self) -> dict:
        """给前端/日志的可读摘要。"""
        return {
            "toggle": self.toggle,
            "effort_values": (list(EFFORT_ORDER) if self.effort_all
                              else [v for v in EFFORT_ORDER if v in self.effort_values]),
            "supports_effort": self.effort_all or bool(self.effort_values),
            "max_tokens_ok": self.max_tokens_ok,
            "observed_thinking": self.observed_thinking,
        }


# ---------------------------------------------------------------- 存储

_lock = threading.Lock()
_table: dict = {}          # key -> (Capability, last_touch_ts)


def _now() -> float:
    return time.time()


def _prune_locked() -> None:
    """TTL 清理 + LRU 裁剪（调用方必须持锁）。"""
    now = _now()
    for k in [k for k, (_c, ts) in _table.items() if now - ts > TTL_SECONDS]:
        _table.pop(k, None)
    if len(_table) > LRU_LIMIT:
        # 按最近使用时间升序淘汰
        for k in sorted(_table, key=lambda k: _table[k][1])[: len(_table) - LRU_LIMIT]:
            _table.pop(k, None)


def make_key(base_url: str, model: str, api_key: str = "") -> str:
    """构造能力表键：host + model + key 指纹（不存明文）。"""
    import hashlib
    from urllib.parse import urlparse
    try:
        host = urlparse(base_url or "").hostname or (base_url or "")
    except ValueError:
        host = base_url or ""
    seed = (api_key or "").encode("utf-8")
    fp = hashlib.sha256(seed).hexdigest()[:12] if seed else "anon"
    return f"{host}|{model or ''}|{fp}"


def get(key: str) -> Capability:
    """取（或创建）能力记录；已过期的条目按新记录对待。"""
    with _lock:
        item = _table.get(key)
        if item is not None:
            cap, ts = item
            if _now() - ts <= TTL_SECONDS:
                _table[key] = (cap, _now())      # 触碰：刷新 LRU
                return cap
            _table.pop(key, None)
        cap = Capability()
        _table[key] = (cap, _now())
        _prune_locked()
        return cap


def peek(key: str) -> Capability | None:
    """只读查看（不存在返回 None，不创建）。"""
    with _lock:
        item = _table.get(key)
        if item is None:
            return None
        cap, ts = item
        return cap if _now() - ts <= TTL_SECONDS else None


# ---------------------------------------------------------------- 学习


def note_success(key: str, *, observed_thinking: bool | None = None) -> None:
    """一次成功请求：刷新时间戳，记录是否观察到思考内容。"""
    cap = get(key)
    with _lock:
        cap.learned_at = _now()
        if observed_thinking is not None:
            if cap.observed_thinking is None:
                cap.observed_thinking = observed_thinking
            else:
                # 只从"观察到"升级为 True，不因一次没看到就改判 False
                cap.observed_thinking = cap.observed_thinking or observed_thinking


def note_field_rejected(key: str, field: str, value: str | None = None) -> Capability:
    """记录"某字段（或某取值）被端点拒绝"——仅在**摘掉它之后请求成功**时调用。

    语义（行为判定，不解析错误文本）：

    * `field="thinking"` 且 `value="disabled"` → 端点强制思考（`force_on`）
    * `field="thinking"` 且 `value="enabled"`  → 端点不支持思考（`unsupported`）
    * `field="reasoning_effort"` → 把该取值从可用集合移除；全空则标记不支持强度
    * `field="max_tokens"` → 端点不接受该字段
    """
    cap = get(key)
    with _lock:
        if field == "thinking":
            if value == "disabled":
                cap.toggle = TOGGLE_FORCE_ON
            elif value == "enabled":
                cap.toggle = TOGGLE_UNSUPPORTED
        elif field == "reasoning_effort":
            cap.effort_values.discard(value or "")
            cap.effort_all = False
        elif field == "max_tokens":
            cap.max_tokens_ok = False
        cap.learned_at = _now()
    return cap


def note_accepted(key: str, field: str, value: str | None = None) -> None:
    """记录"某字段/取值被接受"（用于把乐观集合收敛为确定结论）。"""
    cap = get(key)
    with _lock:
        if field == "thinking":
            # 只有"成功关闭"是确定结论（说明既可开也可关）；
            # "成功开启"不能推出"可关闭"（GLM 就是开启可用、关闭被拒），
            # 因此不据此改判 toggle——留给后续的拒绝式学习。
            if value == "disabled" and cap.toggle in (TOGGLE_UNKNOWN, TOGGLE_BOTH):
                cap.toggle = TOGGLE_BOTH
        elif field == "reasoning_effort" and value:
            cap.effort_values.add(value)
        elif field == "max_tokens":
            cap.max_tokens_ok = True
        cap.learned_at = _now()


def reset_for_tests() -> None:
    """清空能力表（仅测试用）。"""
    with _lock:
        _table.clear()


def snapshot() -> dict:
    """当前表内容摘要（诊断/测试用，不含密钥）。"""
    with _lock:
        return {k: cap.describe() for k, (cap, _ts) in _table.items()}


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    k = make_key("https://open.bigmodel.cn/api/paas/v4", "glm-5.3-flash", "sk-secret-xyz")
    print("键（不含明文）:", k)
    cap = note_field_rejected(k, "thinking", "disabled")
    print("GLM 场景 → toggle:", cap.toggle, "允许关闭:", cap.allows_off())
    k2 = make_key("https://generativelanguage.googleapis.com/v1beta/openai",
                  "gemini-3.7-flash", "sk-secret-abc")
    note_field_rejected(k2, "reasoning_effort", "medium")
    cap2 = get(k2)
    print("Gemini 场景 → 支持 medium:", cap2.supports_effort("medium"),
          "| 支持 low/high:", cap2.supports_effort("low"), cap2.supports_effort("high"))
    print("用户选 medium 时就近换档 →", cap2.best_effort_at_most("medium"))
    print("快照:", snapshot().__len__(), "条")
