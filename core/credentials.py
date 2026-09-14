# -*- coding: utf-8 -*-
"""core/credentials.py — 每请求模型凭证与参数（BYOK，bring your own key）。

背景
----
`core.config.settings` 是**导入期冻结**的模块级单例（从 `.env` 读取），
清小搭接入服务（`api.py`）只有一个服务端密钥。要做"任何用户带自己的
API key 来用"的公开网页，凭证必须变成**每请求可变**，而管线
（`app.process_question` + 20 多个模块）全部按 `settings.llm` 读取配置。

方案
----
不改 `settings` 的既有语义，新增一层**请求作用域覆盖**：

    from core.credentials import user_credentials, llm_config

    with user_credentials({"api_key": "sk-...", "model": "deepseek-flash",
                           "thinking": "on", "effort": "low"}):
        process_question(question)      # 整条管线自动用用户凭证与参数

    llm_config()      # 管线内替代 settings.llm
    vision_config()   # 管线内替代 settings.vision

用 `contextvars.ContextVar`（而非全局字典）承载覆盖：

* **线程安全**——`api.py` 的 SSE 在独立线程里跑 `process_question`；
  注意 Python 的**新线程不会自动继承 contextvars**，凡在子线程跑管线的入口
  必须显式 `contextvars.copy_context().run(...)`（已处理：`api._sse_stream`、
  `web_api._sse_stream`、`app._run_with_timeout`）。全局字典在并发多用户下
  会互相串凭证（严重安全问题）。
* 无请求上下文（CLI、测试、清小搭 `/v1`）时 `llm_config()` 返回基准配置，
  行为与改造前一致。

参数体系（请求头 → 每请求凭证，字段见 `_FIELDS`）
-------------------------------------------------------
网页面板 / 请求头 / `.env` 是**同一个参数体系的三张皮**，同名同义：

    面板标签        请求头                      .env
    API Key         X-Chem-Api-Key              API_KEY
    接口地址         X-Chem-Base-Url             BASE_URL
    模型            X-Chem-Model                MODEL_NAME
    思考            X-Chem-Thinking             THINKING_DEFAULT
    思考强度         X-Chem-Effort               EFFORT_DEFAULT
    最大输出         X-Chem-Max-Tokens           MAX_TOKENS
    视觉模型         X-Chem-Vision-Model         VISION_MODEL
    视觉接口地址      X-Chem-Vision-Base-Url      VISION_BASE_URL
    视觉 API Key    X-Chem-Vision-Api-Key       VISION_API_KEY
    视觉思考         X-Chem-Vision-Thinking      VISION_THINKING
    视觉思考强度      X-Chem-Vision-Effort        VISION_EFFORT

安全约定
--------
* 凭证只从**请求头**进入（不由 URL 携带，避免落入访问日志/Referer）。
* `redact()` 给日志用的指纹形如 `sk-abc…f9c2`，**任何日志/异常都不得
  打印完整密钥**。
* 服务端不回显、不落盘用户密钥；`user_credentials` 只在请求生命周期内
  存活（contextmanager 退出即 reset）。
"""

from __future__ import annotations

import hashlib
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Optional

from .config import settings

# 请求作用域的凭证覆盖；None 表示"无覆盖"（用基准配置）。
_CTX: ContextVar[Optional[dict]] = ContextVar("chem_agent_credentials", default=None)

# 凭证字段名 → 允许覆盖的配置项。键同时是前端/请求头里的字段名（规范名）。
_FIELDS = (
    # 主模型
    "api_key", "base_url", "model",
    # 思考参数（两个正交维度）
    "thinking", "effort", "max_tokens",
    # 视觉组（可能完全独立的模型）
    "vision_model", "vision_base_url", "vision_api_key",
    "vision_thinking", "vision_effort",
)

# 基准配置（服务器 .env）。默认即 `core.config.settings`；测试可用
# `set_base_settings_for_tests()` 注入替身，避免"测试 monkeypatch 了调用方模块
# 的 settings、凭证层却仍读旧对象"的隐性耦合。
_BASE = settings
_BASE_LOCK = threading.Lock()


def set_base_settings_for_tests(base) -> None:
    """替换凭证层的基准配置对象（**仅测试用**；传 None 恢复默认）。"""
    global _BASE
    with _BASE_LOCK:
        _BASE = base if base is not None else settings


def base_settings():
    """当前基准配置对象（供测试与诊断使用）。"""
    return _BASE


def normalize(raw: Optional[dict]) -> dict:
    """规范化凭证 dict：去空白、去空值、base_url 去尾部 `/`。

    只保留 `_FIELDS` 中的键；未知键忽略（防止请求头注入任意配置项）。
    """
    if not raw:
        return {}
    out = {}
    for key in _FIELDS:
        val = raw.get(key)
        if val is None:
            continue
        val = str(val).strip()
        if not val:
            continue
        if key in ("base_url", "vision_base_url"):
            val = val.rstrip("/")
        out[key] = val
    return out


def redact(secret: Optional[str]) -> str:
    """密钥指纹（日志用）：`sk-abc…wxyz`（前 6 + 后 4），短密钥只留首字符。绝不返回完整值。"""
    if not secret:
        return "(空)"
    s = str(secret)
    if len(s) <= 8:
        return s[0] + "***"
    return f"{s[:6]}…{s[-4:]}"


def fingerprint(creds: Optional[dict] = None) -> str:
    """凭证指纹（用于按用户分桶：并发限制、失败缓存、能力表隔离）。

    以 api_key + base_url 的 sha256 前 16 位为准——同一用户的多次请求同桶，
    不同用户/不同端点不同桶。无凭证时返回 `"default"`（共享 .env 配置）。
    """
    creds = normalize(creds) if creds is not None else current()
    if not creds or not creds.get("api_key"):
        return "default"
    seed = f"{creds.get('api_key', '')}|{creds.get('base_url', '')}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def current() -> dict:
    """当前请求作用域的凭证覆盖（无则空 dict）。"""
    return _CTX.get() or {}


@contextmanager
def user_credentials(raw: Optional[dict]):
    """在 with 块内把 `raw` 设为当前请求的凭证覆盖（块退出自动还原）。

    支持嵌套（内层优先，退出后回到外层）。`raw` 为 None/空时等价于
    "不覆盖"——仍会压入一层空 dict，避免内层泄漏到外层。

    ★ **不要跨"迭代边界"使用**：`contextvars` 的 reset 要求 set/reset 在
    同一个 Context 内。Starlette 会把 StreamingResponse 的生成器放进线程池
    逐块迭代，相邻 `next()` 可能属于不同 Context —— 此时 `with` 退出时
    `_CTX.reset(token)` 会抛 `ValueError: Token was created in a different
    Context`（实测 SSE 流当场中断）。

    SSE 场景的正确写法见 `apply_credentials()`：把凭证直接传给 SSE 生成器，
    由**工作线程函数内部**（`web_api._sse_stream_inner.work`）调用一次
    `apply_credentials()` —— 那个函数整体跑在同一个 Context 里。
    """
    token = _CTX.set(normalize(raw))
    try:
        yield
    finally:
        _CTX.reset(token)


def apply_credentials(raw: Optional[dict]) -> dict:
    """设置当前 Context 的凭证覆盖并返回它（**只设不重置**）。

    供**生成器**使用：生成器体在迭代时才执行、且可能在线程池里跨 Context
    逐块推进，因此不能配对 reset。这里只在当前 Context 里写入一次，
    该 Context 结束时自动丢弃——安全且无跨 Context 异常。

    ★ **写入点要选对**：`set` 只作用于**当前 Context**。Starlette 逐块迭代
    `StreamingResponse` 时，每次 `next()` 都从请求任务上下文**重新拷贝**一份
    Context，所以在"生成器迭代处"set 的值传不到后面启动工作线程的那一次迭代
    （实测：流式仍回退服务器 `.env`）。凭证必须在工作线程函数**内部**
    set —— 那个函数整体跑在同一个 `copy_context().run(...)` 里。
    """
    creds = normalize(raw)
    _CTX.set(creds)
    return creds


def reset_for_tests() -> None:
    """清空**当前 Context** 的凭证覆盖（测试隔离用）。

    存在的理由：`apply_credentials()` 故意不 reset，正常请求里 Context 用完即弃，
    但 pytest 全程跑在同一线程的同一个 Context 里，不清理就会污染后续用例。
    """
    _CTX.set(None)


# ---------------------------------------------------------------- 主模型

def user_model() -> str:
    """用户**显式指定**的模型名（未指定返回空串）。

    用途：BYOK 下用户填了模型就必须全程用他填的那个——服务器 `.env` 里
    配置的模型不得介入（既未授权、也可能无权访问）。
    """
    return (current().get("model") or "").strip()


def thinking_setting() -> str:
    """当前生效的思考**开关**：`on` / `off`。

    用户显式提供优先；否则用基准配置的 `thinking_default`。
    """
    from .config import normalize_thinking
    creds = current()
    raw = creds.get("thinking") or getattr(_BASE.llm, "thinking_default", "on")
    return normalize_thinking(raw)


def effort_setting() -> str:
    """当前生效的思考**强度**：low/medium/high/max（开关为 off 时无意义）。"""
    from .config import normalize_effort
    creds = current()
    raw = creds.get("effort") or getattr(_BASE.llm, "effort_default", "low")
    return normalize_effort(raw)


def max_tokens_setting() -> int:
    """当前生效的最大输出上限（用户提供优先，否则基准配置默认）。"""
    creds = current()
    raw = creds.get("max_tokens")
    if raw:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            pass
    return int(getattr(_BASE.llm, "max_tokens", 32768) or 32768)


def _with_overrides(obj, changes: dict):
    """在配置对象上应用覆盖。

    生产路径是 frozen dataclass → `dataclasses.replace`；
    测试可能注入 `SimpleNamespace` 替身（早期测试约定），此时做属性合并。
    """
    import dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return replace(obj, **changes)
    merged = dict(vars(obj))
    merged.update(changes)
    return type(obj)(**merged) if _accepts_kwargs(obj) else _Namespace(**merged)


def _accepts_kwargs(obj) -> bool:
    """对象能否用 `type(obj)(**kwargs)` 重建（SimpleNamespace 可以）。"""
    try:
        type(obj)(**vars(obj))
        return True
    except Exception:
        return False


class _Namespace:
    """最小命名空间（`_with_overrides` 在无法重建时的兜底）。"""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def llm_config():
    """当前生效的主 LLM 配置（请求覆盖 ∪ 基准配置）。

    主模型/端点/密钥：用户提供即用用户的，否则用服务器 `.env`
    （清小搭 `/v1` 与 Streamlit 路径即此分支）。
    思考参数：把"开关 + 强度"归一后写回 `thinking_default` / `effort_default`
    （`ask_llm` 读这两个字段），因此调用方无需感知凭证是否存在。
    """
    base = _BASE.llm
    creds = current()
    if not creds:
        return base
    changes = {
        "api_key": creds.get("api_key") or base.api_key,
        "base_url": creds.get("base_url") or base.base_url,
        "model_name": creds.get("model") or base.model_name,
        "thinking_default": thinking_setting(),
        "effort_default": effort_setting(),
        "max_tokens": max_tokens_setting(),
    }
    return _with_overrides(base, changes)


def is_configured() -> bool:
    """当前生效的主 LLM 配置是否可用（供 API 层快速判 400）。"""
    return bool(llm_config().is_configured)


# ---------------------------------------------------------------- 视觉组

def vision_config():
    """当前生效的视觉配置。

    规则——视觉是**可能完全独立**的一个模型：

    1. 无请求覆盖（清小搭 / Streamlit）→ 用基准配置的 `VISION_*`；
    2. 有请求覆盖且**用户填了视觉模型** → 用用户给的；端点/Key 未单独给时
       沿用**用户自己的**主模型端点/Key（多数 OpenAI 兼容端点同一把 Key 即可
       访问视觉模型）；
    3. 有请求覆盖但**未填视觉模型** → **用主模型**（Key/端点/模型名全同）——
       原生多模态模型（deepseek-flash / Gemini / GLM 视觉版）自带视觉，
       此时无需任何额外配置。

    ★ 绝不在用户路径上继承**服务器**的视觉配置（避免把服务器配置的视觉模型
    与计费静默施加到用户请求上）。

    思考参数（VISION_THINKING / VISION_EFFORT）：
    * 留空 = 自动：视觉与主模型是同一个（host+model+key 全同）时**复用主模型**
      设置；否则默认 `off`（识图要快、要省，不需要长思考）。
    """
    base = _BASE.vision
    creds = current()
    if not creds:
        return base

    user_model_name = creds.get("model") or getattr(_BASE.llm, "model_name", "")
    user_base = creds.get("base_url") or ""
    user_key = creds.get("api_key") or ""

    if creds.get("vision_model"):
        # 用户显式指定视觉模型
        v_base = creds.get("vision_base_url") or user_base or base.base_url
        v_key = creds.get("vision_api_key") or user_key or base.api_key
        v_model = creds["vision_model"]
    else:
        # 未指定 → 用主模型
        v_base, v_key, v_model = user_base, user_key, user_model_name

    same_as_main = (v_model == user_model_name and v_base == user_base
                    and v_key == user_key)
    if creds.get("vision_thinking"):
        v_thinking = creds["vision_thinking"]
    elif same_as_main:
        v_thinking = ""            # 空 = 复用主模型（由调用方按同一套规则取值）
    else:
        v_thinking = "off"
    v_effort = creds.get("vision_effort") or ""

    if not (v_model and v_base and v_key):
        # 用户既没填视觉模型、主模型凭证也不全 → 视为未配置
        return _with_overrides(base, {"api_key": "", "base_url": "",
                                      "model_name": "", "thinking": "",
                                      "effort": ""})
    return _with_overrides(base, {"api_key": v_key, "base_url": v_base,
                                  "model_name": v_model,
                                  "thinking": v_thinking, "effort": v_effort})


def vision_is_main() -> bool:
    """当前视觉配置是否就是主模型（决定思考参数能否复用主模型设置）。"""
    base = _BASE
    creds = current()
    if not creds:
        # 无覆盖：基准配置里视觉三项为空也视为"用主模型"
        v = base.vision
        return not v.model_name or (v.model_name == base.llm.model_name)
    v = vision_config()
    return (v.model_name == (creds.get("model") or getattr(base.llm, "model_name", ""))
            and (v.api_key or "") == (creds.get("api_key") or "")
            and (v.base_url or "") == (creds.get("base_url") or ""))


# ---------------------------------------------------------------- 端点校验

def client_host_allowed(base_url: str) -> tuple[bool, str]:
    """校验用户提供的 base_url 可用（安全 + 可用性）。

    规则：
    * 必须是 http/https，且有 host（拒绝 `file://` 等）；
    * **拒绝内网/回环/链路本地/云元数据地址**——服务端会带着用户密钥
      去请求该地址，放任内网地址等于把服务器当 SSRF 跳板（与 api.py
      下载图片的 SSRF 防护同一口径）；
    * 允许 `localhost` / 内网地址**仅当**显式开启
      `WEB_ALLOW_PRIVATE_BASE_URL=1`（本地自测用）。

    返回 (是否允许, 原因)。
    """
    import ipaddress
    import os
    import socket
    from urllib.parse import urlparse

    raw = (base_url or "").strip()
    if not raw:
        return True, ""      # 空 = 用服务器默认端点，无需校验
    try:
        parsed = urlparse(raw)
    except ValueError:
        return False, "端点地址无法解析"
    if parsed.scheme not in ("http", "https"):
        return False, "端点地址必须以 http:// 或 https:// 开头"
    host = (parsed.hostname or "").strip()
    if not host:
        return False, "端点地址缺少主机名"

    if os.environ.get("WEB_ALLOW_PRIVATE_BASE_URL", "").strip() in ("1", "true", "yes"):
        return True, ""

    def _blocked(ip_str: str) -> bool:
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return True
        return (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified)

    try:
        ip = ipaddress.ip_address(host)
        if _blocked(str(ip)):
            return False, "端点地址不能指向内网/本机地址（如为本地模型服务，请管理员开启 WEB_ALLOW_PRIVATE_BASE_URL）"
        return True, ""
    except ValueError:
        pass    # 域名，需 DNS 解析校验
    if host.lower() in ("localhost",) or host.lower().endswith(
            (".local", ".internal", ".localhost", ".lan", ".corp", ".home")):
        return False, "端点地址不能指向本机/内网域名"
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False, f"端点地址无法解析：{host}"
    for info in infos:
        if _blocked(str(info[4][0])):
            return False, "端点地址解析到内网/本机地址"
    return True, ""


# ---------------------------------------------------------------- 并发闸门
# 原实现是模块级 BoundedSemaphore（固定 4）：BYOK 下所有用户共用会互相
# 拖慢，且"用户自带 key 却被服务器闸门排队"没有道理。改为**按凭证指纹
# 分桶**的信号量——同一 key 的请求仍受 max_concurrent 约束（保护该用户
# 自己的 API 限额），不同用户互不阻塞。
_LOCK = threading.Lock()
_SEMS: dict = {}
_SESSION = None


def llm_semaphore():
    """当前凭证对应的并发信号量（同 key 共享一个，上限 max_concurrent）。"""
    fp = fingerprint()
    with _LOCK:
        sem = _SEMS.get(fp)
        if sem is None:
            limit = max(1, int(getattr(_BASE.llm, "max_concurrent", 0) or 1))
            sem = threading.BoundedSemaphore(limit)
            _SEMS[fp] = sem
        return sem


def session():
    """模块级 `requests.Session`（连接池复用；所有 LLM/视觉调用共用）。"""
    global _SESSION
    with _LOCK:
        if _SESSION is None:
            import requests
            _SESSION = requests.Session()
        return _SESSION


def _reset_semaphores_for_tests() -> None:
    """清空信号量池（仅测试用）。"""
    with _LOCK:
        _SEMS.clear()


# ---------------------------------------------------------------- 全局 / 按 IP 在途上限（安全审查 R5）
# 为什么"按凭证分桶"对服务器没有保护：**假 key 是无限的**。攻击者每个请求换
# 一个假 key，就得到一个全新的桶；再把 `base_url` 指向自己控制的、故意不回包的
# 公网地址，就能让每个请求占住一个 worker 直到读超时（`llm_client.READ_TIMEOUT`）。
# 这里加两道**与凭证无关**的闸门，只护服务器自己：
#   1. 全局在途上限 —— 服务器最多同时扛这么多出站调用（内存/线程有上界）；
#   2. 同一客户端 IP 的在途上限 —— 防止单个 IP 把全局额度吃光、把别人饿死。
# 设 0 即关闭该道闸门（本地/测试用）。
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


GLOBAL_MAX_INFLIGHT = _env_int("CHEM_AGENT_MAX_INFLIGHT", 16)
PER_IP_MAX_INFLIGHT = _env_int("CHEM_AGENT_MAX_INFLIGHT_PER_IP", 4)
# 等不到槽位就**快速失败**：把请求堆在队列里只会把资源耗尽推迟成雪崩。
INFLIGHT_WAIT_SECONDS = 5.0


class ServerBusy(RuntimeError):
    """服务器在途请求已达上限（安全审查 R5）。文案面向终端用户。"""

    def __init__(self, msg: str = "服务器繁忙：同时在处理的请求过多，请稍后重试"):
        super().__init__(msg)


# 请求作用域的客户端 IP：由 Web 层设置；Streamlit / CLI 不设 → 跳过按 IP 的闸门
_REQUEST_IP: ContextVar[Optional[str]] = ContextVar("chem_agent_request_ip",
                                                    default=None)

_inflight_lock = threading.Lock()
_global_sem: Optional[threading.BoundedSemaphore] = None
_ip_inflight: dict = {}


def set_request_ip(ip: Optional[str]) -> None:
    """把当前请求的客户端 IP 放进 contextvar（供按 IP 的在途闸门分桶）。

    ★ 必须由**能拿到真实客户端地址的那一层**（`core/web_api.py` 的
    `_client_ip`，即 uvicorn 净化后的值）来设置——不要在这里自己解析请求头。
    """
    _REQUEST_IP.set(ip or None)


def request_ip() -> Optional[str]:
    """当前请求的客户端 IP（未设置返回 None）。"""
    return _REQUEST_IP.get()


def _global_semaphore() -> Optional[threading.BoundedSemaphore]:
    """惰性创建全局在途信号量；`GLOBAL_MAX_INFLIGHT <= 0` 时返回 None（关闭）。"""
    global _global_sem
    if GLOBAL_MAX_INFLIGHT <= 0:
        return None
    with _inflight_lock:
        if _global_sem is None:
            _global_sem = threading.BoundedSemaphore(max(1, int(GLOBAL_MAX_INFLIGHT)))
        return _global_sem


def reset_inflight_for_tests() -> None:
    """清空在途闸门状态（仅测试用；改了上面两个常量后必须调一次）。"""
    global _global_sem
    with _inflight_lock:
        _global_sem = None
        _ip_inflight.clear()


@contextmanager
def inflight_guard():
    """全局在途上限 +（已知客户端 IP 时）同一 IP 的在途上限。

    拿不到槽位就抛 `ServerBusy`（最多等 `INFLIGHT_WAIT_SECONDS` 秒）。
    """
    sem = _global_semaphore()
    if sem is not None and not sem.acquire(timeout=INFLIGHT_WAIT_SECONDS):
        raise ServerBusy()
    try:
        ip = request_ip()
        per_ip = max(0, int(PER_IP_MAX_INFLIGHT)) if ip else 0
        if per_ip:
            with _inflight_lock:
                if _ip_inflight.get(ip, 0) >= per_ip:
                    raise ServerBusy()
                _ip_inflight[ip] = _ip_inflight.get(ip, 0) + 1
        try:
            yield
        finally:
            if per_ip:
                with _inflight_lock:
                    left = _ip_inflight.get(ip, 0) - 1
                    if left > 0:
                        _ip_inflight[ip] = left
                    else:
                        _ip_inflight.pop(ip, None)
    finally:
        if sem is not None:
            sem.release()


@contextmanager
def llm_slot():
    """出站 LLM/视觉调用的槽位 = 在途闸门 + 该凭证的分桶上限。

    ★ 供 `core/llm_client.py` 与 `utils/ocr_utils.py` 使用。
    两道闸门的获取顺序**固定**（先全局/按 IP，再按凭证），全仓库一致 → 不会死锁。
    """
    with inflight_guard():
        with llm_semaphore():
            yield


if __name__ == "__main__":
    print("无覆盖 →", redact(llm_config().api_key), llm_config().model_name,
          "思考:", thinking_setting(), effort_setting(),
          "max_tokens:", max_tokens_setting())
    with user_credentials({"api_key": "sk-abcdefghijklmn", "model": "deepseek-flash",
                           "thinking": "off", "effort": "high",
                           "max_tokens": "16384"}):
        cfg = llm_config()
        print("有覆盖 →", redact(cfg.api_key), cfg.model_name,
              "思考:", thinking_setting(), effort_setting(),
              "max_tokens:", max_tokens_setting())
        print("视觉（未填 → 用主模型）→", vision_config().model_name,
              "is_main:", vision_is_main())
        print("指纹 →", fingerprint())
    with user_credentials({"api_key": "sk-1", "model": "deepseek-flash",
                           "vision_model": "glm-5.3-flash"}):
        v = vision_config()
        print("视觉（独立模型）→", v.model_name, "| thinking:", v.thinking or "(空=复用)")
    print("退出覆盖 →", redact(llm_config().api_key))
    print("base_url 校验 →", client_host_allowed("https://api.deepseek.com/v1"))
    print("base_url 拦截 →", client_host_allowed("http://127.0.0.1:8000/v1"))
