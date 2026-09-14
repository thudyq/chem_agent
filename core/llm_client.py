# -*- coding: utf-8 -*-
"""core/llm_client.py — LLM 客户端（OpenAI 兼容）。

配置统一从 `core.credentials` 读取（每请求凭证；无覆盖时用 `.env`）。
system_prompt 未传时自动加载 prompts/system_prompt.txt。

思考参数模型（`instructions/Model-Config-Refactor.md` §4.4）
-----------------------------------------------------------
用户意图是**两个正交维度**：**开关**（on/off）× **强度**（low/medium/high/max）。
映射为请求字段：

    关        → thinking={"type":"disabled"}，不发 reasoning_effort，**发** temperature
    开 + 强度 → thinking={"type":"enabled"} + reasoning_effort=<强度>，**不发** temperature

`temperature` 只在"关思考"时发送：思考模式下该参数被上游静默忽略（官方 F6）。

端点不配合时（§4.5.3）
--------------------
**判定只看行为、不解析错误文本**——GLM 的拒绝是中文（"该模型始终思考，不支持关闭
思考；请使用 low、high 或 max。"），任何按字段名匹配的规则都会漏判。
规则：400 → **逐个摘掉候选字段重试** → 摘掉后成功即认定该字段是原因（记入能力表）；
仍失败即认定与档位无关的真实错误，原样报错（不掩盖）。

降级链（§4.5）
-------------
`首选档位 → 逐级往低（max→high→medium→low）→ 关思考`（只降不升，同一模型内；
**不再换模型**）。详见 `_effort_stages`。
`finish_reason=length` **一律降档**（同预算重试结果必然相同，纯浪费）。
"""

import json
import re
import time

import requests

from . import capabilities, credentials
from .config import (EFFORT_LOW, EFFORT_ORDER, THINKING_ON, normalize_effort,
                     normalize_thinking)
from .prompt_manager import load_system_prompt

# 建连与读取分开（安全审查 R5）：
# * 建连超时要短 —— 不可达/被丢包的地址不再让每个请求干等满整个超时；
# * 读取超时保持宽松 —— 思考模型首字可能很慢，且流式下它只约束
#   "两块数据之间的间隔"，不是总时长。
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 180
DEFAULT_TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)
DEFAULT_RETRIES = 3

# 各调用点的 max_tokens（§4.2.1）。**上限不是预留**，按实际用量计费；
# 设小会导致截断（思考与回答共享该额度），设大不会多扣费。
MAX_TOKENS_MAIN = 32768      # 主生成：思考 5k~15k + 回答 1k~3k，留足余量
MAX_TOKENS_REWRITE = 4096    # 箭头/结构重写：要吐一个 COMPOSITE 块
MAX_TOKENS_FIX = 2048        # 常规修正：只输出修正后的标记
MAX_TOKENS_TINY = 64         # 翻译 / 会话标题
MAX_TOKENS_VISION = 8192     # 视觉识别：复杂机理图描述长 + 思考可能占一部分

# 内容完整性校验：行尾残留的半截渲染标记（如 "[STRUCT:c1ccc" 无闭合 ]）
_TRUNC_MARK_RE = re.compile(
    r"\[(?:STRUCT|ARROW|REACTION|COMPOSITE|NEWMAN|ENERGY|LEWIS|"
    r"STEREO|CHARGE|HBOND|RETRO|XH|BOND|CHAIR):[^\]]*$",
    re.MULTILINE,
)
_ENV_BEGIN_RE = re.compile(r"\\begin\{(\w+)\}")
_ENV_END_RE = re.compile(r"\\end\{(\w+)\}")

# 摘字段重试的顺序：先摘"最可能引起 400"的（思考相关），最后才是 max_tokens。
_STRIP_ORDER = ("thinking", "reasoning_effort", "temperature", "max_tokens")


class LLMResult:
    """一次 LLM 调用的结果（含"实际执行情况"，供诚实上报）。

    text: 回答文本；None 表示失败。
    error: 失败原因（面向调用方，可含上游原始信息）。
    effort_effective: 实际生效的档位（"off"/"low"/…）；端点不配合时为 None。
    effort_requested: 调用方请求的档位。
    downgraded: 是否发生了静默改写（用于主生成的"诚实上报"，§4.5.5）。
    notice: 面向用户的提示（仅在 downgraded 时非空）。
    """

    __slots__ = ("text", "error", "effort_requested", "effort_effective",
                 "downgraded", "notice", "model", "endpoint_rejected")

    def __init__(self, text=None, error=None, effort_requested="", 
                 effort_effective=None, downgraded=False, notice="",
                 model="", endpoint_rejected=None):
        self.text = text
        self.error = error
        self.effort_requested = effort_requested
        self.effort_effective = effort_effective
        self.downgraded = downgraded
        self.notice = notice
        self.model = model
        self.endpoint_rejected = endpoint_rejected or []

    def __bool__(self) -> bool:
        return bool(self.text)

    def __str__(self) -> str:      # 兼容把返回值当字符串用的旧调用方
        return self.text or ""


def _is_truncated(content: str) -> str | None:
    """检测输出是否被截断；截断返回原因，完整返回 None。

    两个信号（不依赖 API 的 finish_reason，防止兼容接口谎报 stop）：
    1. LaTeX 环境未配对：\\begin{tikzpicture} 多于 \\end{tikzpicture}；
    2. 渲染标记未闭合：行尾残留半截标记，或 [COMPOSITE: 多于 [/COMPOSITE]。
    """
    begins, ends = {}, {}
    for m in _ENV_BEGIN_RE.finditer(content):
        begins[m.group(1)] = begins.get(m.group(1), 0) + 1
    for m in _ENV_END_RE.finditer(content):
        ends[m.group(1)] = ends.get(m.group(1), 0) + 1
    for name, count in begins.items():
        if count > ends.get(name, 0):
            return f"LaTeX 环境 \\begin{{{name}}} 未闭合"
    if content.count("[COMPOSITE:") > content.count("[/COMPOSITE]"):
        return "COMPOSITE 标记未闭合"
    if _TRUNC_MARK_RE.search(content):
        return "含未闭合的渲染标记"
    return None


def _normalize_effort_request(thinking, effort, config) -> tuple:
    """把调用方意图归一为 (thinking_on: bool, effort: str)。

    thinking: None → 用配置的 thinking_default；"disabled"/"off" → 关；
              "enabled"/"on" → 开（此时 effort 用配置的 effort_default）
    effort:   None → 用配置的 effort_default
    """
    if thinking is None:
        on = normalize_thinking(getattr(config, "thinking_default", THINKING_ON)) == THINKING_ON
    else:
        on = normalize_thinking(thinking, default=THINKING_ON) == THINKING_ON
    eff = normalize_effort(effort if effort else
                           getattr(config, "effort_default", EFFORT_LOW))
    return on, eff


def _effort_stages(thinking_on: bool, effort) -> list:
    """降级链：`首选 → 逐级往低 → 关思考`（严格单调下降、只降不升、去重）。

    返回 [(thinking_on, effort_or_None), ...]。
    从"用户所选档位"开始，沿 low < medium < high < max 逐级下降
    （max → high → medium → low → 关），再到关思考。
    关思考起步时只有一级（没有更低的了）。
    """
    if not thinking_on:
        return [(False, None)]
    order = list(EFFORT_ORDER)
    if effort in order:
        idx = order.index(effort)
    else:
        idx = 0                       # 未指定强度：按最低档起步
    stages = [(True, order[i]) for i in range(idx, -1, -1)]
    stages.append((False, None))
    seen, out = set(), []
    for st in stages:
        if st not in seen:
            seen.add(st)
            out.append(st)
    return out


def _build_payload(model: str, messages: list, on: bool, effort, max_tokens,
                   temperature: float, cap, dropped: set) -> dict:
    """按 (开关, 强度) 组装 payload；遵守"已确认被拒的字段"（dropped）。"""
    payload = {"model": model, "messages": messages, "stream": True}
    if "max_tokens" not in dropped and max_tokens:
        payload["max_tokens"] = int(max_tokens)

    if not on:
        # 关思考：发显式 disabled（这是能探测"端点是否强制思考"的前提），
        # 不发 effort，发 temperature（思考模式下 temperature 无效，F6）
        if "thinking" not in dropped:
            payload["thinking"] = {"type": "disabled"}
        if "temperature" not in dropped:
            payload["temperature"] = temperature
        return payload

    # 开思考：发 enabled + effort；不发 temperature（思考模式下它被忽略）
    if "thinking" not in dropped:
        payload["thinking"] = {"type": "enabled"}
    if effort and "reasoning_effort" not in dropped:
        payload["reasoning_effort"] = effort
    return payload


class _HttpFailure(Exception):
    """上游返回非 200（携带状态码与响应体，供行为判定使用）。"""

    def __init__(self, status: int, body: str):
        # ★ 3xx：安全审查 R3 —— 出站请求**不跟随重定向**（否则一个公网域名可以
        # 返回 302 跳到内网/云元数据，绕过 `client_host_allowed` 的校验）。
        # 这里把"该怎么办"写进异常文本，让 CLI 日志与网页提示都能直接用。
        hint = ""
        if 300 <= status < 400:
            hint = ("（端点返回重定向；为安全起见本服务不跟随跳转，"
                    "请把**最终**的完整地址直接填进「接口地址」）")
        super().__init__(f"HTTP {status}: {body[:200]}{hint}")
        self.status = status
        self.body = body or ""


def _stream_chat(url: str, headers: dict, payload: dict, on_piece=None) -> tuple:
    """SSE 流式调用：逐帧累积 content。

    返回 (content, finish_reason, reasoning_chars, observed_reasoning)。
    非 200 抛 `_HttpFailure`（调用方据此做"摘字段重试"的行为判定）。

    on_piece: 可选回调，每收到一段 content 增量立即调用（流式转发用）。
    """
    content_parts = []
    reasoning_chars = 0
    observed_reasoning = False
    finish_reason = ""
    # ★ `allow_redirects=False`（安全审查 R3）：端点地址已过 SSRF 校验，但
    # 302 之后跳到哪不受那个校验管 —— 跟随重定向等于把校验作废。
    with credentials.session().post(
            url, headers=headers, json=payload, stream=True,
            allow_redirects=False,
            timeout=DEFAULT_TIMEOUT) as resp:
        if resp.status_code != 200:
            body = (resp.text or "")[:500]
            print(f"[ask_llm] HTTP {resp.status_code}: {body[:200]}")
            raise _HttpFailure(resp.status_code, body)
        # SSE 按 UTF-8 解码：响应头缺 charset 时 requests 默认 ISO-8859-1，
        # 中文会被按 latin-1 误读（双重编码乱码，gemini 端点实测出现）
        resp.encoding = "utf-8"
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except ValueError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                content_parts.append(piece)
                if on_piece is not None:
                    on_piece(piece)
            for key in ("reasoning_content", "reasoning"):
                rp = delta.get(key) or ""
                if rp:
                    reasoning_chars += len(rp)
                    observed_reasoning = True
                    break
            fr = choices[0].get("finish_reason")
            if fr:
                finish_reason = fr
    content = "".join(content_parts).strip()
    return (content or None), finish_reason, reasoning_chars, observed_reasoning


def _capability_key(base_url: str, model: str, api_key: str) -> str:
    return capabilities.make_key(base_url, model, api_key)


def _downgrade_notice(requested_on: bool, eff_on: bool, eff_effort) -> str:
    """生成"用户意图被静默改写"的用户可见提示（§4.5.5）。

    只在**主生成**路径使用，且同会话只提示一次（调用方负责去重）。
    文案原则：**描述我们观测到什么，不替厂商下结论**——反例是旧实现里的
    "模型 gemini-3.7-flash 不支持 thinking 参数"（Gemini 支持思考，
    只是不认这套字段语法，这句话会误导用户）。
    """
    from .config import EFFORT_LABELS
    if requested_on and not eff_on:
        return "（该端点未按「开启思考」执行，本次按关闭思考运行）"
    label = EFFORT_LABELS.get(eff_effort or "", eff_effort or "")
    return f"（该模型始终思考，无法关闭；本次按「{label}」执行）"


def _plan_with_capability(cap, on: bool, effort, stages: list) -> tuple:
    """按能力表调整**首选**档位，并裁剪降级链。返回 (on, effort, 改写了吗, stages)。

    已知"不能关"而用户要关 → 换成该端点允许范围内最省的一档（优先 low）；
    已知"不支持思考"而用户要开 → 改为关；
    强度取值被拒过 → 就近换档（Gemini 拒 medium → low）。
    """
    if not on:
        if cap.allows_off() is False:
            alt = cap.best_effort_at_most(EFFORT_LOW) or EFFORT_LOW
            print(f"[ask_llm] 能力表：该端点不支持关闭思考，改用最低档 {alt}")
            return True, alt, True, _effort_stages(True, alt)
        return False, None, False, stages
    if cap.allows_on() is False:
        print("[ask_llm] 能力表：该端点不支持思考，改为关闭")
        return False, None, True, _effort_stages(False, None)
    if effort and not cap.supports_effort(effort):
        alt = cap.best_effort_at_most(effort)
        if alt is None:
            print(f"[ask_llm] 能力表：该端点不接受任何思考强度，忽略强度 {effort}")
            return True, None, True, _effort_stages(True, None)
        print(f"[ask_llm] 能力表：强度 {effort} 不被接受，就近改用 {alt}")
        return True, alt, True, _effort_stages(True, alt)
    return True, effort, False, stages


def ask_llm(
    user_question: str,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    retries: int = None,
    history: list = None,
    on_piece=None,
    thinking: str = None,
    model: str = None,
    effort: str = None,
    return_result: bool = False,
    user_scoped: bool = False,
):
    """调用 LLM（SSE 流式）。

    参数:
        user_question: 用户问题。
        system_prompt: 系统提示；None 时自动加载 prompts/system_prompt.txt。
        temperature: 采样温度；None 用配置默认（0.2）。**仅"关思考"时发送**。
        max_tokens: 最大生成 token 数；None 用配置的 `max_tokens`。
            各调用点建议用本模块的 MAX_TOKENS_* 常量（§4.2.1）。
        retries: 失败重试次数；None 用配置默认（3）。
        history: 多轮对话历史，插在 system 与当前问题之间；None 表示单轮。
        on_piece: 可选回调，每收到一段生成内容立即调用（流式转发）。
        thinking: 思考开关覆盖："enabled"/"on"/"disabled"/"off"；None 用配置。
        model: 模型名覆盖；None 走**统一凭证解析**
            （用户请求头指定 > 服务器 `.env`），避免漏传时静默用服务器模型。
        effort: 思考强度覆盖 low/medium/high/max；None 用配置。
        return_result: True 时返回 `LLMResult`（含实际档位等元信息），
            False 时返回文本（None 表示失败）——保持旧调用方兼容。
        user_scoped: True 表示这是**用户请求作用域内**的调用（BYOK 下由用户
            自己的 key 付费）。此时若在**有请求凭证**但没有模型名，**不再回退
            服务器 `.env` 的模型**，而是直接判失败——杜绝"漏传 model 导致
            静默用服务器模型"这一整类缺陷（实测：网页填 deepseek-flash，
            翻译却打了 `.env` 的 gemini-3.7-flash）。
            无请求凭证（清小搭 / Streamlit / CLI）时该参数无影响。

    返回:
        return_result=False: 回答文本 / None。
        return_result=True:  LLMResult。
    """
    config = credentials.llm_config()
    if not config.is_configured:
        print("[ask_llm] 未配置 API_KEY/BASE_URL/MODEL_NAME，请创建 .env（参考 .env.example）。")
        res = LLMResult(error="未配置模型凭证")
        return res if return_result else None
    api_key, base_url = config.api_key, config.base_url
    # ★ 统一模型解析：显式传入 > 用户请求头指定 > 服务器配置
    user_model = credentials.user_model()
    model = (model or user_model or config.model_name).strip()
    if not model:
        print("[ask_llm] 未指定模型名（请求头与 .env 均为空）")
        res = LLMResult(error="未指定模型名")
        return res if return_result else None
    if user_scoped and credentials.current() and model == config.model_name \
            and not user_model:
        # 用户作用域内、却退化成了服务器模型的别名：宁可失败也不偷偷用服务器的
        print("[ask_llm] 用户作用域内无可用模型名，拒绝回退服务器模型")
        res = LLMResult(error="用户未指定模型名")
        return res if return_result else None
    # 模型来源标注：用于一眼看出"这次调用是谁指定的模型"，
    # 避免再出现"网页填了 A、日志却是 B"时无法定位。
    if model == (user_model or ""):
        _src = "用户请求头"
    elif model == config.model_name and credentials.current():
        _src = "服务器.env（用户未给模型）"
    else:
        _src = "服务器.env"
    print(f"[ask_llm] 模型来源={_src} endpoint={base_url} "
          f"credential={credentials.redact(api_key)}")

    resolved_temperature = config.temperature if temperature is None else temperature
    resolved_max_tokens = config.max_tokens if max_tokens is None else max_tokens
    resolved_retries = config.retries if retries is None else retries

    if system_prompt is None:
        system_prompt = load_system_prompt()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_question})

    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    req_on, req_effort = _normalize_effort_request(thinking, effort, config)
    cap_key = _capability_key(base_url, model, api_key)
    cap = capabilities.get(cap_key)
    eff_on, eff_effort, pre_downgraded, stages = _plan_with_capability(
        cap, req_on, req_effort, _effort_stages(req_on, req_effort))

    requested_label = "关" if not req_on else (req_effort or "默认")
    print(f"[ask_llm] 调用 {model} @ {base_url}（credential={credentials.redact(api_key)}, "
          f"思考={requested_label}, max_tokens={resolved_max_tokens}, 流式）")

    # 能力表已知结论 → 本次请求直接不带该字段（不再付一次 400 学费）
    dropped = set()
    if cap.allows_off() is False:
        dropped.add("thinking")
    if cap.max_tokens_ok is False:
        dropped.add("max_tokens")

    result = LLMResult(effort_requested=("off" if not req_on else req_effort),
                       model=model)
    effective = None            # 实际生效的 (on, effort)
    last_error = ""
    rejected = []               # 本次摘掉的字段（用于成功时记入能力表）
    forced_off = []             # 被摘掉的是"关闭思考"的显式请求 → 需要诚实上报

    def _attempt():
        """一次 HTTP 调用（含"摘字段重试"）。返回 ((content, finish, rchars,
        observed), payload)。

        行为判定（§4.5.3）：非 200 → 逐个摘掉候选字段；**摘掉后成功**即认定该字段
        是原因（记入能力表）；摘光仍失败 → 判定为与档位无关的真实错误。
        """
        while True:
            payload = _build_payload(
                model, messages, eff_on, eff_effort, resolved_max_tokens,
                resolved_temperature, cap, dropped)
            try:
                return _stream_chat(url, headers, payload, on_piece=on_piece), payload
            except _HttpFailure as e:
                if e.status not in (400, 422):
                    raise
                stripped = _strip_one_field(payload, dropped)
                if stripped is None:
                    raise
                field, value = stripped
                rejected.append((field, value))
                if field == "thinking" and value == "disabled":
                    # 用户明确要"关思考"却关不掉 → 端点强制思考，需如实上报
                    forced_off.append(True)
                print(f"[ask_llm] HTTP {e.status} → 摘掉 {field}={value!r} 重试"
                      f"（其余字段保留）")

    # 在途闸门（全局 + 按 IP）+ 按凭证分桶上限（安全审查 R5）
    with credentials.llm_slot():
        for stage_i, (st_on, st_effort) in enumerate(stages):
            eff_on, eff_effort = st_on, st_effort
            advanced = False
            for attempt in range(1, resolved_retries + 1):
                try:
                    (content, finish_reason, reasoning_chars, observed), payload = \
                        _attempt()
                except _HttpFailure as e:
                    last_error = f"HTTP {e.status}: {e.body[:160]}"
                    print(f"[ask_llm] 摘字段后仍失败（HTTP {e.status}）"
                          f"→ 判定为与思考参数无关的真实错误")
                    return _fail(result, last_error, return_result, effective,
                                 req_on, req_effort)
                except requests.exceptions.Timeout as e:
                    last_error = f"请求超时: {e}"
                    print(f"[ask_llm] 第 {attempt}/{resolved_retries} 次请求超时"
                          f"（数据间隔 >{READ_TIMEOUT}s）: {e}")
                    continue
                except requests.exceptions.RequestException as e:
                    last_error = f"请求异常: {e}"
                    print(f"[ask_llm] 第 {attempt}/{resolved_retries} 次请求异常: {e}")
                    continue

                # ---- 请求成功：先把"摘掉的字段"记入能力表，再判定是否达标 ----
                for field, value in rejected:
                    capabilities.note_field_rejected(cap_key, field, value)
                    result.endpoint_rejected.append(f"{field}={value}")
                rejected = []
                effective = (st_on, st_effort)
                capabilities.note_success(cap_key, observed_thinking=observed)
                if "thinking" in payload:
                    capabilities.note_accepted(cap_key, "thinking",
                                              (payload.get("thinking") or {}).get("type"))
                if "reasoning_effort" in payload:
                    capabilities.note_accepted(cap_key, "reasoning_effort",
                                              payload["reasoning_effort"])
                if "max_tokens" in payload:
                    capabilities.note_accepted(cap_key, "max_tokens")

                if content is None:
                    # 只想不答：思考吃光预算或模型未生成 content → 降档
                    if reasoning_chars:
                        print(f"[ask_llm] 第 {attempt}/{resolved_retries} 次失败"
                              f"（无正式回答，仅输出思考 {reasoning_chars} 字符）")
                    else:
                        print(f"[ask_llm] 第 {attempt}/{resolved_retries} 次失败"
                              f"（无内容返回，finish_reason={finish_reason or '无'}）")
                    if stage_i < len(stages) - 1:
                        advanced = True
                        break
                elif finish_reason == "length":
                    # ★ 一律降档：同一预算下重试结果必然相同，纯浪费一次调用。
                    print(f"[ask_llm] 输出被 max_tokens={resolved_max_tokens} 截断"
                          f"（finish_reason=length，思考 {reasoning_chars} 字符），降档重试")
                    if stage_i < len(stages) - 1:
                        advanced = True
                        break
                else:
                    trunc = _is_truncated(content)
                    if trunc:
                        print(f"[ask_llm] 第 {attempt}/{resolved_retries} 次失败"
                              f"（内容完整性校验失败：{trunc}）")
                    else:
                        print(f"[ask_llm] 成功（{len(content)} 字符，{model}）")
                        if forced_off and not result.endpoint_rejected:
                            # 端点不接受关闭 → 实际是"开着思考"跑的：
                            # 有效档位记为实际发生的那一档（用户在"关"档）
                            effective = (True, eff_effort or EFFORT_LOW)
                        return _ok(result, content, effective, req_on, req_effort,
                                   pre_downgraded or bool(forced_off), return_result)

                if attempt < resolved_retries:
                    backoff = attempt * 2
                    print(f"[ask_llm] {backoff}s 后重试 ...")
                    time.sleep(backoff)
            if advanced:
                continue

    return _fail(result, last_error or "重试均失败（无正式回答/被截断）",
                 return_result, effective, req_on, req_effort)


def _ok(result, text, effective, req_on, req_effort, pre_downgraded,
        return_result):
    """填充成功结果（含"实际档位"与静默改写提示）。"""
    eff_on, eff_effort = effective
    result.text = text
    result.effort_effective = ("off" if not eff_on else (eff_effort or ""))
    changed = (eff_on != req_on) or (req_on and eff_effort != req_effort)
    if changed or pre_downgraded:
        result.downgraded = True
        result.notice = _downgrade_notice(req_on, eff_on, eff_effort)
    return result if return_result else text


def _fail(result, error, return_result, effective=None, req_on=False,
          req_effort=None):
    """填充失败结果。"""
    result.error = error
    if effective:
        eff_on, eff_effort = effective
        result.effort_effective = ("off" if not eff_on else (eff_effort or ""))
        if eff_on != req_on or (req_on and eff_effort != req_effort):
            result.downgraded = True
            result.notice = _downgrade_notice(req_on, eff_on, eff_effort)
    print(f"[ask_llm] 失败：{error}")
    return result if return_result else None


def _strip_one_field(payload: dict, dropped: set) -> tuple | None:
    """从 payload 里摘掉**一个**候选字段（逐个摘，保留仍可用的档位）。

    返回 (field, value) 或 None。GLM 明确说"请使用 low、high 或 max"——
    它要 `reasoning_effort` 不要 `thinking`，所以必须先摘 `thinking` 而保留 effort
    （把思考字段全部摘掉会让用户的档位白丢）。
    """
    for field in _STRIP_ORDER:
        if field not in payload:
            continue
        value = None
        if field == "thinking":
            value = (payload.get("thinking") or {}).get("type")
        elif field == "reasoning_effort":
            value = payload.get("reasoning_effort")
        payload.pop(field, None)
        dropped.add(field)
        return field, value
    return None


if __name__ == "__main__":
    res = ask_llm("苯的结构式是什么？请用 [STRUCT] 标记输出。",
                  return_result=True)
    if res:
        print("\n[LLM 回答]", res.text[:200])
        print(f"[实际档位] 请求={res.effort_requested} 生效={res.effort_effective} "
              f"静默改写={res.downgraded} {res.notice}")
    else:
        print("\n[结果] 调用失败，请检查 .env 配置。", res.error)
