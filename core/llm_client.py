# -*- coding: utf-8 -*-
"""core/llm_client.py — LLM 客户端（OpenAI 兼容）。

调用 DeepSeek / SiliconFlow / 智谱等兼容 OpenAI 接口的服务。
配置统一从 core.config.settings 读取（兼容旧 MODEL 环境变量）。
system_prompt 未传时自动加载 prompts/system_prompt.txt。
"""

import json
import re
import threading
import time

import requests

from .config import settings
from .prompt_manager import load_system_prompt

DEFAULT_TEMPERATURE = settings.llm.temperature
DEFAULT_MAX_TOKENS = settings.llm.max_tokens
DEFAULT_TIMEOUT = settings.llm.timeout
DEFAULT_RETRIES = settings.llm.retries

# B3 并发控制：Session 复用 TCP 连接（keep-alive），信号量限制同时调用数
_SESSION = requests.Session()
_SEMAPHORE = threading.BoundedSemaphore(settings.llm.max_concurrent)

# 内容完整性校验：行尾残留的半截渲染标记（如 "[STRUCT:c1ccc" 无闭合 ]）
_TRUNC_MARK_RE = re.compile(
    r"\[(?:STRUCT|ARROW|REACTION|COMPOSITE|NEWMAN|ENERGY|LEWIS|"
    r"STEREO|CHARGE|HBOND|RETRO|XH|BOND):[^\]]*$",
    re.MULTILINE,
)
_ENV_BEGIN_RE = re.compile(r"\\begin\{(\w+)\}")
_ENV_END_RE = re.compile(r"\\end\{(\w+)\}")


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


def _stream_chat(url: str, headers: dict, payload: dict,
                 on_piece=None) -> tuple[str | None, str, int]:
    """SSE 流式调用：逐帧累积 content，返回 (content, finish_reason, reasoning_chars)。

    流式下 requests 的 timeout 只作用于"两块数据之间的间隔"
    （默认 180s 已足够），**总生成时间不受整体超时限制**——解决
    8000+ tokens 长输出整体超时的问题（实测曾 3 次重试全超时）。

    注意：带思考的模型（deepseek 系）在思考阶段只输出
    `delta.reasoning_content` 帧、`content` 为空；若思考吃掉全部预算
    或模型思考后未生成正式回答，content 永远为空——此时返回
    reasoning_chars（思考字符数）供上层诊断，绝不把思考当回答。

    on_piece: 可选回调，每收到一段 content 增量立即调用（B2 流式转发用）；
        重试/回退时各段增量都会回调，由调用方按需限频合并。

    返回:
        content: 累积的正式回答；无任何 content 时为 None。
        finish_reason: 最后一帧的 finish_reason（length 表示被截断）。
        reasoning_chars: 收到的思考内容总字符数（诊断用）。
    """
    content_parts = []
    reasoning_chars = 0
    finish_reason = ""
    with _SESSION.post(url, headers=headers, json=payload,
                       stream=True, timeout=DEFAULT_TIMEOUT) as resp:
        if resp.status_code != 200:
            print(f"[ask_llm] HTTP {resp.status_code}: {resp.text[:200]}")
            return None, "", 0
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
                    break
            fr = choices[0].get("finish_reason")
            if fr:
                finish_reason = fr
    content = "".join(content_parts).strip()
    return (content or None), finish_reason, reasoning_chars


def _thinking_stages(config, override: str = None) -> list:
    """主模型思考参数的渐进降级链，返回 [(mode, effort), ...]。

    mode 为 None 表示不传 thinking 参数（用 API 默认，DeepSeek 默认为开启）。
    「思考过长只输出 reasoning 而无正式回答」时逐级尝试下一级：配置的 effort
    较高或未配时先降到 low，最后一级关思考（disabled）；THINKING_MODE=disabled
    时无降级空间，仅一级。
    override：调用方覆盖思考模式（"enabled"/"disabled"），修正/标题等
    机械性调用传 "disabled"——思考链对此类任务收益小、延迟高。
    """
    mode = (override if override in ("enabled", "disabled")
            else config.thinking_mode)
    if mode == "disabled":
        return [("disabled", None)]
    mode0 = "enabled" if mode == "enabled" else None
    effort0 = (config.reasoning_effort
               if config.reasoning_effort in ("low", "high", "max") else None)
    stages = [(mode0, effort0)]
    if effort0 != "low":
        stages.append(("enabled", "low"))
    stages.append(("disabled", None))
    return stages


def ask_llm(
    user_question: str,
    system_prompt: str = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = DEFAULT_RETRIES,
    history: list = None,
    on_piece=None,
    thinking: str = None,
    model: str = None,
) -> str:
    """调用 LLM（SSE 流式），返回回答文本。

    参数:
        user_question: 用户问题。
        system_prompt: 系统提示；None 时自动加载 prompts/system_prompt.txt。
        temperature: 采样温度，默认 0.2（事实性强）。
        max_tokens: 最大生成 token 数，默认 8192。
        retries: 失败重试次数，默认 3。
        history: 多轮对话历史 [{"role": "user"/"assistant", "content": str}, ...]，
            插在 system_prompt 与当前问题之间；None 表示单轮。
        on_piece: 可选回调，每收到一段生成内容立即调用（B2 流式转发）。
        thinking: 可选覆盖思考模式（"enabled"/"disabled"），None 用配置；
            修正/标题等机械性调用传 "disabled"（思考阶段无内容帧，收益小、
            延迟高，前端看似卡死）。
        model: 可选覆盖模型名（如路由升级时传 "deepseek-v4-pro"）；None 用
            配置 MODEL_NAME。

    返回:
        回答文本；配置缺失或重试耗尽返回 None。
    """
    config = settings.llm
    if not config.is_configured:
        print("[ask_llm] 未配置 API_KEY/BASE_URL/MODEL_NAME，请创建 .env（参考 .env.example）。")
        return None
    api_key, base_url = config.api_key, config.base_url
    model = (model or config.model_name).strip()

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
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    print(f"[ask_llm] 调用 {model} @ {base_url}（temperature={temperature}, 流式）")
    # 渐进降级 + 回退链：主模型按配置逐级降思考强度（思考过长只输出 reasoning
    # 而无正式回答时立即进入下一级，不在本级浪费重试）；回退模型作为最后一级
    # 强制 thinking=disabled，保证给出正式回答。非思考类失败（超时/HTTP/截断）
    # 按 retries 重试本级后再进入下一级。
    stages = [(model, ts) for ts in _thinking_stages(config, override=thinking)]
    fallback = config.fallback_model_name.strip()
    if fallback and fallback != model:
        stages.append((fallback, ("disabled", None)))
        print(f"[ask_llm] 回退模型已配置: {fallback}（回退调用强制关闭思考）")

    with _SEMAPHORE:  # B3：限流——同时最多 max_concurrent 个 LLM 调用
        for stage_i, (current_model, (tmode, teffort)) in enumerate(stages):
            payload["model"] = current_model
            if tmode is None:
                payload.pop("thinking", None)
            else:
                payload["thinking"] = {"type": tmode}
            if teffort and tmode != "disabled":
                payload["reasoning_effort"] = teffort
            else:
                payload.pop("reasoning_effort", None)
            if stage_i == 0:
                print(f"[ask_llm] 思考参数: thinking={tmode or 'API 默认'}, "
                      f"effort={teffort or 'API 默认'}")
            advanced = False
            for attempt in range(1, retries + 1):
                try:
                    content, finish_reason, reasoning_chars = _stream_chat(
                        url, headers, payload, on_piece=on_piece)
                except requests.exceptions.Timeout as e:
                    print(f"[ask_llm] 第 {attempt}/{retries} 次请求超时"
                          f"（数据间隔 >{DEFAULT_TIMEOUT}s，模型思考过久或网络慢）: {e}")
                except requests.exceptions.RequestException as e:
                    print(f"[ask_llm] 第 {attempt}/{retries} 次请求异常: {e}")
                else:
                    if content is None:
                        if reasoning_chars and stage_i < len(stages) - 1:
                            nmodel, (nmode, neffort) = stages[stage_i + 1]
                            if nmodel != current_model:
                                nxt = f"切换回退模型 {nmodel}（强制关闭思考）"
                            else:
                                nxt = (f"降级思考参数（thinking={nmode or '默认'}, "
                                       f"effort={neffort or '默认'}）")
                            print(f"[ask_llm] 模型 {current_model} 思考过长"
                                  f"（{reasoning_chars} 字符）且无正式回答，{nxt} ...")
                            advanced = True
                            break
                        if reasoning_chars:
                            print(f"[ask_llm] 第 {attempt}/{retries} 次失败"
                                  f"（无正式回答，模型仅输出思考 {reasoning_chars} 字符"
                                  f"——思考过长吃光预算，或模型未生成 content）")
                        else:
                            print(f"[ask_llm] 第 {attempt}/{retries} 次失败（无内容返回，"
                                  f"finish_reason={finish_reason or '无'}）")
                    elif finish_reason == "length":
                        # 思考退化：reasoning 字符数远超实际回答（≥2×）时，
                        # 说明思考循环吃光预算（用户观察：1 万字重复思考后
                        # 截断）——重试只会重复同一循环，直接降级下一 stage
                        # （同 content is None 的降级逻辑）。
                        if (reasoning_chars and content
                                and len(content) * 2 < reasoning_chars
                                and stage_i < len(stages) - 1):
                            nmodel, (nmode, neffort) = stages[stage_i + 1]
                            if nmodel != current_model:
                                nxt = f"切换回退模型 {nmodel}（强制关闭思考）"
                            else:
                                nxt = (f"降级思考参数（thinking={nmode or '默认'}, "
                                       f"effort={neffort or '默认'}）")
                            print(f"[ask_llm] 模型 {current_model} 思考循环吃光预算"
                                  f"（reasoning {reasoning_chars} 字符 >> 回答 "
                                  f"{len(content)} 字符，finish_reason=length），{nxt} ...")
                            advanced = True
                            break
                        print(f"[ask_llm] 第 {attempt}/{retries} 次失败"
                              "（输出被 max_tokens 截断，finish_reason=length）")
                    else:
                        trunc = _is_truncated(content)
                        if trunc:
                            print(f"[ask_llm] 第 {attempt}/{retries} 次失败"
                                  f"（内容完整性校验失败：{trunc}）")
                        else:
                            print(f"[ask_llm] 成功（{len(content)} 字符，{current_model}）")
                            return content

                if attempt < retries:
                    backoff = attempt * 2
                    print(f"[ask_llm] {backoff}s 后重试 ...")
                    time.sleep(backoff)
            if advanced:
                continue

    print(f"[ask_llm] {' → '.join(dict.fromkeys(m for m, _ in stages))} 重试均失败。")
    return None


if __name__ == "__main__":
    # Day 6-7 测试入口
    response = ask_llm("苯的结构式是什么？请用 [STRUCT] 标记输出。")
    if response:
        print("\n[LLM 回答]")
        print(response)
    else:
        print("\n[结果] 调用失败，请检查 .env 配置。")
