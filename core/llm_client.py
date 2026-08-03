# -*- coding: utf-8 -*-
"""core/llm_client.py — LLM 客户端（OpenAI 兼容）。

调用 DeepSeek / SiliconFlow / 智谱等兼容 OpenAI 接口的服务。
配置统一从 core.config.settings 读取（兼容旧 MODEL 环境变量）。
system_prompt 未传时自动加载 prompts/system_prompt.txt。
"""

import re
import time

import requests

from .config import settings
from .prompt_manager import load_system_prompt

DEFAULT_TEMPERATURE = settings.llm.temperature
DEFAULT_MAX_TOKENS = settings.llm.max_tokens
DEFAULT_TIMEOUT = settings.llm.timeout
DEFAULT_RETRIES = settings.llm.retries

# 内容完整性校验：行尾残留的半截渲染标记（如 "[STRUCT:c1ccc" 无闭合 ]）
_TRUNC_MARK_RE = re.compile(
    r"\[(?:STRUCT|ARROW|REACTION|REACTIONMECH|COMPOSITE|NEWMAN|ENERGY|LEWIS|"
    r"STEREO|MECH|CHARGE|RESONANCE|HBOND|RETRO):[^\]]*$",
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


def ask_llm(
    user_question: str,
    system_prompt: str = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = DEFAULT_RETRIES,
) -> str:
    """调用 LLM，返回回答文本。

    参数:
        user_question: 用户问题。
        system_prompt: 系统提示；None 时自动加载 prompts/system_prompt.txt。
        temperature: 采样温度，默认 0.2（事实性强）。
        max_tokens: 最大生成 token 数，默认 2048。
        retries: 失败重试次数，默认 3。

    返回:
        回答文本；配置缺失或重试耗尽返回 None。
    """
    config = settings.llm
    if not config.is_configured:
        print("[ask_llm] 未配置 API_KEY/BASE_URL/MODEL_NAME，请创建 .env（参考 .env.example）。")
        return None
    api_key, base_url, model = config.api_key, config.base_url, config.model_name

    if system_prompt is None:
        system_prompt = load_system_prompt()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_question})

    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    print(f"[ask_llm] 调用 {model} @ {base_url}（temperature={temperature}）")
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
        except requests.exceptions.RequestException as e:
            print(f"[ask_llm] 第 {attempt}/{retries} 次请求异常: {e}")
        else:
            if resp.status_code == 200:
                data = resp.json()
                choice = data["choices"][0]
                msg = choice.get("message", {})
                content = (msg.get("content") or "").strip()
                finish_reason = choice.get("finish_reason", "")
                # 输出完整性校验（三道防线）：
                # 1) content 为空——推理模型（deepseek-reasoner 等）思考过长吃光
                #    token 预算时 content 为空；绝不可回退 reasoning_content
                #    （被截断的思考过程混着草稿与半截标记，会把下游解析/渲染带崩）。
                # 2) finish_reason=length——输出被 max_tokens 截断。
                # 3) 内容完整性自检——部分兼容接口不返回/谎报 finish_reason，
                #    截断内容会被当作成功；检测未闭合 LaTeX 环境与半截标记兜底。
                # 三种情况都走统一重试逻辑。
                if not content:
                    print("[ask_llm] content 为空（推理模型思考过长或异常），视为失败")
                elif finish_reason == "length":
                    print("[ask_llm] 输出被 max_tokens 截断（finish_reason=length），视为失败")
                else:
                    trunc = _is_truncated(content)
                    if trunc:
                        print(f"[ask_llm] 内容完整性校验失败（{trunc}），视为失败")
                    else:
                        usage = data.get("usage", {})
                        print(f"[ask_llm] 成功（tokens: {usage.get('total_tokens', '?')}）")
                        return content
            else:
                print(f"[ask_llm] 第 {attempt}/{retries} 次失败 HTTP {resp.status_code}: {resp.text[:200]}")

        if attempt < retries:
            backoff = attempt * 2
            print(f"[ask_llm] {backoff}s 后重试 ...")
            time.sleep(backoff)

    print(f"[ask_llm] {retries} 次重试均失败。")
    return None


if __name__ == "__main__":
    # Day 6-7 测试入口
    response = ask_llm("苯的结构式是什么？请用 [STRUCT] 标记输出。")
    if response:
        print("\n[LLM 回答]")
        print(response)
    else:
        print("\n[结果] 调用失败，请检查 .env 配置。")
