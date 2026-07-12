# -*- coding: utf-8 -*-
"""core/llm_client.py — LLM 客户端（OpenAI 兼容）。

调用 DeepSeek / SiliconFlow / 智谱等兼容 OpenAI 接口的服务。
从环境变量读取 API_KEY / BASE_URL / MODEL_NAME（兼容旧 MODEL）。
system_prompt 未传时自动加载 prompts/system_prompt.txt。
"""

import os
import time
from pathlib import Path

import requests

from core.prompt_manager import load_system_prompt

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 3


def _load_env():
    """读取 .env 到 os.environ（python-dotenv 优先，缺失时手动解析兜底）。"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    env_path = Path(".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip().strip("'").strip('"')
        os.environ.setdefault(key.strip(), val)


def _get_config():
    """读取并校验 LLM 配置。返回 (api_key, base_url, model) 或 None。"""
    _load_env()
    api_key = os.environ.get("API_KEY", "").strip()
    base_url = os.environ.get("BASE_URL", "").strip().rstrip("/")
    # MODEL_NAME（TRANSITION 规范），兼容旧 .env 的 MODEL
    model = os.environ.get("MODEL_NAME", "").strip() or os.environ.get("MODEL", "").strip()
    if not (api_key and base_url and model):
        return None
    return api_key, base_url, model


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
    config = _get_config()
    if config is None:
        print("[ask_llm] 未配置 API_KEY/BASE_URL/MODEL_NAME，请创建 .env（参考 .env.example）。")
        return None
    api_key, base_url, model = config

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
                msg = data["choices"][0]["message"]
                # reasoning 模型（如 deepseek-reasoner）可能把输出放 reasoning_content，content 为空
                content = msg.get("content") or ""
                if not content and msg.get("reasoning_content"):
                    content = msg["reasoning_content"]
                    print("[ask_llm] content 为空，回退 reasoning_content")
                usage = data.get("usage", {})
                print(f"[ask_llm] 成功（tokens: {usage.get('total_tokens', '?')}）")
                return content
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
