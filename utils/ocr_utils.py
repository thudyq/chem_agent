# -*- coding: utf-8 -*-
"""utils/ocr_utils.py — 图片 → SMILES（视觉大模型识别化学结构）。

MVP 方案：复用 OpenAI 兼容 API 的 vision 能力（content 含 image_url），
让视觉模型识别结构式并输出 SMILES。无需 DECIMER 等重 ML 依赖。
用户需配置支持视觉的模型（如 GLM-4V / Qwen2-VL / GPT-4o）。
"""

import base64
import os
import re

import requests

from core.llm_client import _load_env, _get_config


def _get_vision_config():
    """读取视觉模型独立配置（VISION_* 系列优先，回退到主配置）。"""
    _load_env()
    api_key = os.environ.get("VISION_API_KEY", "").strip() or os.environ.get("API_KEY", "").strip()
    base_url = (os.environ.get("VISION_BASE_URL", "").strip() or os.environ.get("BASE_URL", "").strip()).rstrip("/")
    model = os.environ.get("VISION_MODEL", "").strip() or os.environ.get("MODEL_NAME", "").strip() or os.environ.get("MODEL", "").strip()
    if not (api_key and base_url and model):
        return None
    return api_key, base_url, model


def image_to_smiles(image_path: str) -> str:
    """上传结构式图片 → 视觉 LLM 识别 → 返回 SMILES 字符串。

    需配置 VISION_MODEL + VISION_BASE_URL + VISION_API_KEY（或回退到主配置）。
    失败返回 None。
    """
    config = _get_vision_config()
    if config is None:
        print("[ocr] 未配置 VISION_MODEL/VISION_BASE_URL/VISION_API_KEY")
        return None
    api_key, base_url, model = config

    # 读图 + base64
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"[ocr] 读图失败: {e}")
        return None

    ext = str(image_path).rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    data_url = f"data:{mime};base64,{b64}"

    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "识别这个化学结构式图片，只输出对应的 SMILES 字符串，不要任何解释或其他文字。"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "temperature": 0.1,
        "max_tokens": 200,
    }

    print(f"[ocr] 调用视觉模型 {model} 识别图片 ...")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
    except requests.exceptions.RequestException as e:
        print(f"[ocr] 请求异常: {e}")
        return None

    if resp.status_code != 200:
        print(f"[ocr] HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code in (400, 404):
            print("[ocr] 该模型可能不支持视觉输入，请在 .env 中设置 VISION_MODEL 为支持图片的模型（如 GLM-4V）。")
        return None

    try:
        msg = resp.json()["choices"][0]["message"]
    except (KeyError, ValueError):
        return None
    content = msg.get("content") or ""
    if not content and msg.get("reasoning_content"):
        content = msg["reasoning_content"]

    return _extract_smiles(content)


def _looks_like_smiles(s: str) -> bool:
    if not s or len(s) > 200:
        return False
    if re.search(r"[=#\[\]()@/\\]", s):
        return True
    # 全 SMILES 原子字符 + 含数字（环号）
    if re.fullmatch(r"[cnospfbiCNOSPFBIIH\d.+-]+", s) and re.search(r"\d", s):
        return True
    return False


def _extract_smiles(text: str) -> str:
    """从 LLM 输出中提取 SMILES（可能含多余文字/引号）。"""
    if not text:
        return None
    text = text.strip().strip("\"'` \n")
    if _looks_like_smiles(text):
        return text
    for token in re.split(r"[\s,;]+", text):
        token = token.strip("\"'.,:;")
        if _looks_like_smiles(token):
            return token
    return text or None
