# -*- coding: utf-8 -*-
"""utils/ocr_utils.py — 图片多模态理解（视觉大模型）。

方案 B 两段式：视觉模型（如 GLM-4.6V）负责"理解"图片——转录文字、
结构式给 SMILES、反应式列三要素、机理图逐步描述弯箭头去向；
产出结构化描述交给主模型按标记体系答题。无需 DECIMER 等重 ML 依赖。
用户需配置支持视觉的模型（VISION_MODEL 等，如 GLM-4.6V / Qwen2-VL / GPT-4o）。
"""

import base64
import re

import requests

from core.config import settings

_DESCRIBE_PROMPT = """请描述这张图片的内容，供后续化学问答使用。按内容类型分别处理：
1. 文字（题目/问题/解答等）：完整转录原文，不要改写。
2. 化学结构式：给出 SMILES（能确定时）；不能确定时用文字描述（如"苯环连一个硝基"）。
3. 反应方程式：列出反应物、产物、反应条件，分子尽量用 SMILES。
4. 反应机理图（弯箭头/电子转移）：分步文字描述——每步哪个原子或键的电子流向哪里。
5. 其他图（势能面/构象/能级等）：说明图的类型与关键数值或标注。

输出格式（严格两行）：
类型：文字题 | 结构式 | 反应式 | 机理图 | 混合 | 其他
内容：<按上述要求的转录与描述>"""


def _parse_description(text: str) -> dict:
    """解析"类型：/内容："两行格式；不合格式时整体作为 content（type=未分类）。"""
    t = (text or "").strip()
    m = re.search(r"类型[:：]\s*(\S+)", t)
    ctype = m.group(1) if m else "未分类"
    m2 = re.search(r"内容[:：]\s*(.*)", t, re.DOTALL)
    content = (m2.group(1) if m2 else t).strip()
    return {"type": ctype, "content": content}


def describe_image(image_path: str) -> dict | None:
    """上传图片 → 视觉 LLM 理解 → {"type": ..., "content": ...}。

    需配置 VISION_MODEL + VISION_BASE_URL + VISION_API_KEY（或回退到主配置）。
    失败（未配置/网络/思考过长无 content/内容为空）返回 None。
    """
    config = settings.vision
    if not config.is_configured:
        print("[ocr] 未配置 VISION_MODEL/VISION_BASE_URL/VISION_API_KEY")
        return None
    api_key, base_url, model = config.api_key, config.base_url, config.model_name

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
                {"type": "text", "text": _DESCRIBE_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "temperature": 0.1,
        "max_tokens": 800,
    }

    print(f"[ocr] 调用视觉模型 {model} 理解图片 ...")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
    except requests.exceptions.RequestException as e:
        print(f"[ocr] 请求异常: {e}")
        return None

    if resp.status_code != 200:
        print(f"[ocr] HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code in (400, 404):
            print("[ocr] 该模型可能不支持视觉输入，请在 .env 中设置 VISION_MODEL 为支持图片的模型（如 GLM-4.6V）。")
        return None

    try:
        msg = resp.json()["choices"][0]["message"]
    except (KeyError, ValueError):
        return None
    content = msg.get("content") or ""
    # 不回退 reasoning_content：视觉模型思考过长时 content 为空，此时返回 None，
    # 而不是把思考过程当图片描述。
    if not content.strip():
        return None
    desc = _parse_description(content)
    return desc if desc["content"] else None
