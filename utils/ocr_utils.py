# -*- coding: utf-8 -*-
"""utils/ocr_utils.py — 图片多模态理解（视觉大模型）。

方案 B 两段式：视觉模型（如 GLM-4.6V）负责"理解"图片——转录文字、
结构式给 SMILES、反应式列三要素、机理图逐步描述弯箭头去向；
产出结构化描述交给主模型按标记体系答题。无需 DECIMER 等重 ML 依赖。
用户需配置支持视觉的模型（VISION_MODEL 等，如 GLM-4.6V / Qwen2-VL / GPT-4o）。
"""

import base64
import re
import time

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

# 视觉调用总尝试次数：初始 1 次 + 失败重试 2 次（连接不稳定场景，20260818）
_VISION_MAX_ATTEMPTS = 3
# 重试间隔秒数（避免瞬时限流/抖动时立即连打）
_RETRY_DELAY = 1.0


def _parse_description(text: str) -> dict:
    """解析"类型：/内容："两行格式；不合格式时整体作为 content（type=未分类）。"""
    t = (text or "").strip()
    m = re.search(r"类型[:：]\s*(\S+)", t)
    ctype = m.group(1) if m else "未分类"
    m2 = re.search(r"内容[:：]\s*(.*)", t, re.DOTALL)
    content = (m2.group(1) if m2 else t).strip()
    return {"type": ctype, "content": content}


def _is_valid_smiles(smi: str) -> bool:
    """SMILES 是否可解析（RDKit）；无 rdkit 环境保守放行 True。"""
    try:
        from utils.rdkit_utils import validate_smiles
        return validate_smiles(smi)
    except ImportError:
        return True


def _structure_smiles_ok(content: str, ctype: str) -> bool:
    """B1（20260826）：结构式内容里声称的 SMILES 是否可解析（RDKit 硬校验）。

    - 非结构式（文字题/反应式/机理图/其他/未分类）→ True（未声称具体结构，
      无可核验的 SMILES 断言）；
    - 显式 "SMILES: xxx" 声明 → 校验声明的 token，非法则 False；
    - 无显式声明、且内容为"单一、无汉字、无空格"的化学串 → 当裸 SMILES
      校验（非法则 False）；
    - 其余（含汉字的文字描述，如"苯环连一个硝基（无法确定 SMILES）"）→ True，
      不做硬判（避免把描述文字误判为错误 SMILES）。
    说明：只能拦截"不可解析的垃圾"，拦不住"合法但认错"的分子（如
    环癸二炔被认成环辛四烯——两者都是合法 SMILES）。
    """
    if ctype != "结构式":
        return True
    s = (content or "").strip()
    if not s:
        return True
    # 显式 SMILES: 声明——token 只取纯 SMILES 字符，碰到中文/括号/句逗即止，
    # 否则会把"c1ccccc1（等价写法：C1=CC=CC=C1）。"整串拿去解析而误判非法
    # （模型其实已给出合法 SMILES，只是带了注释，应视为正确并放行）。
    # 允许的 SMILES 字符：字母数字 + 成键/环/立体/电荷/配位等符号。
    claim = re.search(r"SMILES\s*[:：]\s*([^\s，。；、()（）【】\[\]{}]*[\w=\-+#@/[\]()*%\.\\]+)", s)
    if claim:
        tok = claim.group(1).strip().rstrip("，。；、()（）")
        if tok and not re.search(r"[\u4e00-\u9fff]", tok):
            return _is_valid_smiles(tok)
        # 声明里混了中文（描述性）→ 不做硬判，放行
        return True
    # 无显式声明：单一、无汉字、无空格的化学串 → 当裸 SMILES 校验
    if not re.search(r"[\u4e00-\u9fff]", s) and not re.search(r"\s", s):
        return _is_valid_smiles(s)
    return True



def _read_image_b64(image_path: str) -> str | None:
    """读图片文件 → base64 字符串；失败返回 None。"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"[ocr] 读图失败: {e}")
        return None


def _describe_once(url: str, headers: dict, payload: dict,
                   model: str) -> tuple:
    """单次视觉调用。返回 (desc, retryable)：

    - desc 非 None：本次成功；
    - retryable=True：本次失败但值得重试（网络异常 / 5xx / 空响应）；
    - retryable=False：配置性失败（400/404 不支持视觉），重试无意义。
    """
    print(f"[ocr] 调用视觉模型 {model} 理解图片 ...")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.exceptions.RequestException as e:
        print(f"[ocr] 请求异常: {e}")
        return None, True

    # 端点不识别 thinking 参数（如 Gemini OpenAI 兼容端点报
    # 'Unknown name "thinking"'）→ 去掉该参数重试一次（自适应：
    # 智谱带 thinking disabled，Gemini 等不带）
    if resp.status_code == 400 and "thinking" in (resp.text or ""):
        print("[ocr] 端点不识别 thinking 参数，去掉重试 ...")
        payload.pop("thinking", None)
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
        except requests.exceptions.RequestException as e:
            print(f"[ocr] 请求异常: {e}")
            return None, True

    if resp.status_code != 200:
        print(f"[ocr] HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code in (400, 404):
            print("[ocr] 该模型可能不支持视觉输入，请在 .env 中"
                  "设置 VISION_MODEL 为支持图片的模型（如 GLM-4.6V）。")
            return None, False  # 配置性错误，重试无意义
        return None, True  # 5xx/429 等服务端/限流错误，可重试

    try:
        msg = resp.json()["choices"][0]["message"]
    except (KeyError, ValueError):
        return None, True
    content = msg.get("content") or ""
    # 思考型模型兜底：content 为空但 reasoning_content 有内容时回退提取
    # （thinking disabled 生效时 content 直接有值，此分支为兼容不识别
    # 该参数的端点）
    if not content.strip():
        reasoning = msg.get("reasoning_content") or ""
        if reasoning.strip():
            print("[ocr] content 为空，回退 reasoning_content")
            content = reasoning
        else:
            return None, True  # 空响应（瞬时抖动），可重试
    desc = _parse_description(content)
    return (desc if desc["content"] else None), True


def describe_image(image_path: str, max_attempts: int = _VISION_MAX_ATTEMPTS) -> dict | None:
    """上传图片 → 视觉 LLM 理解 → {"type": ..., "content": ...}。

    单模型整图描述（glm-4.6v 等）：关闭深度思考（thinking disabled）让
    模型直接输出 content——glm-4.6v 思考型行为会把回答吞进 reasoning_content
    致 content 为空（20260817 实测：本地失败/智谱平台成功即此差异）；
    个别端点不识别 thinking 参数或 content 仍空时，回退 reasoning_content。

    连接不稳定容错（20260818）：失败（网络异常 / 5xx / 空响应）自动重试，
    默认最多 max_attempts=3 次（初始 1 次 + 重试 2 次）；配置性失败
    （未配置 / 400/404 不支持视觉 / 本地读图失败）不重试直接返回 None。

    需配置 VISION_MODEL + VISION_BASE_URL + VISION_API_KEY（或回退到主配置）。
    失败（未配置/网络/无 content）返回 None。
    """
    config = settings.vision
    if not config.is_configured:
        print("[ocr] 未配置 VISION_MODEL/VISION_BASE_URL/VISION_API_KEY")
        return None
    api_key, base_url, model = config.api_key, config.base_url, config.model_name

    b64 = _read_image_b64(image_path)
    if b64 is None:
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
        # 800 → 2000：复杂图（教材文字+反应式）描述长，避免思考/回答被截断
        "max_tokens": 2000,
        # 关闭深度思考（智谱 thinking 参数）：直接输出 content，避免
        # 回答进 reasoning_content 致 content 空（20260817 本地失败根因）
        "thinking": {"type": "disabled"},
    }

    for attempt in range(1, max_attempts + 1):
        desc, retryable = _describe_once(url, headers, dict(payload), model)
        if desc:
            # B1（20260826）：结构式 SMILES 过 RDKit 硬校验，供调用方示警
            desc["smiles_ok"] = _structure_smiles_ok(
                desc.get("content"), desc.get("type"))
            return desc
        if not retryable or attempt >= max_attempts:
            return None
        print(f"[ocr] 第 {attempt} 次尝试失败，重试（{attempt + 1}/{max_attempts}）...")
        time.sleep(_RETRY_DELAY)
    return None

