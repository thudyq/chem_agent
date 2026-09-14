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

from core import credentials

# 视觉调用超时拆成（建连, 读取）（安全审查 R5）：不可达/被丢包的端点快速失败，
# 不再让每个请求干等满整个超时（识图本身不需要长思考，读取 60s 足够）。
VISION_CONNECT_TIMEOUT = 10
VISION_READ_TIMEOUT = 60
VISION_TIMEOUT = (VISION_CONNECT_TIMEOUT, VISION_READ_TIMEOUT)

_DESCRIBE_PROMPT = """请描述这张图片的内容，供后续化学问答使用。按内容类型分别处理：
1. 文字（题目/问题/解答等）：完整转录原文，不要改写。
2. 化学结构式：给出 SMILES（能确定时）；不能确定时用文字描述（如"苯环连一个硝基"）。
3. 反应方程式：列出反应物、产物、反应条件，分子尽量用 SMILES。
4. 反应机理图（弯箭头/电子转移）：分步文字描述——每步哪个原子或键的电子流向哪里。
5. 其他图（势能面/构象/能级等）：说明图的类型与关键数值或标注。

输出格式（严格两行）：
类型：文字题 | 结构式 | 反应式 | 机理图 | 混合 | 其他
内容：<按上述要求的转录与描述>

纪律（必须遵守）：
- 先识别并转录图上的文字（题目/标签/数值/图例/条件等），文字优先，别漏。
- 内容只放结论性描述，把分析/犹豫过程放到思考里，不要写进"内容"。
- 有多个对象（多个结构/多步/多取代基）时，用 [1] [2] 编号逐条列出。
- 无法从图中确定的部分（如取代基/自由基的确切位置）不要编造确定答案：
  标"不确定"并给最有把握的判断；能确定位置时就给出邻/间/对或编号。
- 关键数值、图例、坐标轴刻度、图上文字要逐字转录，不要概括。
- 尽快输出这两行结论，不要长时间空想；若图过于复杂或关键信息无法可靠
  确定，也在"内容"里给出你确定的部分（至少转录到的文字），不要给空。"""

# 视觉调用总尝试次数：初始 1 次 + 失败重试 2 次（连接不稳定场景）
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
    """结构式内容里声称的 SMILES 是否可解析（RDKit 硬校验）。

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
                   model: str, cap_key: str = "") -> tuple:
    """单次视觉调用。返回 (desc, retryable)：

    - desc 非 None：本次成功；
    - retryable=True：本次失败但值得重试（网络异常 / 5xx / 空响应）；
    - retryable=False：配置性失败（400/404 不支持视觉），重试无意义。

    端点参数适配（重构 §4.8 / 缺陷 A）：与主客户端**同一套行为判定**——
    非 200 时按字段名逐个摘掉重试（`thinking` → `reasoning_effort` →
    `temperature` → `max_tokens`），**摘掉后成功**即认定该字段是原因并记入
    能力表（`core.capabilities`）。不解析错误文本（GLM 的拒绝是中文，
    按字段名匹配会漏判）。
    """
    print(f"[ocr] 调用视觉模型 {model} 理解图片 ...")
    dropped = set()
    while True:
        try:
            # ★ 不跟随重定向（安全审查 R3）：视觉端点地址已经过 SSRF 校验，
            # 但 302 之后跳到哪不受校验管 —— 跟随等于把校验作废。
            # 超时拆成（建连, 读取）（安全审查 R5）：不可达地址快速失败。
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=VISION_TIMEOUT, allow_redirects=False)
        except requests.exceptions.RequestException as e:
            print(f"[ocr] 请求异常: {e}")
            return None, True

        if resp.status_code == 200:
            break

        # 3xx：配置性问题（端点会跳转），重试无意义，直接把话说清楚
        if 300 <= resp.status_code < 400:
            print(f"[ocr] 端点返回重定向（HTTP {resp.status_code}）→ 拒绝跟随"
                  "（安全策略）；请把**最终**的完整地址填进「视觉接口地址」")
            return None, False

        # 非 200：先尝试"摘字段重试"（仅 400/422，且还有可摘字段）
        if resp.status_code in (400, 422):
            stripped = _strip_optional_field(payload, dropped)
            if stripped is not None:
                field, value = stripped
                print(f"[ocr] HTTP {resp.status_code} → 摘掉 {field}={value!r} 重试")
                try:
                    resp2 = requests.post(url, headers=headers, json=payload,
                                          timeout=VISION_TIMEOUT,
                                          allow_redirects=False)
                except requests.exceptions.RequestException as e:
                    print(f"[ocr] 请求异常: {e}")
                    return None, True
                if resp2.status_code == 200:
                    _learn_rejected(cap_key, field, value)
                    print(f"[ocr] 确认：端点拒绝 {field}={value!r}（已记入能力表）")
                    resp = resp2
                    break
                print(f"[ocr] 摘掉 {field} 后仍失败（{resp2.status_code}）"
                      f"→ 判定与思考参数无关的真实错误")
                resp = resp2

        print(f"[ocr] HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code in (400, 404):
            print("[ocr] 该模型可能不支持视觉输入，请配置一个支持图片的模型"
                  "（如 glm-5.3-flash / gemini-3.7-flash / deepseek-flash）。")
            return None, False  # 配置性错误，重试无意义
        return None, True  # 5xx/429 等服务端/限流错误，可重试

    try:
        msg = resp.json()["choices"][0]["message"]
    except (KeyError, ValueError):
        return None, True
    content = msg.get("content") or ""
    if not content.strip():
        # ★ **不再**回退 reasoning_content。
        # 理由：思考草稿是未完成的推理过程，把它当"识别结果"会让错误结构
        # 静默到达用户（与 Drawbacks §P0 修过的老 bug 同型，也与本项目
        # "宁可不画，不画错"的理念冲突）。content 为空 → 判识别失败，
        # 由调用方提示用户重试或改用文字描述。
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if reasoning.strip():
            print(f"[ocr] 模型只输出思考（{len(reasoning)} 字符）而无识别结论"
                  f" → 判识别失败（不使用思考草稿）")
        else:
            print("[ocr] 空响应（瞬时抖动），可重试")
        return None, True
    desc = _parse_description(content)
    return (desc if desc["content"] else None), True


# 视觉端点的可选字段（摘字段顺序与主客户端一致）
_VISION_STRIP_ORDER = ("thinking", "reasoning_effort", "temperature", "max_tokens")


def _strip_optional_field(payload: dict, dropped: set) -> tuple | None:
    """从视觉请求体里摘掉**一个**可选字段；无可摘时返回 None。"""
    for field in _VISION_STRIP_ORDER:
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


def _learn_rejected(cap_key: str, field: str, value) -> None:
    """把"端点拒绝该字段/取值"记入能力表（键为空则跳过）。"""
    if not cap_key:
        return
    try:
        from core import capabilities
        capabilities.note_field_rejected(cap_key, field, value)
    except Exception:
        pass


def describe_image(image_path: str, max_attempts: int = _VISION_MAX_ATTEMPTS) -> dict | None:
    """上传图片 → 视觉 LLM 理解 → {"type", "content", "smiles_ok"}。

    思考参数默认显式发 disabled；端点强制思考（如 GLM 对 disabled 报 400）
    时摘字段重试并记入能力表。若模型把结论写进 reasoning_content 致
    content 空，直接判失败返回 None（不回退抢救草稿——草稿透传会污染
    下游 SMILES 校验）。

    连接不稳定容错：失败（网络异常 / 5xx / 空响应）自动重试，
    默认最多 max_attempts=3 次（初始 1 次 + 重试 2 次）；配置性失败
    （未配置 / 400/404 不支持视觉 / 本地读图失败）不重试直接返回 None。

    需配置 VISION_MODEL + VISION_BASE_URL + VISION_API_KEY（或回退到主配置）。
    失败（未配置/网络/无 content）返回 None。
    """
    config = credentials.vision_config()
    if not config.is_configured:
        print("[ocr] 视觉模型未配置（且主模型凭证不可用）")
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

    # ---- 视觉自己的思考参数（§4.9）----
    # 与主生成**解耦**：视觉常是另一家的模型，且识图不需要长思考。
    # `config.thinking` 为空 = "与主模型相同 → 复用主模型设置"（由调用方
    # credentials.vision_is_main() 判定后写入空值）；否则默认关思考。
    from core.llm_client import MAX_TOKENS_VISION
    v_on, v_effort = _vision_thinking(config)

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _DESCRIBE_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        # 视觉输出上限（§4.2.1）：复杂机理图描述长，且强制思考端点还要吃掉
        # 一部分预算——设小会导致 content 被截断（缺陷 E）。
        "max_tokens": MAX_TOKENS_VISION,
    }
    if v_on:
        payload["thinking"] = {"type": "enabled"}
        if v_effort:
            payload["reasoning_effort"] = v_effort
    else:
        # 关思考：显式发送 disabled（与主客户端同一策略——这样端点若**强制**
        # 思考会返回 400，我们据此摘字段并把结论记入能力表）
        payload["thinking"] = {"type": "disabled"}
        payload["temperature"] = 0.1

    # 能力表：已知结论（该端点拒绝过某字段）→ 本次直接不带
    cap_key = ""
    try:
        from core import capabilities
        cap_key = capabilities.make_key(base_url, model, api_key)
        cap = capabilities.get(cap_key)
        if cap.allows_off() is False:
            payload.pop("thinking", None)
        if cap.max_tokens_ok is False:
            payload.pop("max_tokens", None)
    except Exception:
        pass

    try:
        # 在途闸门（安全审查 R5）：视觉调用同样是"带着用户凭证出站"，也要受
        # 全局/按 IP 上限约束 —— 否则攻击者可以绕过主模型的闸门，用视觉端点
        # 那台"永不回包的服务器"占满 worker。
        with credentials.inflight_guard():
            for attempt in range(1, max_attempts + 1):
                desc, retryable = _describe_once(url, headers, dict(payload), model,
                                                cap_key=cap_key)
                if desc:
                    # 结构式 SMILES 过 RDKit 硬校验，供调用方示警
                    desc["smiles_ok"] = _structure_smiles_ok(
                        desc.get("content"), desc.get("type"))
                    return desc
                if not retryable or attempt >= max_attempts:
                    return None
                print(f"[ocr] 第 {attempt} 次尝试失败，重试"
                      f"（{attempt + 1}/{max_attempts}）...")
                time.sleep(_RETRY_DELAY)
    except credentials.ServerBusy as e:
        print(f"[ocr] 跳过视觉识别：{e}")
        return None
    return None


def _vision_thinking(config) -> tuple:
    """视觉调用的思考意图 → (开关 on, 强度)。

    * `config.thinking` 显式给了 `on`/`off` → 用它；
    * 为空且**视觉就是主模型** → 复用主模型的思考设置（同一个模型）；
    * 为空且视觉是独立模型 → 默认关思考（识图要快、要省）。
    """
    from core.config import normalize_effort, normalize_thinking
    raw = (getattr(config, "thinking", "") or "").strip()
    if raw:
        on = normalize_thinking(raw) == "on"
    else:
        try:
            on = credentials.vision_is_main() and credentials.thinking_setting() == "on"
        except Exception:
            on = False
    eff_raw = (getattr(config, "effort", "") or "").strip()
    effort = (normalize_effort(eff_raw) if eff_raw
              else (credentials.effort_setting() if on else ""))
    return on, (effort if on else "")

