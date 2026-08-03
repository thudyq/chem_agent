# -*- coding: utf-8 -*-
"""app.py — 主入口：LLM → 标记解析 → 渲染 → 注入 的端到端管线。

process_question(user_question) 是核心编排函数，后续 Day 17-18 的 FastAPI
适配层会在此基础上包装 /chat/completions 端点。
"""

from core.llm_client import ask_llm
from core.tag_parser import parse_tags
from core.tag_injector import inject_tags_into_text
from core.tag_validator import degrade_text, validate_tags
from renderers.registry import RENDERER_REGISTRY


def process_question(user_question: str) -> str:
    """端到端处理用户问题，返回含渲染后图示代码的文本。

    流程：LLM 生成 → 解析标记 → 标记契约校验（P1）→ 逐标记渲染 → 注入替换。
    LLM 失败、无标记、部分标记未注册或校验失败均优雅降级。
    """
    # 1. 调用 LLM（自动加载 system prompt，含标记协议）
    full_response = ask_llm(user_question)
    if not full_response:
        return "（LLM 调用失败，请检查 .env 配置与网络）"

    # 2. 解析标记
    tags = parse_tags(full_response)
    if not tags:
        return full_response  # 纯文本回答，无需渲染

    # 2.5 标记契约校验（P1）：渲染前拦截坏参数（非法 SMILES / 越界引用 /
    #    超长 label / 格式错误），降级为友好提示，坏参数不进渲染器
    valid_tags, invalid = validate_tags(tags)
    degraded = {r.tag.raw: degrade_text(r.tag, r.reason) for r in invalid}

    # 3. 逐标记渲染（REASONING 无渲染器，由注入器特殊处理）
    rendered = {}
    for tag in valid_tags:
        if tag.type == "REASONING":
            continue
        renderer = RENDERER_REGISTRY.get(tag.type)
        if renderer is None:
            continue  # 未注册类型（如 NEWMAN/ENERGY 暂未实现），注入时保留原标记
        try:
            rendered[tag.raw] = renderer(*tag.args)
        except Exception as e:
            rendered[tag.raw] = f"（{tag.type} 渲染失败：{e}）"

    # 4. 校验失败标记注入降级提示（否则注入器会保留原始标记文本）
    rendered.update(degraded)
    return inject_tags_into_text(full_response, tags, rendered)


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "请画出苯的结构式，并说明它的分子式"
    print("=" * 60)
    print(f"用户问题：{question}")
    print("=" * 60)
    print(process_question(question))
