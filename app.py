# -*- coding: utf-8 -*-
"""app.py — 主入口：LLM → 标记解析 → 渲染 → 注入 的端到端管线。

process_question(user_question) 是核心编排函数，后续 Day 17-18 的 FastAPI
适配层会在此基础上包装 /chat/completions 端点。
"""

from core.llm_client import ask_llm
from core.tag_parser import parse_tags
from core.tag_injector import inject_tags_into_text
from renderers.base import RENDERER_REGISTRY


def process_question(user_question: str) -> str:
    """端到端处理用户问题，返回含渲染后图示代码的文本。

    流程：LLM 生成 → 解析标记 → 逐标记渲染 → 注入替换。
    LLM 失败、无标记、或部分标记未注册均优雅降级。
    """
    # 1. 调用 LLM（自动加载 system prompt，含标记协议）
    full_response = ask_llm(user_question)
    if not full_response:
        return "（LLM 调用失败，请检查 .env 配置与网络）"

    # 2. 解析标记
    tags = parse_tags(full_response)
    if not tags:
        return full_response  # 纯文本回答，无需渲染

    # 3. 逐标记渲染（REASONING 无渲染器，由注入器特殊处理）
    rendered = {}
    for tag in tags:
        if tag.type == "REASONING":
            continue
        renderer = RENDERER_REGISTRY.get(tag.type)
        if renderer is None:
            continue  # 未注册类型（如 NEWMAN/ENERGY 暂未实现），注入时保留原标记
        try:
            rendered[tag.raw] = renderer(*tag.args)
        except Exception as e:
            rendered[tag.raw] = f"（{tag.type} 渲染失败：{e}）"

    # 4. 注入替换
    return inject_tags_into_text(full_response, tags, rendered)


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "请画出苯的结构式，并说明它的分子式"
    print("=" * 60)
    print(f"用户问题：{question}")
    print("=" * 60)
    print(process_question(question))
