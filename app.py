# -*- coding: utf-8 -*-
"""app.py — 主入口：LLM → 标记解析 → 校验 → 渲染 → 注入 的端到端管线。

process_question(user_question) 是核心编排函数，FastAPI 适配层
（api.py）在此基础上包装 /chat/completions 端点。
"""

from core.llm_client import ask_llm
from core.tag_parser import parse_tags
from core.tag_injector import inject_tags_into_text
from core.tag_validator import degrade_text, validate_tags
from renderers.registry import RENDERER_REGISTRY

# 渲染器失败串的统一前缀（各渲染器内部约定："（XX渲染失败：原因）"）
_RENDER_ERROR_PREFIX = "（"


def _build_correction_prompt(user_question: str, original: str,
                             failures: list) -> str:
    """构造 P2 修正 prompt：原始问题 + 失败标记清单 + 修正要求。

    failures: [(RenderTag, 失败原因字符串), ...]。
    """
    lines = [
        "你刚才的回答中有一些化学标记无法渲染。请修正后重新输出完整的回答。",
        "",
        "原始用户问题：",
        user_question,
        "",
        "你的上一个回答：",
        original,
        "",
        "渲染失败的标记及原因：",
    ]
    for tag, err in failures[:10]:
        lines.append(f"- {tag.raw}：{err}")
    lines += [
        "",
        "修正要求：",
        "1. 保持回答的内容和结构不变，只修正上述失败标记"
        "（SMILES、原子编号、组件引用、格式、化学一致性——"
        "label 与 SMILES 指向同一物质、方程式两侧原子守恒）。",
        "2. 不要新增或删除其他标记。",
        "3. 修正后的标记必须严格遵循标记语法。",
        "4. 若失败原因是化学校验（两侧原子不守恒或净电荷不守恒）：先核对两侧"
        "元素计数与电荷，补全缺失的具体反应物/生成物（如催化脱氢/芳构化确实"
        "放 H2 才补 -H2），或修正化学计量系数（整数或 n/2，如 2CCO、1/2O2）"
        "使两侧守恒；也可用 2b 箭头补足——把省略的具体物质写进反应条件"
        "（无符号=反应物侧补足、如 H2O；\"-\"前缀=产物侧补足、如 -H2O），"
        "补上后两侧守恒即可；"
        "**氧化剂（KMnO4/K2Cr2O7 等）参与反应（被还原）时是反应物，必须写"
        "完整配平方程式，不能用 2b 省略**（如乙醇被 KMnO4 氧化产物是乙酸："
        "5CCO+4KMnO4+6H2SO4→5CH3COOH+4MnSO4+2K2SO4+11H2O）；"
        "**氧化反应不要补 -H2**（氧化不放氢气，H 与氧化剂供的 O 结合成水）；"
        "禁止用 [O]/[H] 占位符配平，补足物质必须真实参与该反应。"
        "催化剂/溶剂等辅助试剂写在箭头条件里，不列入反应物或产物列表。"
        "单→单反应建议改用 [ARROW]（只画主物种，不做全元素守恒）。"
        "**原始用户问题是\"化学方程式/配平\"时：必须保持 REACTION 并补全物种"
        "使守恒（配离子带电荷写，如银氨 [Ag+]([NH3])([NH3])、氢氧根 [OH-]，"
        "按电荷配系数），不得降级为 ARROW——ARROW 只在用户只要转化示意时使用。**",
        "5. 若失败原因是无效 SMILES（如 [Ag(NH3)2]OH、NH3 裸写）：SMILES 不是"
        "化学式——配位化合物/络离子（银氨 [Ag(NH3)2]+ 等）不能用 \"(NH3)2\" 表示"
        "配位，氨必须写 [NH3] 或 N（RDKit 中 NH3 裸写非法）。改法：络离子拆成"
        "可表达的组分（如银氨写 [Ag]([NH3])[NH3] 或 [Ag+]，氨写 N，氢氧化银"
        "写 [Ag+] 与 [OH-] 分离）；实在写不出合法 SMILES 的物种（如复杂配合物）"
        "降级为文字描述或从方程式中省略，只保留能渲染的主物种。",
        "6. 若失败原因是无机盐/含氧酸盐 SMILES 非法（如 KMnO4 写成 K[Mn](=O)(=O)=O"
        "或 KMn(=O)=O——金属与中心原子无直接键）：改离子式——阳离子 [K+]/[Na+]"
        "与阴离子用 . 分隔，含氧酸根中心原子带足双键氧：KMnO4 写"
        "[K+].[O-][Mn](=O)(=O)=O，K2Cr2O7 写 [K+].[K+].[O-][Cr](=O)(=O)O[Cr]"
        "(=O)(=O)[O-]，KClO3 写 [K+].[O-][Cl](=O)=O，Na2CO3 写"
        "[Na+].[Na+].[O-]C(=O)[O-]，H2SO4 写 OS(=O)(=O)O，HNO3 写"
        "[O-][N+](=O)O。",
    ]
    return "\n".join(lines)


def process_question(user_question: str, max_corrections: int = 1,
                     history: list = None, progress_callback=None,
                     correction_callback=None) -> str:
    """端到端处理用户问题，返回含渲染后图示代码的文本。

    流程：LLM 生成 → 解析标记 → 契约校验（P1）→ 逐标记渲染 → 注入替换。
    P2 渲染反馈闭环：首次渲染若有失败（校验拦截 / 渲染器失败），携带失败
    清单回传 LLM 自动修正（最多 max_corrections 次），修正版重新走管线；
    仍失败则降级（校验失败标记 → 友好提示，渲染失败标记 → 渲染器错误串）。
    history: 多轮对话历史（透传给 ask_llm，见 core.llm_client）。
    progress_callback: 可选，LLM 每段生成内容实时回调（B2 流式转发草稿）。
    correction_callback: 可选，P2 修正触发时回调（无参），前端据此提示
        "正在修正回答…"；修正调用强制 thinking=disabled（机械性任务，
        思考链收益小、延迟高）。
    """
    # 1. 调用 LLM（自动加载 system prompt，含标记协议）
    full_response = ask_llm(user_question, history=history,
                            on_piece=progress_callback)
    if not full_response:
        return "（LLM 调用失败，请检查 .env 配置与网络）"

    for attempt in range(max_corrections + 1):
        # 2. 解析标记
        tags = parse_tags(full_response)
        if not tags:
            return full_response  # 纯文本回答，无需渲染

        # 2.5 标记契约校验（P1）：渲染前拦截坏参数（非法 SMILES / 越界引用 /
        #    超长 label / 格式错误），降级为友好提示，坏参数不进渲染器
        valid_tags, invalid = validate_tags(tags)
        degraded = {r.tag.raw: degrade_text(r.tag, r.reason) for r in invalid}

        # 3. 逐标记渲染（REASONING 无渲染器，由注入器特殊处理）
        rendered, failures = {}, []
        for tag in valid_tags:
            if tag.type == "REASONING":
                continue
            renderer = RENDERER_REGISTRY.get(tag.type)
            if renderer is None:
                continue  # 未注册类型，注入时保留原标记
            try:
                out = renderer(*tag.args)
            except Exception as e:
                out = f"（{tag.type} 渲染失败：{e}）"
            if out.startswith(_RENDER_ERROR_PREFIX):
                failures.append((tag, out))
            else:
                rendered[tag.raw] = out

        # 3.5 P2 渲染反馈闭环：有失败（校验拦截或渲染失败）且还有修正机会
        #     → 回传 LLM 修正重试
        problems = [(r.tag, r.reason) for r in invalid] + failures
        if problems and attempt < max_corrections:
            print(f"[process_question] {len(problems)} 个标记未通过校验/渲染，"
                  f"回传 LLM 修正（第 {attempt + 1}/{max_corrections} 次）：")
            for tag, err in problems[:5]:
                print(f"  - {tag.raw[:60]} → {err[:80]}")
            if correction_callback is not None:
                correction_callback()
            correction = _build_correction_prompt(
                user_question, full_response, problems)
            fixed = ask_llm(correction, on_piece=progress_callback,
                            thinking="disabled")
            if fixed:
                full_response = fixed
                continue

        # 4. 注入：校验失败 → 降级提示；渲染失败（重试机会耗尽）→ 渲染器错误串
        rendered.update(degraded)
        for tag, err in failures:
            rendered.setdefault(tag.raw, err)
        return inject_tags_into_text(full_response, tags, rendered)

    return "（LLM 调用失败，请检查 .env 配置与网络）"


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "请画出苯的结构式，并说明它的分子式"
    print("=" * 60)
    print(f"用户问题：{question}")
    print("=" * 60)
    print(process_question(question))
