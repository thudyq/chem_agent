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

# 非化合物名的角色/流程 label（过滤：不作为 PubChem 查询依据）
_ROLE_LABELS = {
    "反应物", "产物", "中间体", "底物", "亲核试剂", "亲电试剂",
    "过渡态", "离去基团", "溶剂", "催化剂", "加成产物", "σ络合物",
    "σ 络合物", "氧负离子", "碳正离子", "自由基",
}

# SMILES 相关失败原因关键词（值得 PubChem 兜底的类型）
_SMILES_FAILURE_KEYWORDS = (
    "无效 SMILES", "价态", "化学校验", "显式 H", "XH",
    "超出", "越界", "不存在", "不成键",
)

# 中文化学名 → 英文翻译提示（轻量，不加载完整化学 prompt）
_TRANSLATE_SYSTEM = (
    "你是化学名称翻译器。把用户输入的中文化学名称翻译成标准的英文名"
    "（IUPAC 或常用俗名）。只输出英文名称本身，不要解释、不要加引号。"
    "如果输入已是英文，原样输出。"
)


def _extract_chem_labels(failures: list) -> list:
    """从失败标记提取化合物 label（过滤角色/流程词）。

    failures: [(RenderTag, 原因字符串), ...]。
    返回 [(label, 原因), ...]——label 为疑似化合物名（args[1]）。
    """
    labels = []
    for tag, err in failures:
        if not err or not any(k in err for k in _SMILES_FAILURE_KEYWORDS):
            continue
        if tag.type != "STRUCT" or len(tag.args) < 2:
            continue
        label = str(tag.args[1]).strip() if tag.args[1] else ""
        if not label or label in _ROLE_LABELS:
            continue
        labels.append((label, err))
    return labels


def _extract_names_from_question(question: str) -> list:
    """从用户问题提取化合物名（REACTION/无 label 标记的兜底来源）。

    匹配 "乙醇被高锰酸钾氧化" 中的物质名——按常见分隔词切分，
    取中文/英文词片段（排除反应/机理/方程等流程词）。
    """
    import re
    if not question or not isinstance(question, str):
        return []
    # 按"被/与/和/加/氧化/还原/生成/得/反应"等切分，取名词片段
    tokens = re.split(r"被|与|和|加|氧化|还原|生成|反应|方程式|的|，|。| ", question)
    names = []
    for tok in tokens:
        tok = tok.strip()
        # 中文名（≥2 字）或英文名（含字母）
        if (re.fullmatch(r"[\u4e00-\u9fff]{2,10}", tok)
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9\- ]{1,20}", tok)):
            if tok not in _ROLE_LABELS and not any(
                    k in tok for k in ("反应", "机理", "方程", "氧化数")):
                names.append(tok)
    return names[:3]


def _translate_name_zh2en(name: str) -> str | None:
    """中文化学名 → 英文（LLM 翻译）；已是英文或翻译失败返回原样/None。"""
    if not name or not isinstance(name, str):
        return None
    import re
    if re.fullmatch(r"[A-Za-z0-9\- ]+", name):
        return name  # 已是英文，无需翻译
    try:
        translated = ask_llm(name, system_prompt=_TRANSLATE_SYSTEM,
                             max_tokens=64, thinking="disabled")
        if translated:
            t = translated.strip().strip('"\'。.')
            if t and len(t) < 60:
                return t
    except Exception:
        pass
    return None


def _fetch_pubchem_references(failures: list, user_question: str = "",
                              limit: int = 2) -> str:
    """校验失败 → PubChem 兜底：提取 label/问题名 → 翻译 → 查 SMILES → 参考。

    名称来源优先级：① 失败标记的 label（STRUCT）；② 用户问题中的化合物名
    （REACTION 等无 label 标记）。任一步失败静默跳过。限流：最多 limit 个。
    """
    from utils.name_resolver import name_to_smiles

    # 候选名称：label + 问题提取
    candidates = []
    for label, _ in _extract_chem_labels(failures):
        candidates.append(label)
    if user_question:
        for name in _extract_names_from_question(user_question):
            candidates.append(name)

    refs = []
    seen = set()
    for label in candidates:
        if label in seen:
            continue
        seen.add(label)
        if len(refs) >= limit:
            break
        en = _translate_name_zh2en(label)
        if not en:
            continue
        try:
            smi = name_to_smiles(en)
        except Exception:
            continue
        if smi:
            refs.append(f"「{label}」的 PubChem 标准 SMILES：`{smi}`")
    if not refs:
        return ""
    return ("\nPubChem 参考（权威 SMILES，可对照修正你的标记）：\n"
            + "\n".join(f"- {r}" for r in refs))


def _fetch_synrbl_balance(failures: list) -> str:
    """化学校验失败的 REACTION 标记 → SynRBL 补全配平参考（失败静默返回空串）。

    failures: [(RenderTag, 原因字符串), ...]。仅处理原因带「化学校验：」
    的 REACTION 标记（两侧原子不守恒），且两侧物种全为合法 SMILES 才尝试。
    """
    try:
        from utils.rxn_balancer import balance_reaction
    except Exception:
        return ""
    hints = []
    for tag, err in failures:
        if tag.type != "REACTION" or not err:
            continue
        if "化学校验：" not in err:
            continue
        if len(tag.args) < 2:
            continue
        hint = balance_reaction(tag.args[0], tag.args[1])
        if hint:
            hints.append(f"- `{tag.raw[:80]}` → {hint}")
    if not hints:
        return ""
    return "配平参考（SynRBL 自动补全，已通过守恒校验，可对照修正）：\n" \
        + "\n".join(hints)


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

    # PubChem 兜底：失败标记的 label 是化合物名时，反查权威 SMILES 作为修正参考
    pubchem_ref = _fetch_pubchem_references(failures, user_question)
    if pubchem_ref:
        lines.append("")
        lines.append(pubchem_ref)

    # SynRBL 配平兜底：化学校验失败（原子不守恒）的 REACTION 标记，尝试用
    # SynRBL 自动补全缺失物种，把配平后的方程式作为修正参考（LLM 直接照抄）。
    # 失败/超时/不可用一律静默，不影响原修正提示。
    synrbl_ref = _fetch_synrbl_balance(failures)
    if synrbl_ref:
        lines.append("")
        lines.append(synrbl_ref)

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


def process_question(user_question: str, max_corrections: int = 2,
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
    #    可选增强：用户问题含明确化学名称（"画 X 的结构/分子式"）时，
    #    PubChem 反查 SMILES 作为参考上下文注入，帮助 LLM 输出正确结构
    llm_input = user_question
    try:
        from utils.name_resolver import name_to_smiles
        import re as _re
        # 提取化学名称（中文/英文/混合），排除"反应/机理/方程"类问题
        m = _re.search(
            r"(?:画出|画|绘制)?\s*"
            r"([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9\- ]{0,29}?)\s*(?:的)?"
            r"(?:结构(?:式)?|分子式|怎么写|是什么结构)", user_question)
        if m:
            chem_name = m.group(1).strip()
            for _p in ("画出", "画 ", "绘制", "画"):
                if chem_name.startswith(_p):
                    chem_name = chem_name[len(_p):].strip()
                    break
            if chem_name and not _re.search(r"反应|机理|方程", user_question):
                pub_smiles = name_to_smiles(chem_name)
                if pub_smiles:
                    llm_input = (
                        f"[参考] 化合物「{chem_name}」的 PubChem 标准 SMILES 为"
                        f" `{pub_smiles}`（仅作结构参考，请用 [STRUCT:...] 输出）。\n"
                        f"用户问题：{user_question}"
                    )
                    print(f"[process_question] PubChem 名称→SMILES: "
                          f"{chem_name} -> {pub_smiles}")
    except Exception as e:
        print(f"[process_question] PubChem 增强跳过: {e}")

    full_response = ask_llm(llm_input, history=history,
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
