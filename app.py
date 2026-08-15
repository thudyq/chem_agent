# -*- coding: utf-8 -*-
"""app.py — 主入口：LLM → 标记解析 → 校验 → 渲染 → 注入 的端到端管线。

process_question(user_question) 是核心编排函数，FastAPI 适配层
（api.py）在此基础上包装 /chat/completions 端点。
"""

from core.config import settings
from core.llm_client import ask_llm
from core.tag_parser import parse_tags
from core.tag_injector import inject_tags_into_text
from core.tag_validator import degrade_text_friendly, validate_tags
from renderers.registry import RENDERER_REGISTRY
# 渲染器失败串的统一前缀（各渲染器内部约定："（XX渲染失败：原因）"）
_RENDER_ERROR_PREFIX = "（"

# 非化合物名的角色/流程 label（过滤：不作为 PubChem 查询依据）
_ROLE_LABELS = {
    "反应物", "产物", "中间体", "底物", "亲核试剂", "亲电试剂",
    "过渡态", "离去基团", "溶剂", "催化剂", "加成产物", "σ络合物",
    "σ 络合物", "氧负离子", "碳正离子", "自由基",
}

# 难题预判关键词（未配置 UPGRADE_KEYWORDS 时的内置默认）：
# 用户问题命中任一关键词 → 跳过主模型首跑、直接走升级模型。
# 覆盖机理/箭头/命名反应/电子流动类提问；漏判由 flash 首跑 + 失败升级兜底，
# 误判（简单题命中）代价仅为一次升级调用。
_DEFAULT_UPGRADE_KEYWORDS = (
    "机理", "箭头", "SN1", "SN2", "SN1/SN2", "自由基", "共振", "势能面",
    "电子转移", "电子推动", "消除反应", "亲核取代", "亲电取代", "亲核加成",
    "亲电加成", "加成反应", "重排", "去质子", "质子化", "催化循环",
    "链式反应", "过渡态", "反应历程", "轨道", "HOMO", "LUMO",
)

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


def _context_around(text: str, tag) -> str:
    """标记在原文中的上下文（前后各 ~40 字符），帮助部分修正时理解语境。"""
    start = getattr(tag, "start_pos", None)
    if start is None:
        return ""
    before = text[max(0, start - 40): start].replace("\n", " ")
    after = text[start + len(tag.raw): start + len(tag.raw) + 40].replace("\n", " ")
    ctx = f"{before} ⟦此处⟧ {after}".strip()
    return ctx[:160] if ctx else ""


def _build_correction_prompt(user_question: str, original: str,
                             failures: list) -> str:
    """构造 P2 修正 prompt：失败标记清单（含上下文）+ 修正要求。

    部分修正模式：模型**只输出修正后的标记**（不重输出整个回答），
    process_question 用修正标记替换原文对应位置后重新校验/渲染——
    相比"全篇重生成"大幅节省 token 且修正更聚焦。

    failures: [(RenderTag, 失败原因字符串), ...]。
    """
    lines = [
        "你刚才的回答中有一些化学标记无法渲染。下面列出每个失败标记及其原因，",
        "请为每个标记输出**修正后的标记**。",
        "",
        "原始用户问题：",
        user_question,
        "",
        "渲染失败的标记及原因：",
    ]
    for i, (tag, err) in enumerate(failures[:10], 1):
        lines.append(f"- 标记 {i}：{tag.raw}")
        lines.append(f"  原因：{err}")
        ctx = _context_around(original, tag)
        if ctx:
            lines.append(f"  上下文：{ctx}")

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
        "1. **只输出修正后的标记本身**（保持 [TYPE:...] 语法；COMPOSITE 容器要完整，"
        "含开闭标签），多个失败标记按上面顺序依次输出；不要输出解释、序号或任何其他文字。",
        "2. 不要新增或删除其他标记；修正后的标记必须严格遵循标记语法。",
        "3. 若失败原因是化学校验（两侧原子不守恒或净电荷不守恒）：先核对两侧"
        "元素计数与电荷，补全缺失的具体反应物/生成物（如催化脱氢/芳构化确实"
        "放 H2 才补 -H2），或修正化学计量系数（整数或 n/2，如 2CCO、1/2O2）"
        "使两侧守恒；也可用 2b 箭头补足——把省略的具体物质写进反应条件"
        "（无符号=反应物侧补足、如 H2O；\"-\"前缀=产物侧补足、如 -H2O），"
        "补上后两侧守恒即可；"
        "**同时核对每个物种的分子式与它在反应中的角色是否对应**（如 SN1/SN2"
        "离去步之后，碳正离子/底物的碳数必须与原料相同——不能多写或漏写碳）；"
        "**氧化剂（KMnO4/K2Cr2O7 等）参与反应（被还原）时是反应物，必须写"
        "完整配平方程式，不能用 2b 省略**（如乙醇被 KMnO4 氧化产物是乙酸："
        "5CCO+4KMnO4+6H2SO4→5CH3COOH+4MnSO4+2K2SO4+11H2O）；"
        "**无机物种（盐/酸/氧化物/单质）可直接写化学式**（KMnO4、H2SO4、"
        "MnSO4、K2SO4、H2O、O2、CO2——渲染为文本），不必转离子 SMILES；"
        "**氧化反应不要补 -H2**（氧化不放氢气，H 与氧化剂供的 O 结合成水）；"
        "禁止用 [O]/[H] 占位符配平，补足物质必须真实参与该反应。"
        "催化剂/溶剂等辅助试剂写在箭头条件里，不列入反应物或产物列表。"
        "单→单反应建议改用 [ARROW]（只画主物种，不做全元素守恒）。"
        "**原始用户问题是\"化学方程式/配平\"时：必须保持 REACTION 并补全物种"
        "使守恒（配离子带电荷写，如银氨 [Ag+]([NH3])([NH3])、氢氧根 [OH-]，"
        "按电荷配系数），不得降级为 ARROW——ARROW 只在用户只要转化示意时使用。**",
        "4. 若失败原因是无效 SMILES（如 [Ag(NH3)2]OH、NH3 裸写）：SMILES 不是"
        "化学式——配位化合物/络离子（银氨 [Ag(NH3)2]+ 等）不能用 \"(NH3)2\" 表示"
        "配位，氨必须写 [NH3] 或 N（RDKit 中 NH3 裸写非法）。改法：络离子拆成"
        "可表达的组分（如银氨写 [Ag]([NH3])[NH3] 或 [Ag+]，氨写 N，氢氧化银"
        "写 [Ag+] 与 [OH-] 分离）；实在写不出合法 SMILES 的物种（如复杂配合物）"
        "降级为文字描述或从方程式中省略，只保留能渲染的主物种。",
        "5. 若失败原因是无机盐/含氧酸盐 SMILES 非法（如 KMnO4 写成 K[Mn](=O)(=O)=O"
        "或 KMn(=O)=O——金属与中心原子无直接键）：**最简单改法是直接写教科书化学式**"
        "（KMnO4、H2SO4、MnSO4、K2SO4、Na2CO3、NaCl——渲染为文本，无需 SMILES）；"
        "需要画结构图时才改离子式——阳离子 [K+]/[Na+] 与阴离子用 . 分隔，"
        "含氧酸根中心原子带足双键氧：KMnO4 写 [K+].[O-][Mn](=O)(=O)=O，K2Cr2O7 写 "
        "[K+].[K+].[O-][Cr](=O)(=O)O[Cr]"
        "(=O)(=O)[O-]，KClO3 写 [K+].[O-][Cl](=O)=O，Na2CO3 写"
        "[Na+].[Na+].[O-]C(=O)[O-]，H2SO4 写 OS(=O)(=O)O，HNO3 写"
        "[O-][N+](=O)O。",
        "6. 若失败原因是无效物种/无效 SMILES（如 `-H+`）：`-` 前缀补足只写在箭头条件里"
        "（第 3 段，如 `|-H2O`），**不能写进反应物/产物列表**——列表中的离子直接写"
        "（H+、Br-、[OH-] 或 [H+]、[Br-]），去掉 `-` 前缀并用 `;` 分隔、保证两侧电荷"
        "守恒；无法解析的物种从方程式省略或降级为文字描述。",
    ]
    return "\n".join(lines)


def _apply_patch_corrections(original: str, failures: list,
                             fixed_text: str) -> str | None:
    """把 LLM 部分修正输出（应为一组修正标记）替换进原文对应位置。

    failures: [(RenderTag, reason), ...]（与修正 prompt 顺序一致）。
    返回替换后的完整文本；修正输出解析不出标记、或原文中找不到对应标记
    时返回 None（调用方保留原文，继续下一轮或降级）。
    """
    fixed_tags = parse_tags(fixed_text)
    if not fixed_tags:
        return None
    new_text = original
    for i, (tag, _err) in enumerate(failures):
        if i >= len(fixed_tags):
            break
        new_raw = fixed_tags[i].raw
        if new_raw == tag.raw:
            continue
        if tag.raw not in new_text:
            return None  # 原文位置丢失（不应发生），保守放弃本次修补
        new_text = new_text.replace(tag.raw, new_raw, 1)
    return new_text


def process_question(user_question: str, max_corrections: int = 2,
                     history: list = None, progress_callback=None,
                     correction_callback=None, diagnostics: list = None,
                     responses: list = None) -> str:
    """端到端处理用户问题，返回含渲染后图示代码的文本。

    流程：LLM 生成 → 解析标记 → 契约校验（P1）→ 逐标记渲染 → 注入替换。
    P2 渲染反馈闭环：首次渲染若有失败（校验拦截 / 渲染器失败），携带失败
    清单回传 LLM 自动修正（最多 max_corrections 次），修正版重新走管线；
    仍失败则降级（校验失败标记 → 友好提示，渲染失败标记 → 渲染器错误串）。

    模型路由（配置 UPGRADE_MODEL_NAME 时启用）：主模型（通常 flash）首跑
    **不做修正**，校验失败立即用升级模型（通常 pro）重新生成完整回答
    （升级后可带修正闭环）。未配置则保持"主模型 + 修正闭环"原行为。
    实测依据：flash 对索引/格式类错误修正能救回，对 SMILES 化学构造错误
    （如碳正离子多写碳）修正救不回，而 pro 一遍过率高——难题直接交 pro。

    history: 多轮对话历史（透传给 ask_llm，见 core.llm_client）。
    progress_callback: 可选，LLM 每段生成内容实时回调（B2 流式转发草稿）。
    correction_callback: 可选，P2 修正触发时回调（无参），前端据此提示
        "正在修正回答…"；修正调用强制 thinking=disabled（机械性任务，
        思考链收益小、延迟高）。
    diagnostics: 可选 list，调用方传入后**每一轮校验/渲染失败**（含修正机会
        耗尽前的最后一轮）都会 append 诊断 dict：
        {round, stage, type, raw, reason, friendly, resolved}——前端只展示
        friendly（已注入回答），后端用 reason/resolved/stage 做日志与质量分析。
    responses: 可选 list，调用方传入后记录**各阶段最终采用的原始 LLM 输出**
        （未注入渲染的标记文本；路由下可能 2 条：主模型 + 升级模型）。渲染后
        的 TikZ 无法反推模型写的标记，此字段用于质量回溯（如图文不符时定位
        模型实际写的 SMILES/序号）。
    """
    upgrade = (settings.llm.upgrade_model_name or "").strip()
    if not upgrade:
        # 未配置路由：主模型 + 修正闭环（原行为）
        return _generate_with_corrections(
            user_question, model=None, max_corrections=max_corrections,
            history=history, progress_callback=progress_callback,
            correction_callback=correction_callback,
            diagnostics=diagnostics, stage="main", responses=responses)

    # 难题预判：命中关键词直接走升级模型（省一次主模型首跑与串行延迟）；
    # 漏判由下方"主模型首跑 + 失败升级"兜底，最坏不劣于不配置关键词。
    keywords = getattr(settings.llm, "upgrade_keywords", None) \
        or _DEFAULT_UPGRADE_KEYWORDS
    if any(k and k in user_question for k in keywords):
        print(f"[process_question] 命中难题关键词，直接使用升级模型 {upgrade}…")
        if correction_callback is not None:
            correction_callback()  # 前端提示"正在修正/优化…"
        return _generate_with_corrections(
            user_question, model=upgrade, max_corrections=max_corrections,
            history=history, progress_callback=progress_callback,
            correction_callback=correction_callback,
            diagnostics=diagnostics, stage="upgrade", responses=responses)

    # flash 首跑 + 失败升级 pro 路由：主模型首跑 max_corrections=0（失败即升级）
    diag1 = []
    text1 = _generate_with_corrections(
        user_question, model=None, max_corrections=0,
        history=history, progress_callback=progress_callback,
        correction_callback=None, diagnostics=diag1, stage="main",
        responses=responses)
    if diagnostics is not None:
        diagnostics.extend(diag1)
    if not diag1:
        return text1  # 主模型一遍过（无失败标记）

    print(f"[process_question] 主模型输出含 {len(diag1)} 个失败标记，"
          f"升级模型 {upgrade} 重新生成…")
    if correction_callback is not None:
        correction_callback()  # 前端提示"正在修正/优化…"
    result = _generate_with_corrections(
        user_question, model=upgrade, max_corrections=max_corrections,
        history=history, progress_callback=progress_callback,
        correction_callback=correction_callback,
        diagnostics=diagnostics, stage="upgrade", responses=responses)
    # 升级阶段最终无未解决失败 → 主模型（flash）阶段的失败视为被升级解决
    if diagnostics is not None:
        upgrade_unresolved = any(
            d.get("stage") == "upgrade" and d.get("resolved") is False
            for d in diagnostics)
        if not upgrade_unresolved:
            for d in diagnostics:
                if d.get("stage") == "main":
                    d["resolved"] = True
    return result


def _generate_with_corrections(user_question: str, model=None,
                               max_corrections: int = 2, history: list = None,
                               progress_callback=None, correction_callback=None,
                               diagnostics: list = None,
                               stage: str = "main",
                               responses: list = None) -> str:
    """单模型生成 + P2 修正闭环（PubChem 增强 → LLM → 校验 → 渲染 → 注入）。

    model: 覆盖 ask_llm 的模型名（None=配置默认）；stage: diagnostics 的阶段
    标识（"main"=主模型、"upgrade"=升级模型）。max_corrections=0 时不做修正
    （校验失败即返回，供路由首跑使用）。responses: 可选 list，最终采用的
    原始 LLM 输出（标记文本）append 到此（渲染前版本，供质量回溯）。
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
                            on_piece=progress_callback, model=model)
    if not full_response:
        return "（LLM 调用失败，请检查 .env 配置与网络）"

    for attempt in range(max_corrections + 1):
        # 2. 解析标记
        tags = parse_tags(full_response)
        if not tags:
            if responses is not None:
                responses.append(full_response)  # 纯文本回答（无标记）
            return full_response  # 纯文本回答，无需渲染

        # 2.5 标记契约校验（P1）：渲染前拦截坏参数（非法 SMILES / 越界引用 /
        #    超长 label / 格式错误），降级为友好提示，坏参数不进渲染器
        valid_tags, invalid = validate_tags(tags)
        degraded = {r.tag.raw: degrade_text_friendly(r.tag) for r in invalid}

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
        #     → 回传 LLM 部分修正（只重写失败标记，不重输出全文）；同时把
        #     每一轮失败记入 diagnostics（含最后一轮——供后端日志/质量分析，
        #     前端只展示注入的友好降级文本）
        problems = [(r.tag, r.reason) for r in invalid] + failures
        if diagnostics is not None:
            for tag, err in problems:
                diagnostics.append({
                    "round": attempt,
                    "stage": stage,
                    "type": tag.type,
                    "raw": tag.raw,
                    "reason": err,
                    "friendly": degrade_text_friendly(tag),
                    "resolved": None,  # 循环后统一填充
                })
        if problems:
            # 每轮失败都打印（含最后一轮，供后端诊断）
            round_info = (f"第 {attempt + 1} 轮校验" if attempt == 0
                          else f"修正后第 {attempt + 1} 轮校验")
            print(f"[process_question] {len(problems)} 个标记未通过校验/渲染"
                  f"（{round_info}"
                  + ("，回传 LLM 修正"
                     if attempt < max_corrections
                     else "，修正机会耗尽，降级处理") + "）：")
            for tag, err in problems[:5]:
                print(f"  - {tag.raw[:60]} → {err[:80]}")
        if problems and attempt < max_corrections:
            if correction_callback is not None:
                correction_callback()
            correction = _build_correction_prompt(
                user_question, full_response, problems)
            fixed = ask_llm(correction, on_piece=progress_callback,
                            thinking="disabled", model=model)
            if fixed:
                # 部分修正：用模型输出的修正标记替换原文对应位置
                patched = _apply_patch_corrections(
                    full_response, problems, fixed)
                if patched is not None:
                    full_response = patched
                    continue

        final_ok = not problems  # 修正救回（最终无失败）或从未失败
        # 4. 注入：校验失败 → 友好降级提示；渲染失败（重试机会耗尽）→ 渲染器错误串
        rendered.update(degraded)
        for tag, err in failures:
            rendered.setdefault(tag.raw, err)
        if diagnostics is not None:
            for d in diagnostics:
                if d["resolved"] is None:
                    d["resolved"] = final_ok
        if responses is not None:
            responses.append(full_response)  # 最终采用的原始标记文本
        return inject_tags_into_text(full_response, tags, rendered)

    return "（LLM 调用失败，请检查 .env 配置与网络）"


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "请画出苯的结构式，并说明它的分子式"
    print("=" * 60)
    print(f"用户问题：{question}")
    print("=" * 60)
    diag = []
    print(process_question(question, diagnostics=diag))
    if diag:
        print("\n[诊断] 未渲染标记（含修正后仍失败的最后一轮）：")
        for d in diag:
            status = "✓已修正救回" if d["resolved"] else "✗未解决"
            print(f"  round {d['round']} {status}："
                  f"{d['raw'][:50]} → {d['reason'][:100]}")
