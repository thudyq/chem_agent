# -*- coding: utf-8 -*-
"""app.py — 主入口：LLM → 标记解析 → 校验 → 渲染 → 注入 的端到端管线。

process_question(user_question) 是核心编排函数，FastAPI 适配层
（api.py）在此基础上包装 /chat/completions 端点。
"""

import re

from core.config import settings
from core.llm_client import ask_llm
from core.prompt_manager import load_mech_arrow_prompt, load_struct_rewrite_prompt
from core.tag_parser import parse_tags
from core.tag_injector import inject_tags_into_text
from core.tag_validator import (
    autofix_mech_bond_endpoint, build_component_atom_maps,
    degrade_text_friendly, iter_struct_components, validate_tags,
)
from renderers.registry import RENDERER_REGISTRY, render_tag
# 渲染器失败串的统一前缀（各渲染器内部约定："（XX渲染失败：原因）"）
_RENDER_ERROR_PREFIX = "（"

# 失败 fingerprint 数字归一化（P3 逃生：同批错误跨轮比对）
_FP_DIGITS_RE = re.compile(r"\d+")

# 非化合物名的角色/流程 label（过滤：不作为 PubChem 查询依据）
_ROLE_LABELS = {
    "反应物", "产物", "中间体", "底物", "亲核试剂", "亲电试剂",
    "过渡态", "离去基团", "溶剂", "催化剂", "加成产物", "σ络合物",
    "σ 络合物", "氧负离子", "碳正离子", "自由基",
    # 扩充：常见角色/流程词（中英文）
    "原料", "生成物", "副产物", "主产物", "目标产物", "氧化产物",
    "还原产物", "脱水产物", "氧化剂", "还原剂", "酸碱", "配体",
    "反应中间体", "反应物A", "产物B", "异构体", "对映体",
    "非对映体", "过渡金属", "官能团", "取代基", "盐", "溶剂分子",
    "product", "reactant", "substrate", "reagent", "catalyst",
    "solvent", "intermediate", "nucleophile", "electrophile",
    "radical", "byproduct", "transition state", "starting material",
    "leaving group", "oxidant", "reductant", "ligand",
    "final product", "main product",
}
# 包含匹配的角色词（label 含任一即视为角色/流程词——"主产物/自由基
# 中间体/亲核试剂/氧化产物"等组合，避免无意义的翻译 + PubChem 查询）
_ROLE_SUBSTRINGS = (
    "产物", "中间体", "过渡态", "试剂", "催化剂", "溶剂", "自由基",
    "底物", "亲核", "亲电", "离去基团", "氧化剂", "还原剂", "配体",
    "原料", "生成物", "加成产物", "副产物", "主产物", "异构体",
    "对映体", "非对映体", "络合物",
)


def _is_role_label(label: str) -> bool:
    """label 是否为角色/流程词（精确或包含），是则不做 PubChem 查询。"""
    return bool(label) and (
        label in _ROLE_LABELS or any(s in label for s in _ROLE_SUBSTRINGS))

# 难题预判关键词（未配置 UPGRADE_KEYWORDS 时的内置默认）：
# 用户问题命中任一关键词 → 跳过主模型首跑、直接走升级模型。
# 覆盖机理/箭头/命名反应/电子流动类提问；漏判由 flash 首跑 + 失败升级兜底，
# 误判（简单题命中）代价仅为一次升级调用。
_DEFAULT_UPGRADE_KEYWORDS = (
    "机理", "箭头", "SN1", "SN2", "SN1/SN2", "自由基", "共振", "势能面",
    "电子转移", "电子推动", "消除反应", "亲核取代", "亲电取代", "亲核加成",
    "亲电加成", "加成反应", "重排", "去质子", "质子化", "催化循环",
    "链式反应", "过渡态", "反应历程", "轨道", "HOMO", "LUMO", "构象", "构型",
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
        if not label or _is_role_label(label):
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
            if not _is_role_label(tok) and not any(
                    k in tok for k in ("反应", "机理", "方程", "氧化数")):
                names.append(tok)
    return names[:3]


def _run_with_timeout(fn, timeout: float, default=None):
    """在独立守护线程运行 fn，超时返回 default（线程继续跑，结果丢弃）。

    用于 PubChem 兜底/增强等"锦上添花"的网络调用：PubChem 在部分网络环境
    慢/被限流（503 重试可卡 90s+），若同步执行会拖住整个回答流程——清小搭
    端表现为"正在思考"长时间无进展。超时放弃后主流程照常走 LLM。
    """
    import threading
    box = {}

    def _target():
        try:
            box["v"] = fn()
        except Exception:
            box["v"] = default

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return default
    return box.get("v", default)


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


# PubChem 兜底失败缓存：label 全链路（翻译 → 查询）失败后不再重复尝试
# （修正循环多轮处理同一失败标记时，避免反复触发 flash 翻译 + PubChem 查询；
# 成功查询不缓存，照常重查）。name_resolver 层另有网络负缓存兜底。
_PUBCHEM_FAIL_CACHE: set = set()


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
        if label in _PUBCHEM_FAIL_CACHE:
            continue  # 该 label 此前全链路失败，不再重复翻译/查询
        en = _translate_name_zh2en(label)
        if not en:
            _PUBCHEM_FAIL_CACHE.add(label)
            continue
        try:
            smi = name_to_smiles(en)
        except Exception:
            _PUBCHEM_FAIL_CACHE.add(label)
            continue
        if smi:
            refs.append(f"「{label}」的 PubChem 标准 SMILES：`{smi}`")
        else:
            _PUBCHEM_FAIL_CACHE.add(label)
    if not refs:
        return ""
    return ("\nPubChem 参考（权威 SMILES，可对照修正你的标记）：\n"
            + "\n".join(f"- {r}" for r in refs))

def _context_around(text: str, tag) -> str:
    """标记在原文中的上下文（前后各 ~40 字符），帮助部分修正时理解语境。"""
    start = getattr(tag, "start_pos", None)
    if start is None:
        return ""
    before = text[max(0, start - 40): start].replace("\n", " ")
    after = text[start + len(tag.raw): start + len(tag.raw) + 40].replace("\n", " ")
    ctx = f"{before} ⟦此处⟧ {after}".strip()
    return ctx[:160] if ctx else ""


def _failure_class(err: str) -> str:
    """失败原因分类（修正 prompt 按类注入规则 + 逐条标注）。"""
    if "MECHARROW" in err:
        return "mech"
    if "不守恒" in err:
        return "balance"
    if "无效 SMILES" in err or "无效物种" in err:
        return "smiles"
    return "other"


_CLASS_LABELS = {"mech": "机理箭头", "balance": "守恒",
                 "smiles": "SMILES", "other": "其他"}


# 手术式箭头重写（20260827 两阶段回放实验驱动）：纯机理箭头类失败
# （编号/方向/配对）时，给 LLM"固定骨架 + 原子编号地图"单独补写箭头——
# 查表代替数编号（实验 5 个箭头类案例 15/15 通过）；STRUCT 化学/守恒级
# 错误骨架不可信，地图无意义（对照组实测），仍走原全量部分修正。
_ARROW_FIXABLE_HINTS = ("MECHARROW", "箭头", "进攻位点", "鱼钩", "成键空白")
# 拦截词须精确："SMILES" 单词会误伤端点错误消息（其指引文本含"在 SMILES
# 中把该 H 写成显式 [H]"）——用完整短语「无效 SMILES」/「无效物种」
_ARROW_FIXABLE_BLOCKERS = ("化学校验", "无效 SMILES", "无效物种", "label")

_MECHARROW_TOKEN_RE = re.compile(r"[ \t]*\[MECHARROW[^\]]*\]")


def _has_mecharrow(children) -> bool:
    """子标记列表（含 BLOCK 共振块嵌套）中是否存在 MECHARROW。"""
    if not isinstance(children, list):
        return False
    for c in children:
        if c.type == "MECHARROW":
            return True
        if c.type == "BLOCK" and c.args and _has_mecharrow(c.args[0]):
            return True
    return False


def _is_arrow_fixable(tag, err: str) -> bool:
    """失败是否纯机理箭头类（可用原子地图手术式重写）。

    排除 STRUCT/守恒/label 级错误（骨架不可信）；要求容器内确有
    MECHARROW（缺配对的拦截也至少有一根已写出的箭头）。
    """
    if tag.type != "COMPOSITE":
        return False
    e = err or ""
    if any(k in e for k in _ARROW_FIXABLE_BLOCKERS):
        return False
    if not any(k in e for k in _ARROW_FIXABLE_HINTS):
        return False
    return len(tag.args) >= 2 and _has_mecharrow(tag.args[1])


def _try_diff_autofix(tag):
    """图 diff 反推箭头（确定性，免 LLM）；重校验通过才返回新原文。"""
    from core.electron_sim import fix_arrows_by_diff
    fix = fix_arrows_by_diff(tag)
    if fix is None:
        return None
    _, bad = validate_tags(parse_tags(fix[0]))
    if bad:
        return None
    print(f"[process_question] {fix[1]}")
    return fix[0]


def _try_product_flip(tag):
    """错侧翻转：产物改为电子流模拟推得结构；重校验通过才返回新原文。"""
    from core.electron_sim import flip_products_to_simulated
    flip = flip_products_to_simulated(tag)
    if flip is None:
        return None
    _, bad = validate_tags(parse_tags(flip[0]))
    if bad:
        return None
    print(f"[process_question] {flip[1]}")
    return flip[0]


def _rewrite_composite_arrows(user_question: str, full_text: str, tag,
                              err: str = "", model=None,
                              on_piece=None) -> str | None:
    """手术式箭头重写：骨架 + 原子地图 + 上文叙述 → LLM 只补写 MECHARROW。

    返回通过完整校验的新 COMPOSITE 原文；LLM 调用失败、输出无合法
    COMPOSITE、未补箭头或重写块校验不过均返回 None（调用方回退常规
    修正路径）。专用系统提示见 prompts/mech_arrow_prompt.txt（每根
    箭头独立标记、碱夺 H 终点写 H 原子序号、无显式 H 先改写 STRUCT
    ——均为 20260827 回放实验实测教训）。

    err：本轮校验失败原因。电子流模拟类失败（含"电子流模拟"）时把模拟
    结论注入 prompt（P2）——模拟消息不含错误箭头原文（不锚定），且
    携带推得的实际结构/超价定位，是最强的重写引导。
    """
    maps = build_component_atom_maps(tag)
    if not maps:
        return None
    skeleton = _MECHARROW_TOKEN_RE.sub("", tag.raw)
    skeleton = re.sub(r"\n\s*\n", "\n", skeleton).strip()
    parts = []
    if user_question:
        parts.append(f"原始用户问题：{user_question}")
    start = getattr(tag, "start_pos", None)
    if start is not None:
        context = full_text[max(0, start - 300):start].strip()
        if context:
            parts.append(f"该图在回答中的上文叙述：\n{context}")
    if err and "电子流模拟" in err:
        parts.append(f"上次这组箭头的电子流模拟结论（针对性避免）：\n{err}")
    parts.append(f"骨架（照抄，仅补写 MECHARROW 行）：\n{skeleton}")
    parts.append(f"组件原子编号地图：\n{maps}")
    out = ask_llm("\n\n".join(parts),
                  system_prompt=load_mech_arrow_prompt(),
                  model=model, thinking="disabled", on_piece=on_piece)
    if not out:
        return None
    candidates = [t for t in parse_tags(out) if t.type == "COMPOSITE"]
    if not candidates or "[MECHARROW" not in candidates[0].raw:
        return None
    _, bad = validate_tags(candidates[:1])
    if bad:
        return None
    return candidates[0].raw


# 手术式结构重写（20260828，与箭头重写同一思想）：SMILES/label 级错误走
# "去锚定微任务"——只给名称/约束/上下文、不给错误答案（回放实验：带错误
# 答案的修正反复振荡，fresh 生成 15/15）。守恒等多物种错误不在此列
# （那不是单个 SMILES 写错，是物种取舍问题）。
_STRUCT_FIXABLE_HINTS = ("无效 SMILES", "与 SMILES 不一致",
                         "label 标注", "label 含")
_STRUCT_COMPONENT_ERR_RE = re.compile(r"^组件 ([^:：]+)[:：]\s*(.*)$",
                                      re.DOTALL)
_COEFF_PREFIX_RE = re.compile(r"^(\d+(?:/\d+)?)(?=[A-Za-z\[])")


def _is_struct_fixable(tag, err: str) -> bool:
    """失败是否单个物种的 SMILES/label 级错误（可用微任务重写）。

    顶层 STRUCT 直接看原因；COMPOSITE 则要求错误定位到具体组件
    （"组件 cid: …"格式）且该组件的问题是 SMILES/label 类。
    """
    e = err or ""
    if tag.type == "STRUCT":
        return any(k in e for k in _STRUCT_FIXABLE_HINTS)
    if tag.type == "COMPOSITE":
        m = _STRUCT_COMPONENT_ERR_RE.match(e)
        return bool(m) and any(k in m.group(2) for k in _STRUCT_FIXABLE_HINTS)
    return False


def _rewrite_struct_smiles(user_question: str, full_text: str, tag, err: str,
                           model=None, on_piece=None) -> str | None:
    """手术式结构重写：为写错的物种重新生成 [STRUCT] 标记。

    微任务只给 名称/label + 错误约束 + 上文叙述——**不含错误 SMILES**
    （「…」引用一律抹除，防止锚定）。重写产物保留原组件的 id/mode/系数
    等全部属性（组件可能被 MECHARROW/HBOND 引用），label 以 LLM 新给的
    为准（处理"SMILES 对、label 写错"的镜像病例）。重写后整个标记（顶层
    STRUCT 或 COMPOSITE）须通过完整校验才采纳，否则返回 None 回退常规
    修正路径。
    """
    if tag.type == "STRUCT":
        cid, child = None, tag
    elif tag.type == "COMPOSITE" and len(tag.args) >= 2:
        m = _STRUCT_COMPONENT_ERR_RE.match(err or "")
        if not m:
            return None
        cid = m.group(1).strip()
        child = next((ch for c, ch in iter_struct_components(tag.args[1])
                      if c == cid), None)
        if child is None:
            return None
    else:
        return None
    label = child.args[1] if len(child.args) > 1 else None
    # 错误原因中的「…」引用的是错误 SMILES——去锚定，不展示给 LLM
    reason = re.sub(r"「[^」]*」", "「…」", err or "")
    parts = [f"原始用户问题：{user_question}"]
    start = getattr(tag, "start_pos", None)
    if start is not None:
        context = full_text[max(0, start - 300):start].strip()
        if context:
            parts.append(f"该图在回答中的上文叙述：\n{context}")
    parts.append(f"需要重写的物种名称/标签：{label or cid or '（未标注）'}")
    parts.append(f"上次写错的原因（不要重犯同样的错误）：\n{reason}")
    out = ask_llm("\n\n".join(parts),
                  system_prompt=load_struct_rewrite_prompt(),
                  model=model, thinking="disabled", on_piece=on_piece)
    if not out:
        return None
    cands = [t for t in parse_tags(out) if t.type == "STRUCT"]
    if not cands or not cands[0].args or not cands[0].args[0]:
        return None
    new_smi = cands[0].args[0].strip()
    new_label = (cands[0].args[1] if len(cands[0].args) > 1 else None)
    # 保留原写法（系数/属性/label 位置）：只替换 SMILES 主体
    old_smi = (child.args[0] or "").strip()
    coeff = ""
    m = _COEFF_PREFIX_RE.match(old_smi)
    if m:
        coeff = m.group(1)
        old_smi = old_smi[m.end():]
    new_raw = child.raw.replace(old_smi, coeff + new_smi, 1)
    if new_label and label and new_label != str(label):
        new_raw = new_raw.replace(f"label={label}", f"label={new_label}", 1)
    if tag.type == "COMPOSITE":
        new_raw = tag.raw.replace(child.raw, new_raw, 1)
    _, bad = validate_tags(parse_tags(new_raw))
    if bad:
        return None
    return new_raw


def _build_correction_prompt(user_question: str, original: str,
                             failures: list) -> str:
    """构造 P2 修正 prompt：失败标记清单（含上下文）+ 修正要求。

    部分修正模式：模型**只输出修正后的标记**（不重输出整个回答），
    process_question 用修正标记替换原文对应位置后重新校验/渲染——
    相比"全篇重生成"大幅节省 token 且修正更聚焦。
    修正要求按失败类型动态裁剪（20260821 P2），风格对齐 system_prompt：
    简洁、明确、无与本次失败无关的语句。

    failures: [(RenderTag, 失败原因字符串), ...]。
    """
    classes = {_failure_class(err) for _, err in failures}
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
        lines.append(f"- 标记 {i}（{_CLASS_LABELS[_failure_class(err)]}）："
                     f"{tag.raw}")
        lines.append(f"  原因：{err}")
        ctx = _context_around(original, tag)
        if ctx:
            lines.append(f"  上下文：{ctx}")
        # COMPOSITE 失败附组件原子编号地图：修正端点引用时照表查，
        # 不让模型自己数编号（20260827 回放实验：编号错误占失败大头）
        if tag.type == "COMPOSITE":
            maps = build_component_atom_maps(tag)
            if maps:
                lines.append("  原子编号地图（引用端点照此查表，不要自己数）：")
                lines.append(maps)

    # PubChem 兜底：失败标记的 label 是化合物名时，反查权威 SMILES 作为修正参考
    # （12s 硬超时：PubChem 慢/限流时放弃，不拖住修正主流程）
    pubchem_ref = _run_with_timeout(
        lambda: _fetch_pubchem_references(failures, user_question), 12.0, "")
    if pubchem_ref:
        lines.append("")
        lines.append(pubchem_ref)

    reqs = [
        "**只输出修正后的标记本身**（保持 [TYPE:...] 语法；COMPOSITE 容器要完整，"
        "含开闭标签），多个失败标记按上面顺序依次输出；不要输出解释、序号或任何其他文字。",
        "不要新增或删除其他标记；修正后的标记必须严格遵循标记语法。",
    ]
    if "mech" in classes:
        reqs.append(
            "机理箭头修正：按原因中给出的建议端点直接改写（「应直接改写为"
            "「x-y」」），不要自己猜编号；提示无显式 H 时，先把该 H 写成显式 "
            "[H]（参与编号）再按原子地图引用。")
        if any("电子流模拟" in err for _, err in failures):
            reqs.append(
                "电子流模拟不一致：声明产物本身已通过守恒与价态检查，优先"
                "重画机理箭头使其能从反应物推出声明产物；原因中含模拟推得"
                "的实际结构（预期产物），若确认箭头无误，则把产物改为该结构。")
    if "balance" in classes:
        reqs.append(
            "守恒修正：按给出的元素/电荷差值补物种或调系数（2CCO、1/2O2）；"
            "辅助试剂写入箭头条件（H2O 补左侧、-H2O 补右侧）；氧化剂被还原时"
            "必须写完整配平方程式（不能用箭头补足省略，氧化反应不补 -H2）；"
            "禁用 [O]/[H] 占位符；核对每个物种的分子式与其角色（底物/中间体/"
            "产物的碳数一致）；用户要配平方程式时必须守恒，只要转化示意时可改"
            "单→单 [ARROW]（不查全元素守恒）。")
    if "smiles" in classes:
        reqs.append(
            "无效 SMILES 修正：无机物/配离子直接写分子式（KMnO4、"
            "[Ag(NH3)2]+，配体括号写全）；金属盐用 . 分隔阴阳离子"
            "（[K+].[O-][Mn](=O)(=O)=O）；含氧酸根中心原子带足双键氧"
            "（H2SO4=OS(=O)(=O)O、HNO3=[O-][N+](=O)O、硝基=R[N+](=O)[O-]）；"
            "氨写 [NH3] 或 N；写不出的物种省略或文字描述。")
        reqs.append(
            "「-」前缀只用于箭头条件（|-H2O）；反应物/产物列表中的离子直接写"
            "（[H+]、[Br-]、[OH-]）。")
    lines.append("")
    lines.append("修正要求：")
    lines += [f"{i}. {r}" for i, r in enumerate(reqs, 1)]
    return "\n".join(lines)


def _tag_identity_tokens(raw: str) -> set:
    """标记身份令牌（id=/label= 值集合）——用于判断"修正后的标记"是否与
    原标记是同一个东西。空集合 = 无法判断（调用方保守放行）。"""
    return set(re.findall(r"(?:id|label)=([^,\]]+)", raw))


def _tag_inventory(text: str) -> dict:
    """文本中标记的类型计数（G2 内容完整性对账用）。
    REASONING 不计——注入时本就剥离，不属用户可见内容。"""
    counts = {}
    for t in parse_tags(text):
        if t.type == "REASONING":
            continue
        counts[t.type] = counts.get(t.type, 0) + 1
    return counts


def _apply_patch_corrections(original: str, failures: list,
                             fixed_text: str) -> str | None:
    """把 LLM 部分修正输出（应为一组修正标记）替换进原文对应位置。

    failures: [(RenderTag, reason), ...]（与修正 prompt 顺序一致）。
    返回替换后的完整文本；修正输出解析不出标记、或原文中找不到对应标记
    时返回 None（调用方保留原文，继续下一轮或降级）。

    G2 身份闸门（20260906，Q10 病例）：修正标记与原标记**类型不同**、或
    身份令牌（id/label）**零交集**时拒绝该处替换——LLM 答非所问（重写
    了别的图）会把题目要求的内容替换没（"删内容保合法"）；拒绝替换后原
    标记留在原文走降级，内容缺失对用户可见。
    """
    fixed_tags = parse_tags(fixed_text)
    if not fixed_tags:
        return None
    new_text = original
    for i, (tag, _err) in enumerate(failures):
        if i >= len(fixed_tags):
            break
        new_tag = fixed_tags[i]
        new_raw = new_tag.raw
        if new_raw == tag.raw:
            continue
        if new_tag.type != tag.type:
            continue  # 类型不同——不是同一个图的修正，拒绝替换
        old_ids = _tag_identity_tokens(tag.raw)
        new_ids = _tag_identity_tokens(new_raw)
        if old_ids and new_ids and not (old_ids & new_ids):
            continue  # 身份零交集——替换等于删除原图，拒绝
        if tag.raw not in new_text:
            return None  # 原文位置丢失（不应发生），保守放弃本次修补
        new_text = new_text.replace(tag.raw, new_raw, 1)
    return new_text


def _partial_render_composite_without_mecharrows(tag) -> str | None:
    """COMPOSITE 仅 MECHARROW 报错（修正机会耗尽）时：剔除全部 MECHARROW
    子标记后重新渲染——分子/加号/主箭头保留（反应骨架完整，仅缺机理弯
    箭头）。返回 TikZ 代码；仍失败返回 None（调用方走整体降级）。

    依据：MECHARROW 错误（索引/引用/格式）不影响分子与反应骨架的正确性，
    整图降级太可惜；剔除箭头后"反应物→产物"仍完整可读。
    """
    if tag.type != "COMPOSITE" or not tag.args or len(tag.args) < 2:
        return None
    children = tag.args[1]
    if not isinstance(children, list) or \
            not any(c.type == "MECHARROW" for c in children):
        return None
    new_children = [c for c in children if c.type != "MECHARROW"]
    if not new_children:
        return None
    new_args = [tag.args[0], new_children] + list(tag.args[2:])
    renderer = RENDERER_REGISTRY.get(tag.type)
    if renderer is None:
        return None
    try:
        out = renderer(*new_args)
    except Exception:
        return None
    if not isinstance(out, str) or out.startswith(_RENDER_ERROR_PREFIX):
        return None  # 剔除箭头后仍有其他错误（渲染器失败串）→ 整体降级
    return out


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
          f"升级模型 {upgrade} 部分修正（不重跑全文）…")
    if correction_callback is not None:
        correction_callback()  # 前端提示"正在修正/优化…"
    result = _generate_with_corrections(
        user_question, model=upgrade, max_corrections=max_corrections,
        history=history, progress_callback=progress_callback,
        correction_callback=correction_callback,
        diagnostics=diagnostics, stage="upgrade", responses=responses,
        seed_text=text1)
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
                               responses: list = None,
                               seed_text: str = None) -> str:
    """单模型生成 + P2 修正闭环（PubChem 增强 → LLM → 校验 → 渲染 → 注入）。

    model: 覆盖 ask_llm 的模型名（None=配置默认）；stage: diagnostics 的阶段
    标识（"main"=主模型、"upgrade"=升级模型）。max_corrections=0 时不做修正
    （校验失败即返回原始标记文本，供路由升级部分修正）。responses: 可选 list，
    最终采用的原始 LLM 输出（标记文本）append 到此（渲染前版本，供质量回溯）。
    seed_text: 非 None 时跳过 LLM 主生成，直接以该文本进入校验/修正循环——
    用于"flash 失败 → 升级模型只做部分修正（不重跑全文）"：对失败标记
    重新校验并让升级模型修正，成本远低于全文重新生成。
    """
    if seed_text is not None:
        full_response = seed_text
    else:
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
                    # 8s 硬超时：PubChem 慢/被限流时跳过增强（不阻塞主流程，
                    # 否则清小搭端表现为"正在思考"长时间无进展）
                    pub_smiles = _run_with_timeout(
                        lambda: name_to_smiles(chem_name), 8.0, None)
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

    prev_fps = None   # 上一轮失败 fingerprint（P3 逃生比对）
    surgical_tried = set()  # 已尝试过手术式箭头重写的标记原文（每标记只试一次）
    baseline_counts = None  # 首跑输出的标记清单（G2 内容完整性基线）
    for attempt in range(max_corrections + 1):
        # 2. 解析标记
        tags = parse_tags(full_response)
        if not tags:
            if responses is not None:
                responses.append(full_response)  # 纯文本回答（无标记）
            return full_response  # 纯文本回答，无需渲染
        if attempt == 0:
            baseline_counts = _tag_inventory(full_response)

        # 2.5 标记契约校验（P1）：渲染前拦截坏参数（非法 SMILES / 越界引用 /
        #    超长 label / 格式错误），降级为友好提示，坏参数不进渲染器
        valid_tags, invalid = validate_tags(tags)
        # 2.6 确定性自动修复（20260821 P1，不经 LLM）：MECHARROW 键端点
        #    "唯一候选"改写——逐标记修复并单标记重校验，全部校验规则
        #    （含化学配对）通过才替换进原文
        if invalid:
            fixed_any = False
            for r in invalid:
                fix = autofix_mech_bond_endpoint(r.tag)
                if fix is None or r.tag.raw not in full_response:
                    continue
                new_raw, note = fix
                _, bad_fix = validate_tags(parse_tags(new_raw))
                if bad_fix:
                    continue
                full_response = full_response.replace(r.tag.raw, new_raw, 1)
                fixed_any = True
                print(f"[process_question] 端点自动修复：{note}")
                if diagnostics is not None:
                    diagnostics.append({
                        "round": attempt,
                        "stage": stage,
                        "type": "AUTOFIX",
                        "raw": new_raw,
                        "reason": f"端点自动修复：{note}",
                        "friendly": "",
                        "resolved": True,
                    })
            if fixed_any:
                tags = parse_tags(full_response)
                valid_tags, invalid = validate_tags(tags)
        degraded = {}
        for r in invalid:
            partial = None
            # 部分降级：COMPOSITE 仅 MECHARROW 报错（修正耗尽）时，剔除全部
            # MECHARROW 子标记后重新渲染——分子/加号/主箭头保留（反应骨架
            # 完整，仅缺机理弯箭头），避免整图省略
            if r.tag.type == "COMPOSITE" and "MECHARROW" in (r.reason or ""):
                partial = _partial_render_composite_without_mecharrows(r.tag)
            if partial is not None:
                degraded[r.tag.raw] = (
                    partial + "\n\n> 反应箭头无法渲染，已省略"
                              "；机理电子流向请以文字说明为准")
            else:
                degraded[r.tag.raw] = degrade_text_friendly(r.tag)

        # 3. 逐标记渲染（REASONING 无渲染器，由注入器特殊处理）
        rendered, failures = {}, []
        for tag in valid_tags:
            if tag.type == "REASONING":
                continue
            try:
                out = render_tag(tag)
            except Exception as e:
                out = f"（{tag.type} 渲染失败：{e}）"
            if out is None:
                continue  # 未注册类型，注入时保留原标记
            if out.startswith(_RENDER_ERROR_PREFIX):
                failures.append((tag, out))
            else:
                rendered[tag.raw] = out

        # 3.5 P2 渲染反馈闭环：有失败（校验拦截或渲染失败）且还有修正机会
        #     → 回传 LLM 部分修正（只重写失败标记，不重输出全文）；同时把
        #     每一轮失败记入 diagnostics（含最后一轮——供后端日志/质量分析，
        #     前端只展示注入的友好降级文本）
        problems = [(r.tag, r.reason) for r in invalid] + failures
        # P3 逃生锚点（20260821）：错误 fingerprint（类型+原因，数字归一化）
        # 与上一轮完全相同 = LLM 修正无进展（同一错误反复犯）——不再消耗
        # 修正轮次，直接走降级
        fps = sorted(f"{t.type}:{_FP_DIGITS_RE.sub('N', e)}"
                     for t, e in problems)
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
            if prev_fps is not None and fps == prev_fps:
                print(f"[process_question] 失败原因与上一轮完全相同，"
                      f"LLM 修正无进展——跳过剩余修正轮次，直接降级")
            else:
                # 手术式重写（20260827 箭头 / 20260828 结构）：逐失败标记
                # 分流——纯机理箭头错走"骨架+原子地图"重写，SMILES/label 级
                # 错走"去锚定微任务"重写；均不依赖错误答案原文。任一成功即
                # 重解析重校验；不在手术范围的（守恒/格式等）或重写失败的
                # 留给下方常规部分修正。每个失败标记只尝试一次手术重写。
                surg_any = False
                for t, e in problems:
                    if t.raw in surgical_tried:
                        continue
                    if _is_arrow_fixable(t, e):
                        kind = "箭头/地图"
                    elif _is_struct_fixable(t, e):
                        kind = "结构/去锚定"
                    else:
                        continue
                    surgical_tried.add(t.raw)
                    if kind == "箭头/地图":
                        # 错侧判定证据序（§16.3）：①确定性 diff 反推（箭头类
                        # 失败 = 产物已过独立检查、可信，免 LLM）→ ②手术重写
                        # （带模拟结论）→ ③仅当失败是"模拟不一致"且修箭头不
                        # 收敛才翻转产物（Q17 类被模式规则拦截的产物是对的，
                        # 不可翻转——翻转门只认模拟口径）
                        new_raw = _try_diff_autofix(t)
                        if new_raw is None:
                            if correction_callback is not None:
                                correction_callback()
                            new_raw = _rewrite_composite_arrows(
                                user_question, full_response, t, e,
                                model=model, on_piece=progress_callback)
                        if new_raw is None and "电子流模拟" in e:
                            new_raw = _try_product_flip(t)
                    else:
                        if correction_callback is not None:
                            correction_callback()
                        new_raw = _rewrite_struct_smiles(
                            user_question, full_response, t, e, model=model,
                            on_piece=progress_callback)
                    if new_raw and new_raw != t.raw and t.raw in full_response:
                        full_response = full_response.replace(t.raw, new_raw, 1)
                        surg_any = True
                        print(f"[process_question] 手术式重写成功（{kind}）")
                if surg_any:
                    prev_fps = fps
                    continue
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
                        prev_fps = fps
                        continue
        prev_fps = fps

        final_ok = not problems  # 修正救回（最终无失败）或从未失败
        if problems and max_corrections == 0:
            # 路由首跑（flash）失败：返回原始标记文本（不降级注入），
            # 供升级模型（pro）基于失败标记做部分修正
            if responses is not None:
                responses.append(full_response)
            return full_response
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
        result_text = inject_tags_into_text(full_response, tags, rendered)
        # G2 内容完整性对账（20260906，Q10 病例）：修正/重写后标记总数少于
        # 首跑输出 = 内容被删（"删内容保合法"）——按未解决记账（路由升级判定
        # 随之视为失败），并在回答末尾显式告知，不允许无声通过
        if baseline_counts:
            missing = []
            final_counts = _tag_inventory(full_response)
            for ttype, n0 in baseline_counts.items():
                n1 = final_counts.get(ttype, 0)
                if n1 < n0:
                    missing.append((ttype, n0, n1))
            if missing:
                notes = []
                for ttype, n0, n1 in missing:
                    notes.append(f"{ttype} 图示（原 {n0} 处，现 {n1} 处）")
                    if diagnostics is not None:
                        diagnostics.append({
                            "round": attempt,
                            "stage": stage,
                            "type": ttype,
                            "raw": "",
                            "reason": (f"修正后内容缺失：{ttype} 标记 "
                                       f"{n0}→{n1}（修正/重写过程中被删除）"),
                            "friendly": f"（{ttype} 图示在修正过程中未能保留，已省略）",
                            "resolved": False,
                        })
                print(f"[process_question] 修正后内容缺失："
                      + "、".join(notes))
                result_text += ("\n\n> 注：修正过程中部分图示未能保留，已省略"
                                "（" + "、".join(notes) + "）——"
                                "需要的话我可以重新绘制。")
        return result_text

    return "（LLM 调用失败，请检查 .env 配置与网络）"


if __name__ == "__main__":
    import sys

    # Windows GBK 控制台打印含 ⁺/⁻ 等字符的失败原因会 UnicodeEncodeError
    # （与 metrics.py/composite.py 同款修复）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
