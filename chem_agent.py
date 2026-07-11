# -*- coding: utf-8 -*-
"""
chem_agent.py
=============
有机化学知识智能体 - 主入口脚本。

Day 1    : validate_mol / get_molecular_formula —— RDKit 基础与分子式计算。
Day 7-8  : ask_llm —— 接入 OpenAI 兼容大模型 API（DeepSeek / SiliconFlow），
           temperature=0.2、重试 3 次、密钥从 .env 读取。
Day 11-12: main_process —— 串联名称解析/知识库/结构渲染/LLM，RAG 主控流程。
"""

import os
import re
import time
from pathlib import Path

import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from utils.name_resolver import name_to_smiles
from utils.structure_render import smiles_to_tikz
from utils.db_helper import query_property
from utils.reaction_render import is_reaction_input, parse_reaction, render_reaction_chemfig

# === LLM 默认参数 ===
DEFAULT_TEMPERATURE = 0.2   # 低温度保证事实性
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 60        # 单次请求超时（秒）
DEFAULT_RETRIES = 3         # 失败重试次数


# --------------------------------------------------------------------------- #
# Day 1：RDKit 基础
# --------------------------------------------------------------------------- #
def validate_mol(smiles: str):
    """校验 SMILES 字符串是否合法。

    返回:
        rdkit.Chem.rdchem.Mol | None：合法返回 Mol 对象，否则 None。
    """
    print(f"[validate_mol] 正在校验 SMILES: {smiles!r}")
    if not smiles or not isinstance(smiles, str):
        print("[validate_mol] 输入为空或类型错误，校验失败。")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[validate_mol] 解析失败：'{smiles}' 不是合法的 SMILES。")
        return None

    print(f"[validate_mol] 校验通过，共解析到 {mol.GetNumAtoms()} 个重原子。")
    return mol


def get_molecular_formula(mol) -> str:
    """根据 RDKit Mol 对象计算分子式（如 'C4H8'）；mol 为 None 时返回空串。"""
    if mol is None:
        print("[get_molecular_formula] 输入的 Mol 对象为空，无法计算分子式。")
        return ""
    formula = CalcMolFormula(mol)
    mol_weight = Descriptors.MolWt(mol)
    print(f"[get_molecular_formula] 分子式={formula}，分子量≈{mol_weight:.2f}")
    return formula


# --------------------------------------------------------------------------- #
# Day 7-8：大模型 API
# --------------------------------------------------------------------------- #
def _load_env() -> None:
    """把 .env 加载进 os.environ。

    优先用 python-dotenv；未安装时用内置最小解析器兜底，保证无该依赖也能读密钥。
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    env_path = Path(".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip().strip("'").strip('"')
        os.environ.setdefault(key.strip(), val)


def get_llm_config():
    """读取并校验 LLM 配置。

    返回:
        tuple | None：(api_key, base_url, model)；任一项缺失返回 None。
    """
    _load_env()
    api_key = os.environ.get("API_KEY", "").strip()
    base_url = os.environ.get("BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("MODEL", "").strip()
    if not (api_key and base_url and model):
        return None
    return api_key, base_url, model


def ask_llm(
    prompt: str,
    *,
    system_prompt: str = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = DEFAULT_RETRIES,
):
    """调用 OpenAI 兼容大模型 API（DeepSeek / SiliconFlow），返回回答文本。

    采用低 temperature(0.2) 保证化学事实性；失败自动重试（默认 3 次，线性退避）。
    DeepSeek 与 SiliconFlow 均兼容 POST {BASE_URL}/chat/completions 的 OpenAI 格式。

    参数:
        prompt (str): 用户问题。
        system_prompt (str): 可选系统提示，用于约束模型角色与输出风格。
        temperature (float): 采样温度，默认 0.2。
        max_tokens (int): 最大生成 token 数。
        retries (int): 失败重试次数。

    返回:
        str | None: 模型回答文本；配置缺失或重试耗尽返回 None。
    """
    config = get_llm_config()
    if config is None:
        print("[ask_llm] 未配置 API_KEY/BASE_URL/MODEL，请先创建 .env（参考 .env.example）。")
        return None
    api_key, base_url, model = config

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    print(f"[ask_llm] 调用 {model} @ {base_url}（temperature={temperature}）")
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
        except requests.exceptions.RequestException as e:
            print(f"[ask_llm] 第 {attempt}/{retries} 次请求异常: {e}")
        else:
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                # reasoning 模型（如 deepseek-reasoner）可能把输出放 reasoning_content，content 为空
                content = msg.get("content") or ""
                if not content and msg.get("reasoning_content"):
                    content = msg["reasoning_content"]
                    print("[ask_llm] content 为空，回退 reasoning_content")
                usage = data.get("usage", {})
                print(f"[ask_llm] 成功（tokens: {usage.get('total_tokens', '?')}）")
                return content
            print(f"[ask_llm] 第 {attempt}/{retries} 次失败 HTTP {resp.status_code}: {resp.text[:200]}")

        # 未达最大次数则退避后重试
        if attempt < retries:
            backoff = attempt * 2
            print(f"[ask_llm] {backoff}s 后重试 ...")
            time.sleep(backoff)

    print(f"[ask_llm] {retries} 次重试均失败。")
    return None


# --------------------------------------------------------------------------- #
# Day 11-12：主控流程（RAG 基础）
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = (
    "你是一位严谨的有机化学知识助手。请聚焦回答用户提出的具体问题，有的放矢、深入分析。"
    "回答应条理清晰，复杂问题可分点论述并给出结论。"
    "重要：不要主动罗列熔点、沸点、密度、分子量等基本理化性质——这些已在界面'详细信息'区展示，"
    "除非用户问题直接询问某项性质（如'熔点是多少'）才引用对应数值。"
    "优先基于分子式与 SMILES 做结构、官能团、电子效应、反应性等深入分析。"
    "结构式的 chemfig 代码与图像已由界面单独展示，无需输出任何 LaTeX。"
    "若数据缺失可基于化学知识补充，但需标注为推测。"
)


def _build_prompt(user_input, smiles, props):
    lines = [f"用户问题：{user_input}", ""]

    if props:
        lines.append(f"化合物：{props.get('name', '')}（{props.get('en_name', '')}）")
        if props.get("molecular_formula"):
            lines.append(f"分子式：{props['molecular_formula']}")
    elif smiles:
        lines.append("化合物（在线解析，不在本地知识库）")
    else:
        lines.append("（未能解析出明确化合物，请基于问题本身作答）")
    if smiles:
        lines.append(f"SMILES：{smiles}")
    lines.append("")

    # 性质数据压缩为单行上下文，标注"按需引用"避免 LLM 主动罗列
    if props:
        prop_parts = []
        for key in ("mol_weight", "xlogp", "melting_point", "boiling_point", "density", "cas"):
            val = props.get(key)
            if val not in (None, "", []):
                prop_parts.append(f"{_FIELD_LABELS.get(key, key)}={val}")
        if prop_parts:
            lines.append("可用性质数据（仅在问题直接涉及时引用，勿主动罗列）：")
            lines.append("  " + "；".join(prop_parts))
        lines.append("")

    lines.append("请聚焦回答上面的用户问题，深入分析而非泛泛介绍。")
    return "\n".join(lines)


_FIELD_LABELS = {
    "name": "中文名", "en_name": "英文名", "iupac_name": "IUPAC名",
    "molecular_formula": "分子式", "mol_weight": "分子量", "xlogp": "XLogP",
    "melting_point": "熔点", "boiling_point": "沸点", "density": "密度",
    "cas": "CAS号",
}


# CJK 统一汉字范围，用于判断输入是否含中文
_CN_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

_TRANSLATE_SYSTEM = (
    "你是一名化学命名翻译器。将用户给出的中文化合物名称翻译为英文或 IUPAC 名称，"
    "供 PubChem 检索使用。只输出名称本身，不要解释、标点或多余文字。"
)


def _is_chinese(text: str) -> bool:
    return bool(_CN_CHAR_RE.search(text or ""))


def _clean_llm_line(answer, none_token=None):
    # 剥离 <think> 思考链、取末非空行、去首尾标点；命中 none_token 视为空
    if not answer:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return None
    name = lines[-1].strip("\"'.,;: ()（）")
    if not name or (none_token and name.upper() == none_token.upper()):
        return None
    return name


def _translate_compound_name(cn_name: str):
    prompt = f"中文化合物名称：{cn_name}\n\n请输出对应的英文或 IUPAC 名称。"
    answer = ask_llm(prompt, system_prompt=_TRANSLATE_SYSTEM, max_tokens=256)
    return _clean_llm_line(answer)


_QUESTION_MARKERS = (
    "为什么", "是什么", "是多少", "多少", "怎么", "哪些", "哪种", "吗", "呢",
    "？", "?", "作用", "性质", "结构", "用途", "熔点", "沸点", "密度",
    "酸性", "碱性", "官能团", "极性", "溶解度", "毒性", "合成", "制备",
    "反应", "介绍", "说明", "解释", "比较", "区别",
)

_EXTRACT_SYSTEM = (
    "你是一名化学实体识别器。从用户输入中提取化学化合物名称（中文或英文均可）。"
    "只输出名称本身；若无法识别出化合物名则输出 NONE。"
)

_THERMO_SYSTEM = (
    "你是一名化学热力学顾问。根据反应信息估算热力学参数。"
    "只输出键值对，每行一个，格式严格为 'pK: 值'、'ΔH: 值 kJ/mol'、'ΔS: 值 J/(mol·K)'、'ΔG: 值 kJ/mol'。"
    "无法估算的项输出 '未知'。不要任何解释或多余文字。"
)


def _looks_like_question(text: str) -> bool:
    if not text:
        return False
    if any(m in text for m in _QUESTION_MARKERS):
        return True
    # 无标记时，中文长句兜底判为问题
    return _is_chinese(text) and len(text) > 12


def _extract_compound_name(text: str):
    prompt = (
        f"用户输入：{text}\n\n请提取其中的化学化合物名称（中文或英文均可），"
        f"只输出名称本身。若不含化合物名则输出 NONE。"
    )
    answer = ask_llm(prompt, system_prompt=_EXTRACT_SYSTEM, max_tokens=64)
    return _clean_llm_line(answer, none_token="NONE")


def main_process(user_input: str) -> dict:
    """主控入口：按输入类型分派——reaction SMILES 走反应处理，其余走化合物处理。"""
    if is_reaction_input(user_input):
        return _process_reaction(user_input)
    return _process_compound(user_input)


def _process_compound(user_input: str) -> dict:
    """化合物处理：名称/问题 -> SMILES + 性质 + 结构式 + 中文回答。

    顺序：
        1. 先查本地知识库（支持中/英/IUPAC 名）；命中即拿到 SMILES 与性质。
        2. 库中无 SMILES 时，问题型输入先抽取化合物名重查。
        3. 仍无 SMILES -> PubChem 在线解析（中文名先 LLM 译为英文）。
        4. SMILES -> chemfig 结构式代码。
        5. 组装 RAG prompt 调用 LLM 生成中文回答。

    任何子环节失败均降级处理，返回部分结果，绝不抛异常。

    返回:
        dict: {type, input, is_question, smiles, properties, chemfig, answer}
    """
    print(f"\n[main_process] 输入: {user_input!r}")

    props = query_property(user_input)
    smiles = props.get("smiles") if props else None
    if smiles:
        print(f"[main_process] 知识库命中: {props.get('name')} ({props.get('molecular_formula')})")

    resolved = user_input
    # 库未命中且输入是问题型：先抽取化合物名，再用抽取结果重查知识库
    if not smiles and _looks_like_question(user_input):
        extracted = _extract_compound_name(user_input)
        if extracted and extracted != user_input:
            print(f"[main_process] 实体抽取: {user_input!r} -> {extracted!r}")
            resolved = extracted
            props = query_property(resolved)
            smiles = props.get("smiles") if props else None
            if smiles:
                print(f"[main_process] 知识库命中(抽取后): {props.get('name')}")

    # 仍无 SMILES -> 在线解析。中文名 PubChem 无法识别，先用 LLM 译为英文/IUPAC 名
    if not smiles:
        if _is_chinese(resolved):
            en = _translate_compound_name(resolved)
            if en:
                print(f"[main_process] 中文名翻译: {resolved!r} -> {en!r}")
                resolved = en
        smiles = name_to_smiles(resolved)

    chemfig = smiles_to_tikz(smiles) if smiles else ""

    print("[main_process] 调用 LLM 生成回答 ...")
    prompt = _build_prompt(user_input, smiles, props)
    answer = ask_llm(prompt, system_prompt=_SYSTEM_PROMPT)

    result = {
        "type": "compound",
        "input": user_input,
        "is_question": _looks_like_question(user_input),
        "smiles": smiles,
        "properties": props,
        "chemfig": chemfig,
        "answer": answer,
    }
    print("[main_process] 完成。")
    return result


def _fetch_reaction_thermo(rxn):
    """用 LLM 估算反应的热力学参数（pK/ΔH/ΔS/ΔG）。

    返回:
        dict | None: 解析到的键值对（可能为空 dict 表示全未知）；LLM 调用失败返回 None。
        数值为 LLM 估算，仅供参考。
    """
    prompt = (
        f"反应物 SMILES：{', '.join(rxn['reactants'])}\n"
        f"产物 SMILES：{', '.join(rxn['products'])}\n"
        f"条件：{rxn['conditions'] or '未标注'}\n\n"
        f"请估算该反应的平衡常数相关 pK（即 -log K）、焓变 ΔH、熵变 ΔS、吉布斯自由能变 ΔG。"
    )
    answer = ask_llm(prompt, system_prompt=_THERMO_SYSTEM, max_tokens=512)
    if not answer:
        return None
    # 剥离 reasoning 模型的 <think> 思考链，只解析正式作答部分
    cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    thermo = {}
    for line in cleaned.splitlines():
        # 兼容全角冒号 ：
        if "：" in line:
            line = line.replace("：", ":", 1)
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        # 去 markdown 粗体/斜体标记
        key = key.strip().strip("*_` ")
        val = val.strip().strip("*_` ")
        if val and val != "未知":
            thermo[key] = val
    if not thermo and cleaned:
        print(f"[_fetch_reaction_thermo] 未解析到键值对，原始返回: {cleaned[:300]!r}")
    return thermo


def _process_reaction(user_input: str) -> dict:
    """反应处理：解析 reaction SMILES，渲染方程式，LLM 分析反应。

    返回:
        dict: {type, input, reactants, products, conditions, reversible, equation_chemfig, answer}
    """
    print(f"\n[main_process] 检测到反应输入: {user_input!r}")
    rxn = parse_reaction(user_input)
    if rxn is None:
        # 解析失败，退化为普通化合物处理
        print("[main_process] 反应解析失败，退化为化合物处理")
        return _process_compound(user_input)

    print(f"[main_process] 反应物: {rxn['reactants']}")
    print(f"[main_process] 产物: {rxn['products']}")

    equation_chemfig = render_reaction_chemfig(rxn)

    r_list = ", ".join(rxn["reactants"])
    p_list = ", ".join(rxn["products"])
    cond = rxn["conditions"] or "未标注"
    prompt = (
        f"用户输入了一个化学反应（reaction SMILES）：{user_input}\n\n"
        f"反应物 SMILES：{r_list}\n"
        f"产物 SMILES：{p_list}\n"
        f"反应条件：{cond}\n"
        f"反应方向：{'可逆' if rxn['reversible'] else '正向'}\n\n"
        f"请分析这个反应：判断反应类型、说明关键结构变化、概述机理要点、提示注意事项。条理清晰，分点论述。"
    )
    print("[main_process] 调用 LLM 分析反应 ...")
    answer = ask_llm(prompt, system_prompt=_SYSTEM_PROMPT)

    print("[main_process] 估算热力学参数 ...")
    thermo = _fetch_reaction_thermo(rxn)

    return {
        "type": "reaction",
        "input": user_input,
        "reactants": rxn["reactants"],
        "products": rxn["products"],
        "conditions": rxn["conditions"],
        "reversible": rxn["reversible"],
        "equation_chemfig": equation_chemfig,
        "thermo": thermo,
        "answer": answer,
    }


# --------------------------------------------------------------------------- #
# 测试入口
# --------------------------------------------------------------------------- #
def _day1_smoke():
    """Day 1 冒烟测试：验证硬编码 'C1CCC1'（环丁烷）并打印分子式。"""
    print("=" * 60)
    print("Day 1 - RDKit 基础测试")
    print("=" * 60)
    test_smiles = "C1CCC1"
    mol = validate_mol(test_smiles)
    if mol is not None:
        print(f"\n[结果] SMILES '{test_smiles}' -> 分子式: {get_molecular_formula(mol)}")
    else:
        print(f"\n[结果] SMILES '{test_smiles}' 校验失败。")


def _ask_llm_smoke():
    """Day 7-8 冒烟测试：若已配置密钥则发一条化学问题验证通路。"""
    print("=" * 60)
    print("Day 7-8 - ask_llm 联调测试")
    print("=" * 60)
    if get_llm_config() is None:
        print("[跳过] 未检测到 .env 配置。请创建 .env 填入 API_KEY/BASE_URL/MODEL 后重试。")
        return
    answer = ask_llm("用一句话说明阿司匹林（乙酰水杨酸）的主要药理作用。")
    if answer:
        print("\n[模型回答]")
        print(answer)
    else:
        print("\n[结果] ask_llm 调用失败，请检查 API_KEY / BASE_URL / 网络。")


def _main_process_smoke(query="苯酚"):
    """端到端冒烟测试：支持化合物与反应两类输入。"""
    print("=" * 60)
    print(f"main_process 端到端测试（{query}）")
    print("=" * 60)
    result = main_process(query)
    print("\n" + "=" * 60)
    print("[结果汇总]")
    print(f"  输入: {result['input']}")
    if result.get("type") == "reaction":
        print(f"  类型: 反应")
        print(f"  反应物: {result['reactants']}")
        print(f"  产物: {result['products']}")
        print(f"  条件: {result.get('conditions') or '未标注'}, 可逆: {result.get('reversible')}")
        eq = result.get("equation_chemfig", "")
        print(f"  方程式 chemfig: {'已生成(' + str(len(eq)) + '字符)' if eq else '空'}")
    else:
        print(f"  类型: 化合物")
        print(f"  SMILES: {result.get('smiles')}")
        if result.get("properties"):
            p = result["properties"]
            print(f"  知识库: {p.get('name')} | {p.get('molecular_formula')} | MW={p.get('mol_weight')}")
        else:
            print("  知识库: 未命中（走 PubChem 在线解析）")
        cf = result.get("chemfig", "")
        print(f"  chemfig: {'已生成(' + str(len(cf)) + '字符)' if cf else '空'}")
    print("\n[模型回答]")
    print(result["answer"] if result.get("answer") else "（LLM 调用失败或未配置）")
    print("=" * 60)


if __name__ == "__main__":
    # 用法: python chem_agent.py                          运行 Day1 冒烟测试
    #       python chem_agent.py --llm                    额外运行 ask_llm 联调测试
    #       python chem_agent.py --main 苯酚              运行化合物端到端测试
    #       python chem_agent.py --main "A.B>>C.D"        运行反应方程式测试
    import sys
    argv = sys.argv[1:]
    if "--main" in argv:
        idx = argv.index("--main")
        query = argv[idx + 1] if idx + 1 < len(argv) else "苯酚"
        _main_process_smoke(query)
    else:
        _day1_smoke()
        if "--llm" in argv:
            _ask_llm_smoke()
