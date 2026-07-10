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
import time
from pathlib import Path

import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from utils.name_resolver import name_to_smiles
from utils.structure_render import smiles_to_tikz
from utils.db_helper import query_property

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
                content = data["choices"][0]["message"]["content"]
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
    "你是一位严谨的有机化学知识助手。请用准确、流畅的中文回答用户的化学问题。"
    "优先依据提供的化合物数据（分子式、分子量、SMILES、结构式代码）；"
    "回答应涵盖关键结构特征、官能团、基本理化性质；结构式以 chemfig 代码块呈现。"
    "若提供的数据缺失，可基于化学知识补充，但需明确标注为推测。"
)


def _build_prompt(user_input, smiles, props, chemfig):
    """组装 RAG prompt：用户问题 + 知识库数据 + 结构式代码。"""
    lines = [f"用户问题/输入：{user_input}", ""]

    if props:
        lines.append("已知化合物信息（来自本地知识库）：")
        for key in ("name", "en_name", "iupac_name", "molecular_formula",
                    "mol_weight", "xlogp", "melting_point", "boiling_point",
                    "density", "cas"):
            val = props.get(key)
            if val not in (None, "", []):
                lines.append(f"- {_FIELD_LABELS.get(key, key)}：{val}")
    elif smiles:
        lines.append("（该化合物不在本地知识库，以下为在线解析结果）")
        lines.append(f"- SMILES：{smiles}")
    else:
        lines.append("（未能解析出明确化合物，请基于问题本身作答）")
    lines.append("")

    if chemfig:
        lines.append("结构式代码（LaTeX chemfig，可复制到 Overleaf 编译）：")
        lines.append(chemfig)
        lines.append("")

    lines.append("请综合以上信息用中文作答，并引用上面的结构式代码。")
    return "\n".join(lines)


_FIELD_LABELS = {
    "name": "中文名", "en_name": "英文名", "iupac_name": "IUPAC名",
    "molecular_formula": "分子式", "mol_weight": "分子量", "xlogp": "XLogP",
    "melting_point": "熔点", "boiling_point": "沸点", "density": "密度",
    "cas": "CAS号",
}


def main_process(user_input: str) -> dict:
    """主控流程：名称/问题 -> SMILES + 性质数据 + 结构式代码 + 中文回答。

    顺序：
        1. 先查本地知识库（支持中/英/IUPAC 名）；命中即拿到 SMILES 与性质。
        2. 库中无 SMILES 时，回退到 PubChem 在线解析（英文/IUPAC 名）。
        3. SMILES -> chemfig 结构式代码。
        4. 组装 RAG prompt 调用 LLM 生成中文回答。

    任何子环节失败均降级处理，返回部分结果（对应字段为 None/""），绝不抛异常。

    返回:
        dict: {input, smiles, properties, chemfig, answer}
    """
    print(f"\n[main_process] 输入: {user_input!r}")

    props = query_property(user_input)
    smiles = props.get("smiles") if props else None
    if smiles:
        print(f"[main_process] 知识库命中: {props.get('name')} ({props.get('molecular_formula')})")

    # 库中无 SMILES -> PubChem 在线解析（中文名 PubChem 无法识别，
    # 主要服务于知识库外、英文/IUPAC 命名的化合物）
    if not smiles:
        smiles = name_to_smiles(user_input)

    chemfig = smiles_to_tikz(smiles) if smiles else ""

    print("[main_process] 调用 LLM 生成回答 ...")
    prompt = _build_prompt(user_input, smiles, props, chemfig)
    answer = ask_llm(prompt, system_prompt=_SYSTEM_PROMPT)

    result = {
        "input": user_input,
        "smiles": smiles,
        "properties": props,
        "chemfig": chemfig,
        "answer": answer,
    }
    print("[main_process] 完成。")
    return result


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
    """Day 11-12 冒烟测试：跑通「名称 -> 结构式 + 性质 + 中文回答」全链路。"""
    print("=" * 60)
    print(f"Day 11-12 - main_process 端到端测试（{query}）")
    print("=" * 60)
    result = main_process(query)
    print("\n" + "=" * 60)
    print("[结果汇总]")
    print(f"  输入: {result['input']}")
    print(f"  SMILES: {result['smiles']}")
    if result["properties"]:
        p = result["properties"]
        print(f"  知识库: {p.get('name')} | {p.get('molecular_formula')} | MW={p.get('mol_weight')}")
    else:
        print("  知识库: 未命中（走 PubChem 在线解析）")
    print(f"  chemfig: {'已生成(' + str(len(result['chemfig'])) + '字符)' if result['chemfig'] else '空'}")
    print("\n[模型回答]")
    print(result["answer"] if result["answer"] else "（LLM 调用失败或未配置）")
    print("=" * 60)


if __name__ == "__main__":
    # 用法: python chem_agent.py              运行 Day1 冒烟测试
    #       python chem_agent.py --llm        额外运行 ask_llm 联调测试
    #       python chem_agent.py --main 苯酚  运行 main_process 端到端测试
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
