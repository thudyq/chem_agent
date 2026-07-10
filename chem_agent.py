# -*- coding: utf-8 -*-
"""
chem_agent.py
=============
有机化学知识智能体 - 主入口脚本。

Day 1   : validate_mol / get_molecular_formula —— RDKit 基础与分子式计算。
Day 7-8 : ask_llm —— 接入 OpenAI 兼容大模型 API（DeepSeek / SiliconFlow），
          temperature=0.2、重试 3 次、密钥从 .env 读取。
"""

import os
import time
from pathlib import Path

import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

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


if __name__ == "__main__":
    # 用法: python chem_agent.py        运行 Day1 冒烟测试
    #       python chem_agent.py --llm  额外运行 ask_llm 联调测试
    import sys
    _day1_smoke()
    if "--llm" in sys.argv:
        _ask_llm_smoke()
