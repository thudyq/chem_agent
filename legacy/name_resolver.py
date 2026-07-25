# -*- coding: utf-8 -*-
"""
legacy/name_resolver.py
      ======================
名称 -> SMILES 解析器。

Day 2-3 任务：
    - 主路径：调用 PubChem REST API 获取 SMILES。
    - 备选路径：API 失败时，使用本地 pyopsin 解析 IUPAC 名称。
    - 全程处理网络超时与异常，确保程序不崩溃。

用法:
    from legacy.name_resolver import name_to_smiles
    smiles = name_to_smiles("aspirin")
"""

import sys
import time
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

# PubChem PUG REST 端点：按名称查询 CanonicalSMILES（返回纯文本 TXT）。
# 注意：AGENT.md 原写的 .../{name}/SMILES 会被判为非法 operation（返回 400），
#       正确格式为 property/CanonicalSMILES/TXT。
PUBCHEM_TEMPLATE = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
    "/property/CanonicalSMILES/TXT"
)
DEFAULT_TIMEOUT = 10  # 秒；兼顾网络抖动与用户等待体验
_503_RETRIES = 3      # PubChem 限流（503）时的最大重试次数


def _http_get(url: str, timeout: int):
    """HTTP GET：优先 curl_cffi（浏览器 TLS 指纹）。

    PubChem 按 TLS ClientHello 指纹分级限流：python-requests 的 OpenSSL
    指纹会被归入机器人流量，持续返回 503（PUGREST.ServerBusy）；
    浏览器指纹（curl_cffi impersonate）可正常访问。curl_cffi 未安装时
    回退 requests（保留原行为）。
    """
    try:
        from curl_cffi import requests as creq
        return creq.get(url, timeout=timeout, impersonate="chrome")
    except ImportError:
        pass
    return requests.get(
        url, timeout=timeout, headers={"User-Agent": "chem_agent/1.0"}
    )


# --------------------------------------------------------------------------- #
# 主路径：PubChem REST API
# --------------------------------------------------------------------------- #
def pubchem_to_smiles(name: str, timeout: int = DEFAULT_TIMEOUT):
    """通过 PubChem REST API 把化学名称转换为 SMILES。

    参数:
        name (str): 化学名称，例如 "aspirin"、"benzene"，或 IUPAC 名称。
        timeout (int): HTTP 请求超时秒数。

    返回:
        str | None: 合法的 Canonical SMILES 字符串；失败时返回 None。
    """
    if not name or not isinstance(name, str):
        print("[pubchem_to_smiles] 输入名称为空或类型错误。")
        return None

    url = PUBCHEM_TEMPLATE.format(name=requests.utils.quote(name))
    print(f"[pubchem_to_smiles] 请求 PubChem: {url}")
    resp = None
    for attempt in range(1, _503_RETRIES + 1):
        try:
            resp = _http_get(url, timeout)
        except requests.exceptions.Timeout:
            print(f"[pubchem_to_smiles] 请求超时（>{timeout}s），将尝试备选路径。")
            return None
        except requests.exceptions.RequestException as e:
            # 覆盖连接错误、DNS 失败、SSL 错误等所有 requests 异常
            print(f"[pubchem_to_smiles] 网络异常: {e}")
            return None
        except Exception as e:
            # curl_cffi 的异常类型与 requests 不同名，一并兜底
            print(f"[pubchem_to_smiles] 网络异常: {e}")
            return None
        if resp.status_code != 503:
            break
        # 限流：按 Retry-After 等待后重试（封顶 30s）
        try:
            wait = int(resp.headers.get("Retry-After", "10") or 10)
        except (TypeError, ValueError):
            wait = 10
        wait = min(wait, 30)
        print(f"[pubchem_to_smiles] PubChem 限流（503），{wait}s 后重试"
              f"（第 {attempt}/{_503_RETRIES} 次）...")
        time.sleep(wait)

    if resp.status_code == 404:
        # PubChem 找不到该名称时返回 404
        print(f"[pubchem_to_smiles] PubChem 未收录名称 '{name}'（404）。")
        return None
    if resp.status_code != 200:
        print(f"[pubchem_to_smiles] PubChem 返回异常状态码: {resp.status_code}")
        return None

    # 响应体可能含多行（同名多个记录），逐行用 rdkit 校验，取首个合法项
    for line in resp.text.splitlines():
        smiles = line.strip()
        if smiles and _is_valid_smiles(smiles):
            print(f"[pubchem_to_smiles] 命中: {name!r} -> {smiles}")
            return smiles

    print(f"[pubchem_to_smiles] PubChem 返回的内容无法解析为合法 SMILES。")
    return None


# --------------------------------------------------------------------------- #
# 备选路径：本地 pyopsin（IUPAC 名称解析）
# --------------------------------------------------------------------------- #
# pyopsin 通过 JPype 启动 JVM，实例化开销较大，故做模块级单例缓存。
# 仅在首次调用 pyopsin_to_smiles 时初始化；后续复用同一实例。
_OPSIN_INSTANCE = None
_OPSIN_AVAILABLE = None


def _get_opsin_instance():
    global _OPSIN_INSTANCE, _OPSIN_AVAILABLE
    if _OPSIN_INSTANCE is not None:
        return _OPSIN_INSTANCE
    if _OPSIN_AVAILABLE is False:
        return None

    try:
        from pyopsin import PyOpsin
    except ImportError:
        print("[pyopsin_to_smiles] pyopsin 未安装，跳过本地备选路径。")
        _OPSIN_AVAILABLE = False
        return None

    try:
        # exit_on_error=False：解析失败时返回 None 而非抛 ValueError
        _OPSIN_INSTANCE = PyOpsin(exit_on_error=False)
        _OPSIN_AVAILABLE = True
        print("[pyopsin_to_smiles] PyOpsin 实例已初始化（JVM 已启动）。")
    except FileNotFoundError as e:
        # 缺少 opsin-cli jar 包
        print(f"[pyopsin_to_smiles] PyOpsin 初始化失败（缺少 jar）: {e}")
        _OPSIN_AVAILABLE = False
        return None
    except Exception as e:
        # JVM 启动失败等其他异常（含 Java 异常）
        print(f"[pyopsin_to_smiles] PyOpsin 初始化异常: {e}")
        _OPSIN_AVAILABLE = False
        return None
    return _OPSIN_INSTANCE


def pyopsin_to_smiles(name: str):
    """使用本地 pyopsin 库解析 IUPAC 名称（离线，无需网络）。

    PubChem 主路径失败时启用。pyopsin 未安装或解析失败时返回 None，
    绝不让异常上抛中断主流程。

    参数:
        name (str): IUPAC 系统命名，例如 "2-acetoxybenzoic acid"。

    返回:
        str | None: 合法 SMILES；失败返回 None。
    """
    opsin = _get_opsin_instance()
    if opsin is None:
        return None

    print(f"[pyopsin_to_smiles] 本地解析: {name!r}")
    try:
        # to_smiles 即便传入单个字符串也返回 List[str]，取首元素
        result = opsin.to_smiles(name)
    except Exception as e:
        # pyopsin 经 JPype 调 Java，可能抛出非 Python 原生异常
        print(f"[pyopsin_to_smiles] pyopsin 调用异常: {e}")
        return None

    if isinstance(result, (list, tuple)):
        if not result or not result[0]:
            print(f"[pyopsin_to_smiles] pyopsin 无法解析名称 '{name}'。")
            return None
        smiles = result[0]
    else:
        smiles = result

    if not isinstance(smiles, str) or not smiles.strip():
        print(f"[pyopsin_to_smiles] pyopsin 无法解析名称 '{name}'。")
        return None

    smiles = smiles.strip()
    if _is_valid_smiles(smiles):
        print(f"[pyopsin_to_smiles] 命中: {name!r} -> {smiles}")
        return smiles

    print(f"[pyopsin_to_smiles] pyopsin 返回值非合法 SMILES: {smiles!r}")
    return None


# --------------------------------------------------------------------------- #
# 主控函数
# --------------------------------------------------------------------------- #
def name_to_smiles(name: str):
    """名称 -> SMILES 主控：先 PubChem，失败回退 pyopsin，全失败返回 None。

    参数:
        name (str): 化学名称或 IUPAC 名称。

    返回:
        str | None: 合法 SMILES；两条路径都失败时返回 None。
    """
    print(f"[name_to_smiles] 开始解析名称: {name!r}")

    # 主路径：在线 PubChem
    smiles = pubchem_to_smiles(name)
    if smiles:
        return smiles

    # 备选路径：本地 pyopsin
    print("[name_to_smiles] 主路径失败，回退到本地 pyopsin ...")
    smiles = pyopsin_to_smiles(name)
    if smiles:
        return smiles

    print(f"[name_to_smiles] 名称 '{name}' 解析失败（PubChem + pyopsin 均失败）。")
    return None


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _is_valid_smiles(smiles: str) -> bool:
    """用 rdkit 校验 SMILES 是否可解析为合法分子。

    rdkit 不可用时降级为「非空字符串」检查，避免硬依赖阻断主流程。
    """
    if not smiles:
        return False
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles) is not None
    except ImportError:
        # rdkit 缺失时仅做轻量校验，不阻断
        return len(smiles.strip()) > 0


# --------------------------------------------------------------------------- #
# 测试入口
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Day 3 末 [USER ACTION] 的测试用例：aspirin、benzene
    print("=" * 60)
    print("name_resolver 测试：aspirin / benzene")
    print("=" * 60)

    for test_name in ("aspirin", "benzene"):
        print("-" * 60)
        result = name_to_smiles(test_name)
        if result:
            print(f">>> {test_name!r} -> SMILES: {result}")
        else:
            print(f">>> {test_name!r} 解析失败。")

    print("=" * 60)
    print("测试结束。")
