# -*- coding: utf-8 -*-
"""utils/name_resolver.py — 化学名称 → SMILES 解析（PubChem 主路径）。

从 legacy/name_resolver.py 提升并现代化：PubChem REST 查询 + RDKit 校验 +
curl_cffi 浏览器 TLS 指纹绕限流。作为 LLM 输出的兜底：当 LLM 的 SMILES
不可靠或用户输入为化学名称时，用 PubChem 反查权威 SMILES。

用法:
    from utils.name_resolver import name_to_smiles
    smiles = name_to_smiles("aspirin")   # → CC(=O)OC1=CC=CC=C1C(=O)O
"""

import contextlib
import time
from urllib.parse import quote

# PubChem PUG REST：按名称查询 CanonicalSMILES（返回纯文本 TXT）。
# 注意：/SMILES 操作非法（400），正确格式为 property/CanonicalSMILES/TXT。
PUBCHEM_TEMPLATE = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
    "/property/CanonicalSMILES/TXT"
)
DEFAULT_TIMEOUT = 10
_503_MAX_WAIT = 30   # PubChem 503 限流时按 Retry-After 等待的上限（秒）

# 常见化合物的中文名 → PubChem 查询用英文名（PubChem 不识别中文）。
# 覆盖 AGENT.md 指定的 20 个常用化合物；其他中文名可扩展或由英文/IUPAC 兜底。
COMMON_CN_EN = {
    "甲烷": "methane", "乙醇": "ethanol", "苯": "benzene",
    "甲苯": "toluene", "苯酚": "phenol", "环己烷": "cyclohexane",
    "乙酸": "acetic acid", "丙酮": "acetone", "乙醚": "diethyl ether",
    "氯仿": "chloroform", "四氯化碳": "carbon tetrachloride",
    "乙二醇": "ethylene glycol", "甘油": "glycerol", "苯胺": "aniline",
    "硝基苯": "nitrobenzene", "苯甲酸": "benzoic acid",
    "水杨酸": "salicylic acid", "阿司匹林": "aspirin",
    "对乙酰氨基酚": "acetaminophen", "咖啡因": "caffeine",
}


def http_get(url: str, timeout: int = DEFAULT_TIMEOUT, retries_503: int = 3):
    """HTTP GET：curl_cffi 浏览器 TLS 指纹 + 503 限流重试。

    PubChem 按 TLS ClientHello 指纹分级限流：python-requests 的 OpenSSL
    指纹归入机器人流量，持续 503；curl_cffi 的浏览器指纹可正常访问。
    """
    def _raw(url: str, timeout: int):
        try:
            from curl_cffi import requests as creq
            return creq.get(url, timeout=timeout, impersonate="chrome")
        except ImportError:
            import requests
            return requests.get(
                url, timeout=timeout,
                headers={"User-Agent": "chem_agent/1.0"},
            )

    last = None
    for attempt in range(1, retries_503 + 1):
        resp = _raw(url, timeout)
        if resp.status_code != 503:
            return resp
        last = resp
        try:
            wait = int(resp.headers.get("Retry-After", "10") or 10)
        except (TypeError, ValueError):
            wait = 10
        wait = min(wait, _503_MAX_WAIT)
        print(f"[http_get] 503 限流，{wait}s 后重试（第 {attempt}/{retries_503} 次）...")
        time.sleep(wait)
    return last


def _is_valid_smiles(smiles: str) -> bool:
    """RDKit 校验 SMILES；rdkit 缺失时降级为非空检查。"""
    if not smiles:
        return False
    try:
        from rdkit import Chem
        with contextlib.nullcontext():
            return Chem.MolFromSmiles(smiles) is not None
    except ImportError:
        return len(smiles.strip()) > 0


# 查询失败负缓存：同一名称**确定性**查询失败（404 未收录 / 200 但无合法
# SMILES）后不再重复发起 HTTP 请求（修正循环/多轮调用场景避免重复网络消耗）。
# 暂时性问题（网络异常、5xx）不缓存，下次重试。
_NEGATIVE_CACHE: set = set()


def name_to_smiles(name: str, timeout: int = DEFAULT_TIMEOUT,
                   use_cache: bool = True):
    """化学名称 → SMILES（PubChem 主路径，RDKit 校验）。

    参数:
        name: 化学名称（俗名 "aspirin"、英文 "benzene"、IUPAC 均可）。
        timeout: HTTP 超时秒数。
        use_cache: 命中失败负缓存（_NEGATIVE_CACHE）时直接返回 None、
            不再发请求；默认开启，测试可传 False 绕过。

    返回:
        str | None: 合法 Canonical SMILES；失败返回 None（不抛异常）。
    """
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if not name:
        return None
    # 中文名映射英文（PubChem 不识别中文；未收录的中文名直接查，失败返回 None）
    query = COMMON_CN_EN.get(name, name)
    if use_cache and query in _NEGATIVE_CACHE:
        return None
    url = PUBCHEM_TEMPLATE.format(name=quote(query))
    try:
        resp = http_get(url, timeout)
    except Exception:
        # 网络异常为暂时性问题：不缓存，下次重试
        return None
    if resp.status_code == 404:
        _NEGATIVE_CACHE.add(query)
        return None
    if resp.status_code != 200:
        return None
    # 响应可能含多行（同名多记录），逐行校验取首个合法项
    for line in resp.text.splitlines():
        smiles = line.strip()
        if smiles and _is_valid_smiles(smiles):
            return smiles
    # 200 但无合法 SMILES → 确定性失败，缓存
    _NEGATIVE_CACHE.add(query)
    return None
