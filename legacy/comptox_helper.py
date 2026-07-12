# -*- coding: utf-8 -*-
"""
utils/comptox_helper.py
=======================
EPA CompTox Chemicals Dashboard (CTX) API 客户端。

用途：获取化合物的**实验**物理化学性质（熔点/沸点/密度/logP/溶解度等），
补 PubChem 与 ChemSpider API 都拿不到的实验数据缺口。

API 文档（经 librarian 核实，源背书：USEPA 官方 ctx-python/ctxR 包 + OpenAPI spec）：
    - Base: https://comptox.epa.gov/ctx-api/
    - 认证: X-Api-Key 头（免费 key，邮件 ccte_api@epa.gov 申请）
    - 搜索: GET /chemical/search/equal/{word}  (word=名称/CAS/DTXSID/InChIKey)
    - 实验性质: GET /chemical/property/experimental/search/by-dtxsid/{dtxsid}
    - 沙箱实测：401 无 key 时确认端点在线

无 key 时所有查询优雅返回 None，不阻断主流程。
"""

import os
from pathlib import Path

import requests

_CTX_BASE = "https://comptox.epa.gov/ctx-api"
_CTX_TIMEOUT = 20

# CompTox propName -> 本项目字段名
_PROP_MAP = {
    "Melting Point": "melting_point",
    "Boiling Point": "boiling_point",
    "Density": "density",
    "logP": "xlogp",
    "Log P": "xlogp",
    "Octanol/Water Partition Coefficient": "xlogp",
    "Water Solubility": "solubility",
    "Vapor Pressure": "vapor_pressure",
}


def _load_env():
    """读取 .env 到 os.environ（python-dotenv 优先，缺失时手动解析兜底）。"""
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


def get_api_key() -> str:
    """读取 COMPTOX_API_KEY；未配置返回空串。"""
    _load_env()
    return os.environ.get("COMPTOX_API_KEY", "").strip()


def _headers():
    return {"accept": "application/json", "x-api-key": get_api_key()}


def search_dtxsid(name: str):
    """按名称/CAS/InChIKey 搜索化合物，返回首个 DTXSID 或 None。"""
    if not get_api_key():
        return None
    url = f"{_CTX_BASE}/chemical/search/equal/{requests.utils.quote(name)}"
    print(f"[comptox] 搜索: {name!r}")
    try:
        resp = requests.get(url, headers=_headers(), timeout=_CTX_TIMEOUT)
    except requests.exceptions.RequestException as e:
        print(f"[comptox] 搜索异常: {e}")
        return None
    if resp.status_code != 200:
        print(f"[comptox] 搜索失败 HTTP {resp.status_code}")
        return None
    results = resp.json()
    if not results:
        print(f"[comptox] 未找到: {name!r}")
        return None
    dtxsid = results[0].get("dtxsid")
    print(f"[comptox] 命中: {name!r} -> {dtxsid}")
    return dtxsid


def fetch_experimental_properties(dtxsid: str):
    """按 DTXSID 取实验性质列表；失败返回 None。"""
    if not dtxsid or not get_api_key():
        return None
    url = f"{_CTX_BASE}/chemical/property/experimental/search/by-dtxsid/{dtxsid}"
    try:
        resp = requests.get(url, headers=_headers(), timeout=_CTX_TIMEOUT)
    except requests.exceptions.RequestException as e:
        print(f"[comptox] 性质查询异常: {e}")
        return None
    if resp.status_code != 200:
        print(f"[comptox] 性质查询失败 HTTP {resp.status_code}")
        return None
    props = resp.json()
    if not isinstance(props, list):
        return None
    print(f"[comptox] {dtxsid} 实验性质 {len(props)} 条")
    return props


def get_experimental_by_name(name: str):
    """名称 -> 实验性质 dict（映射到本项目字段名）。

    返回:
        dict | None: {melting_point, boiling_point, density, xlogp, solubility, vapor_pressure}
        各字段值为 "value unit" 字符串；无 key 或未命中返回 None。
    """
    if not get_api_key():
        return None
    dtxsid = search_dtxsid(name)
    if not dtxsid:
        return None
    props = fetch_experimental_properties(dtxsid)
    if not props:
        return None
    result = {}
    for p in props:
        prop_name = p.get("propName", "")
        field = _PROP_MAP.get(prop_name)
        if not field or field in result:
            # 未映射的字段跳过；同名字段只取首条
            continue
        value = p.get("propValue")
        unit = p.get("propUnit", "")
        if value is None:
            continue
        result[field] = f"{value} {unit}".strip()
    return result or None


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("comptox_helper 测试")
    print("=" * 60)
    key = get_api_key()
    if not key:
        print("[跳过] 未配置 COMPTOX_API_KEY。请邮件 ccte_api@epa.gov 申请后填入 .env。")
        sys.exit(0)
    # 有 key 时测试 aspirin
    target = sys.argv[1] if len(sys.argv) > 1 else "aspirin"
    print(f"\n[测试] {target}")
    res = get_experimental_by_name(target)
    if res:
        for k, v in res.items():
            print(f"  {k}: {v}")
    else:
        print("  （未取到实验性质）")
