# -*- coding: utf-8 -*-
"""legacy/http_utils.py — 带浏览器 TLS 指纹与限流重试的 HTTP GET 工具。

PubChem 按 TLS ClientHello 指纹分级限流：python-requests 的 OpenSSL 指纹
会被归入机器人流量，持续返回 503（PUGREST.ServerBusy）；curl_cffi 的
浏览器指纹可正常访问。本模块把这一策略集中到一处，供 legacy 各模块复用
（name_resolver / db_helper 等）。
"""

import time

import requests

DEFAULT_TIMEOUT = 10      # 秒；兼顾网络抖动与用户等待体验
_503_MAX_WAIT = 30        # Retry-After 等待封顶（秒）


def _raw_get(url: str, timeout: int):
    """优先 curl_cffi（浏览器 TLS 指纹），未安装时回退 requests。"""
    try:
        from curl_cffi import requests as creq
        return creq.get(url, timeout=timeout, impersonate="chrome")
    except ImportError:
        pass
    return requests.get(
        url, timeout=timeout, headers={"User-Agent": "chem_agent/1.0"}
    )


def http_get(url: str, timeout: int = DEFAULT_TIMEOUT, retries_503: int = 3):
    """HTTP GET：浏览器 TLS 指纹 + 503 限流重试。

    参数:
        url: 请求地址。
        timeout: 单次请求超时秒数。
        retries_503: 收到 503（PUGREST.ServerBusy 限流）时的最大尝试次数，
                     每次按响应 Retry-After 头等待（封顶 30s）。

    返回:
        响应对象（status_code / text / json / headers 语义与
        requests.Response 一致）；重试耗尽仍 503 时返回最后一次响应。
        网络异常（超时/连接错误等）按底层客户端原样抛出，由调用方处理。
    """
    last = None
    for attempt in range(1, retries_503 + 1):
        resp = _raw_get(url, timeout)
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


if __name__ == "__main__":
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin"
           "/property/CanonicalSMILES/TXT")
    resp = http_get(url)
    print(f"status: {resp.status_code}")
    print(resp.text.strip()[:80])
