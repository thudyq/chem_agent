# -*- coding: utf-8 -*-
"""tests/test_mcp_api.py — ChatGPT（MCP）工具层测试。

覆盖：MCP 握手与工具清单、四个只读工具的成败路径、鉴权开关（MCP_API_KEY）、
附件路由防穿越。外部依赖全部 mock（PubChem / LaTeX 编译），离线可跑。
"""

import contextlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import mcp_api

INIT_PARAMS = {
    "protocolVersion": "2025-06-18",
    "capabilities": {},
    "clientInfo": {"name": "pytest", "version": "0.0.0"},
}
_HEADERS = {"MCP-Protocol-Version": "2025-06-18"}


@contextlib.contextmanager
def _mount(monkeypatch, tmp_path):
    """独立 app + MCP 挂载 + 附件目录指向 tmp（不动真实 data/）。"""
    monkeypatch.setattr(mcp_api, "_MCP_DIR", tmp_path / "mcp_attachments")

    @contextlib.asynccontextmanager
    async def _lifespan(_app):
        async with mcp_api.mcp_session():
            yield

    app = FastAPI(lifespan=_lifespan)
    assert mcp_api.mount_mcp(app) is True
    with TestClient(app) as client:
        yield client


def _rpc(client, method, params=None, id_=1):
    body = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        body["params"] = params
    return client.post("/mcp", json=body, headers=_HEADERS)


def _init(client):
    r = _rpc(client, "initialize", INIT_PARAMS, id_=0)
    assert r.status_code == 200, r.text
    return r.json()["result"]


def _call(client, name, arguments, id_=99):
    r = _rpc(client, "tools/call",
             {"name": name, "arguments": arguments}, id_=id_)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body
    return body["result"]


def test_mount_without_mcp_package(monkeypatch):
    """mcp 包缺失时 mount 必须优雅跳过（返回 False），绝不能抛异常。"""
    app = FastAPI()
    monkeypatch.setattr(mcp_api, "_MCP_AVAILABLE", False)
    monkeypatch.setattr(mcp_api, "_mcp", None)
    assert mcp_api.mount_mcp(app) is False


def test_initialize_and_tools_listed(monkeypatch, tmp_path):
    with _mount(monkeypatch, tmp_path) as client:
        result = _init(client)
        assert result["serverInfo"]["name"] == "chem-agent"
        tools = _rpc(client, "tools/list").json()["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {"validate_smiles", "resolve_chemical_name",
                         "render_chemistry", "get_tag_syntax"}
        for t in tools:
            assert t["annotations"]["readOnlyHint"] is True
            assert t["inputSchema"]["type"] == "object"


def test_validate_smiles_ok(monkeypatch, tmp_path):
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "validate_smiles", {"smiles": "c1ccccc1"})
        sc = out["structuredContent"]
        assert sc["valid"] is True
        assert sc["canonical_smiles"] == "c1ccccc1"
        assert sc["formula"] == "C6H6"
        assert sc["molecular_weight"] == pytest.approx(78.11, abs=0.1)


def test_validate_smiles_h_prefix_normalized(monkeypatch, tmp_path):
    """[H3O+] 这类氢前缀写法要被规范化后放行（与校验层同口径）。"""
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "validate_smiles", {"smiles": "[H3O+]"})
        sc = out["structuredContent"]
        assert sc["valid"] is True
        assert "[OH3+]" in sc["canonical_smiles"]


def test_validate_smiles_invalid(monkeypatch, tmp_path):
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "validate_smiles", {"smiles": "这不是SMILES"})
        sc = out["structuredContent"]
        assert sc["valid"] is False
        assert sc["canonical_smiles"] == ""


def test_resolve_chemical_name(monkeypatch, tmp_path):
    calls = {}

    def fake_lookup(name):
        calls["name"] = name
        return "CCO" if name == "ethanol" else None

    monkeypatch.setattr(mcp_api, "_name_to_smiles", fake_lookup)
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        hit = _call(client, "resolve_chemical_name", {"name": "ethanol"})
        assert hit["structuredContent"] == {"name": "ethanol",
                                            "smiles": "CCO", "found": True}
        miss = _call(client, "resolve_chemical_name", {"name": "不存在的物质"},
                     id_=2)
        assert miss["structuredContent"]["found"] is False
        assert calls["name"] == "不存在的物质"


def test_render_valid_tag(monkeypatch, tmp_path):
    """合法标记 → 图 URL + 注入文本；LaTeX 编译 mock 成固定 PNG 字节。"""
    monkeypatch.setattr("core.attachments.compile_tikz_to_png",
                        lambda code, **kw: b"fake-png-bytes")
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "render_chemistry",
                    {"text": "苯的结构：[STRUCT:c1ccccc1,label=苯]"})
        sc = out["structuredContent"]
        assert sc["errors"] == []
        assert len(sc["figures"]) == 1
        fig = sc["figures"][0]
        assert fig["kind"] == "STRUCT"
        assert fig["label"] == "苯"
        assert "/mcp/files/" in fig["url"] and fig["url"].endswith(".png")
        assert "![化学图示-1](" in sc["injected_text"]
        assert sc["tikz_sources"] == []          # 默认不带源码
        # 附件真的落在（被改指的）独立目录里
        assert (tmp_path / "mcp_attachments").is_dir()


def test_render_include_tikz(monkeypatch, tmp_path):
    monkeypatch.setattr("core.attachments.compile_tikz_to_png",
                        lambda code, **kw: b"fake-png-bytes")
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "render_chemistry",
                    {"text": "[STRUCT:c1ccccc1]", "include_tikz": True})
        sc = out["structuredContent"]
        assert len(sc["tikz_sources"]) == 1
        assert "\\begin{tikzpicture}" in sc["tikz_sources"][0]


def test_render_invalid_smiles_reports_error(monkeypatch, tmp_path):
    """非法 SMILES：不出图、结构化报错（回传修正闭环的依据）。"""
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "render_chemistry", {"text": "[STRUCT:XYZ!]"})
        sc = out["structuredContent"]
        assert sc["figures"] == []
        assert len(sc["errors"]) == 1
        err = sc["errors"][0]
        assert err["kind"] == "STRUCT"
        assert err["reason"]
        assert err["hint"]                       # 友好降级文案可直接展示


def test_render_compile_failure_reported(monkeypatch, tmp_path):
    """LaTeX 缺失（编译返回 None）→ 记入 errors，不静默吞掉。"""
    monkeypatch.setattr("core.attachments.compile_tikz_to_png",
                        lambda code, **kw: None)
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "render_chemistry", {"text": "[STRUCT:c1ccccc1]"})
        sc = out["structuredContent"]
        assert sc["figures"] == []
        assert any("LaTeX" in e["reason"] for e in sc["errors"])


def test_render_without_tags(monkeypatch, tmp_path):
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "render_chemistry", {"text": "纯文本，没有标记"})
        sc = out["structuredContent"]
        assert sc["figures"] == [] and sc["errors"] == []
        assert "未发现渲染标记" in sc["note"]


def test_render_over_limit(monkeypatch, tmp_path):
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "render_chemistry", {"text": "字" * 9000})
        sc = out["structuredContent"]
        assert sc["figures"] == []
        assert sc["errors"][0]["kind"] == "INPUT"


def test_get_tag_syntax(monkeypatch, tmp_path):
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "get_tag_syntax", {})
        sc = out["structuredContent"]
        assert "[STRUCT:" in sc["syntax_reference"]
        assert sc["examples"] == ""              # 默认不带示例段
        out2 = _call(client, "get_tag_syntax",
                     {"include_examples": True}, id_=2)
        assert "示例" in out2["structuredContent"]["examples"]


def test_auth_toggle(monkeypatch, tmp_path):
    """MCP_API_KEY 设置后：无头 401 / 错 key 401 / 对 key 200；未设置放行。"""
    monkeypatch.setenv("MCP_API_KEY", "sk-mcp-test")
    with _mount(monkeypatch, tmp_path) as client:
        body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": INIT_PARAMS}
        no_header = client.post("/mcp", json=body, headers=_HEADERS)
        assert no_header.status_code == 401
        bad = client.post("/mcp", json=body,
                          headers={**_HEADERS,
                                   "Authorization": "Bearer wrong"})
        assert bad.status_code == 401
        good = client.post("/mcp", json=body,
                           headers={**_HEADERS,
                                    "Authorization": "Bearer sk-mcp-test"})
        assert good.status_code == 200

    monkeypatch.delenv("MCP_API_KEY", raising=False)
    with _mount(monkeypatch, tmp_path) as client:
        assert client.post("/mcp", json={"jsonrpc": "2.0", "id": 1,
                                         "method": "initialize",
                                         "params": INIT_PARAMS},
                           headers=_HEADERS).status_code == 200


def test_attachment_route_traversal(monkeypatch, tmp_path):
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        assert client.get("/mcp/files/..%2Fsecret.png",
                          headers=_HEADERS).status_code in (400, 404)
        assert client.get(
            f"/mcp/files/{'a' * 32}.png", headers=_HEADERS).status_code == 404


def test_tool_functions_direct_call():
    """纯函数直调（不经 MCP 协议）也要维持同一契约——工具即函数。"""
    ok = mcp_api.tool_validate_smiles("c1ccccc1")
    assert ok["valid"] is True and ok["formula"] == "C6H6"
    empty = mcp_api.tool_validate_smiles("")
    assert empty["valid"] is False
