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


@pytest.fixture(autouse=True)
def isolate_mcp_api_key(monkeypatch):
    """默认隔离本地 .env 的 MCP 运行配置（密钥 + B′/压缩/图像块开关），
    避免测试依赖开发机配置状态。需要真实值的用例自行 monkeypatch.setenv 覆盖。"""
    monkeypatch.setenv("MCP_API_KEY", "")
    monkeypatch.setenv("MCP_PUBLIC_BASE_URL", "")
    monkeypatch.setenv("MCP_AUTO_OPEN", "")
    monkeypatch.setenv("MCP_IMAGE_MAX_KB", "")
    monkeypatch.setenv("MCP_IMAGE_BLOCK_KB", "")


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
                         "render_chemistry", "get_tag_syntax", "test_image",
                         "test_image_content"}
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
    """合法标记 → widget 交付（data URI 内嵌）+ 元数据；编译 mock 成固定 PNG。"""
    png = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
    monkeypatch.setattr("utils.latex_compile.compile_tikz_to_png",
                        lambda code, title="", **kw: png)
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
        assert fig["size_bytes"] == len(png)
        assert fig["url"] is None                # 默认（隧道模式）不返回 URL
        assert sc["widget_uri"].startswith("ui://chem-agent/render/")
        assert "（图 1）" in sc["injected_text"]   # 无 URL → 占位而非图链
        assert "![化学图示-1](" not in sc["injected_text"]
        assert sc["tikz_sources"] == []          # 默认不带源码
        # 结果级 _meta 携带本次渲染的组件地址（官方 kitchen_sink 扁平键格式）
        meta = out.get("_meta") or out.get("meta") or {}
        assert meta["openai/outputTemplate"] == sc["widget_uri"]
        assert meta["openai/widgetAccessible"] is True
        assert meta["openai/toolInvocation/invoked"] == "图示已生成"
        assert meta["ui"]["resourceUri"] == sc["widget_uri"]
        # 图示落盘（运维可检查 data/mcp_attachments）——widget 模式也必须写
        files = list((tmp_path / "mcp_attachments").glob("*.png"))
        assert len(files) == 1 and files[0].stat().st_size == len(png)

        # widget HTML 经 resources/read 取回：data URI 内嵌且字节完整
        rr = _rpc(client, "resources/read",
                  {"uri": sc["widget_uri"]}, id_=77).json()["result"]
        html_doc = rr["contents"][0]["text"]
        assert 'src="data:image/png;base64,' in html_doc
        import base64 as _b64
        embedded = _b64.b64decode(
            html_doc.split("base64,", 1)[1].split('"', 1)[0])
        assert embedded == png
        assert html_doc == mcp_api._widgets[sc["widget_uri"].rsplit("/", 1)[-1]
                                            .removesuffix(".html")]
        assert "bridge诊断 diag-v2" in html_doc   # A 实验：桥接诊断条在位
        # A′：图像 content block 随结果返回（2026-09 成功报告形态）
        import base64 as _b64
        blocks = [c for c in out["content"] if c.get("type") == "image"]
        assert len(blocks) == 1
        assert blocks[0]["mimeType"] == "image/png"
        assert _b64.b64decode(blocks[0]["data"]) == png


def test_render_widget_uri_is_content_addressed(monkeypatch, tmp_path):
    """同输入 → 同 widget URI（缓存命中）；PNG 不同 → 不同 URI。"""
    monkeypatch.setattr("utils.latex_compile.compile_tikz_to_png",
                        lambda code, title="", **kw:
                        b"\x89PNG\r\n\x1a\n" + code.encode())
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        a = _call(client, "render_chemistry", {"text": "[STRUCT:c1ccccc1]"})
        b = _call(client, "render_chemistry", {"text": "[STRUCT:c1ccccc1]"},
                  id_=2)
        c = _call(client, "render_chemistry", {"text": "[STRUCT:CCO]"}, id_=3)
        ua = a["structuredContent"]["widget_uri"]
        assert b["structuredContent"]["widget_uri"] == ua   # 同图同址
        assert c["structuredContent"]["widget_uri"] != ua   # 异图异址


def test_tools_list_render_meta(monkeypatch, tmp_path):
    """A″ 对齐官方示例：工具定义级**无** meta（组件绑定在结果级 _meta）。"""
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        tools = _rpc(client, "tools/list").json()["result"]["tools"]
        render = next(t for t in tools if t["name"] == "render_chemistry")
        assert not (render.get("_meta") or render.get("meta"))


def test_render_url_mode(monkeypatch, tmp_path):
    """MCP_PUBLIC_BASE_URL 配置时：额外返回 /mcp/files 下载 URL + markdown 图链。"""
    png = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
    monkeypatch.setattr("utils.latex_compile.compile_tikz_to_png",
                        lambda code, title="", **kw: png)
    monkeypatch.setenv("MCP_PUBLIC_BASE_URL", "https://pub.example")
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "render_chemistry", {"text": "[STRUCT:c1ccccc1]"})
        sc = out["structuredContent"]
        fig = sc["figures"][0]
        assert fig["url"] is not None
        assert fig["url"].startswith("https://pub.example/mcp/files/")
        assert fig["url"].endswith(".png")
        assert "![化学图示-1](" in sc["injected_text"]
        # B′：content 文本显式给出可点击链接（ChatGPT 原样写进回答）
        text = next(c["text"] for c in out["content"] if c.get("type") == "text")
        assert f"图示链接[图 1]: {fig['url']}" in text


def test_render_auto_open(monkeypatch, tmp_path):
    """MCP_AUTO_OPEN=1：渲染完成自动弹出本机 PNG（B′：提问即见图）。"""
    opened = []
    monkeypatch.setattr(mcp_api, "_startfile",
                        lambda path: opened.append(path))
    monkeypatch.setattr("utils.latex_compile.compile_tikz_to_png",
                        lambda code, title="", **kw: b"\x89PNG\r\n\x1a\n" + b"z")
    monkeypatch.setenv("MCP_AUTO_OPEN", "1")
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "render_chemistry", {"text": "[STRUCT:c1ccccc1]"})
        assert out["structuredContent"]["errors"] == []
        assert len(opened) == 1
        assert opened[0].startswith(str(tmp_path / "mcp_attachments"))
        assert opened[0].endswith(".png")


def test_render_auto_open_off_by_default(monkeypatch, tmp_path):
    """未配置 MCP_AUTO_OPEN → 不弹窗（默认关闭，避免云端部署误弹）。"""
    opened = []
    monkeypatch.setattr(mcp_api, "_startfile",
                        lambda path: opened.append(path))
    monkeypatch.setattr("utils.latex_compile.compile_tikz_to_png",
                        lambda code, title="", **kw: b"\x89PNG\r\n\x1a\n" + b"z")
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        _call(client, "render_chemistry", {"text": "[STRUCT:c1ccccc1]"})
        assert opened == []


def test_render_include_tikz(monkeypatch, tmp_path):
    monkeypatch.setattr("utils.latex_compile.compile_tikz_to_png",
                        lambda code, title="", **kw: b"\x89PNG\r\n\x1a\n" + b"fake")
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "render_chemistry",
                    {"text": "[STRUCT:c1ccccc1]", "include_tikz": True})
        sc = out["structuredContent"]
        assert len(sc["tikz_sources"]) == 1
        assert "\\begin{tikzpicture}" in sc["tikz_sources"][0]


def test_adaptive_dpi_ladder(monkeypatch):
    """compile_tikz_to_png_adaptive：超阈值按阶梯降 DPI，取第一个达标者。"""
    import utils.latex_compile as lc

    by_dpi = {300: b"A" * 5000, 200: b"A" * 3000, 150: b"A" * 1500,
              100: b"A" * 800, 75: b"A" * 400}
    monkeypatch.setattr(lc, "_find_latex_engine", lambda: "xelatex")
    monkeypatch.setattr(lc, "_run_latex",
                        lambda engine, tex, flags: tex + ".pdf")
    monkeypatch.setattr(lc, "_pdf_to_png_pymupdf",
                        lambda pdf, dpi: by_dpi.get(dpi))
    code = "\\begin{tikzpicture}\\end{tikzpicture}"
    assert lc.compile_tikz_to_png_adaptive(code, max_kb=1) == by_dpi[100]
    # 无阈值：300dpi 快路径（零行为变化）
    monkeypatch.setattr(lc, "compile_tikz_to_png",
                        lambda c, title="", **kw: by_dpi[300])
    assert lc.compile_tikz_to_png_adaptive(code) == by_dpi[300]


def test_adaptive_dpi_floor(monkeypatch):
    """阶梯耗尽仍超限 → 返回最小（最低 DPI）PNG，宁降清晰度保显示。"""
    import utils.latex_compile as lc

    by_dpi = {300: b"A" * 5000, 200: b"A" * 4000, 150: b"A" * 3000,
              100: b"A" * 2000, 75: b"A" * 1500}
    monkeypatch.setattr(lc, "_find_latex_engine", lambda: "xelatex")
    monkeypatch.setattr(lc, "_run_latex",
                        lambda engine, tex, flags: tex + ".pdf")
    monkeypatch.setattr(lc, "_pdf_to_png_pymupdf",
                        lambda pdf, dpi: by_dpi.get(dpi))
    out = lc.compile_tikz_to_png_adaptive(
        "\\begin{tikzpicture}\\end{tikzpicture}", max_kb=1)   # 全部 > 1KB
    assert out == by_dpi[75]


def test_render_image_budget(monkeypatch, tmp_path):
    """MCP_IMAGE_MAX_KB 保险压缩：widget 交付降 DPI 后的图，字节随之变小。"""
    import base64 as _b64
    import utils.latex_compile as lc

    by_dpi = {300: b"A" * 5000, 200: b"A" * 3000, 150: b"A" * 1500,
              100: b"A" * 800, 75: b"A" * 400}
    monkeypatch.setattr(lc, "_find_latex_engine", lambda: "xelatex")
    monkeypatch.setattr(lc, "_run_latex",
                        lambda engine, tex, flags: tex + ".pdf")
    monkeypatch.setattr(lc, "_pdf_to_png_pymupdf",
                        lambda pdf, dpi: by_dpi.get(dpi))
    monkeypatch.setenv("MCP_IMAGE_MAX_KB", "1")
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "render_chemistry", {"text": "[STRUCT:c1ccccc1]"})
        fig = out["structuredContent"]["figures"][0]
        assert fig["size_bytes"] == len(by_dpi[100])
        uri = out["structuredContent"]["widget_uri"]
        rr = _rpc(client, "resources/read",
                  {"uri": uri}, id_=9).json()["result"]
        embedded = _b64.b64decode(
            rr["contents"][0]["text"].split("base64,", 1)[1].split('"', 1)[0])
        assert embedded == by_dpi[100]


def test_test_image(monkeypatch, tmp_path):
    """A′ 补充实验：纯 image block 工具（无 structuredContent）——排除
    "structuredContent 存在时宿主丢弃 content[]" 的混淆变量。"""
    png = b"\x89PNG\r\n\x1a\n" + b"t" * 32
    (tmp_path / "mcp_attachments").mkdir(parents=True)
    (tmp_path / "mcp_attachments" / "test.png").write_bytes(png)
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "test_image", {})
        # 纯 image 交付：无 structuredContent、无 meta
        assert out.get("structuredContent") in (None, {})
        assert not (out.get("_meta") or out.get("meta"))
        blocks = [c for c in out["content"] if c.get("type") == "image"]
        assert len(blocks) == 1
        assert blocks[0]["mimeType"] == "image/png"
        import base64 as _b64
        assert _b64.b64decode(blocks[0]["data"]) == png


def test_test_image_content(monkeypatch, tmp_path):
    """实际 PNG 经 MCP 序列化后仍完整，且不夹带 URL、组件或结构化内容。"""
    import base64
    import io
    from pathlib import Path

    from PIL import Image

    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        tools = _rpc(client, "tools/list").json()["result"]["tools"]
        tool = next(t for t in tools if t["name"] == "test_image_content")
        assert not tool.get("outputSchema")
        assert not tool.get("_meta")
        out = _call(client, "test_image_content", {})
        assert not out.get("isError")
        assert out.get("structuredContent") is None
        assert not out.get("_meta")
        assert len(out["content"]) == 1
        block = out["content"][0]
        assert block["type"] == "image"
        assert block["mimeType"] == "image/png"
        png = base64.b64decode(block["data"], validate=True)
        assert png == Path(mcp_api.__file__).with_name("test_image.png").read_bytes()
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(png) < 100 * 1024
        with Image.open(io.BytesIO(png)) as image:
            assert image.format == "PNG"
            assert image.mode in ("RGB", "RGBA")
            assert min(image.size) > 1
            image.load()


def test_test_image_fallback(monkeypatch, tmp_path):
    """test.png 缺失 → 内嵌生成的兜底图（工具永不失败）。"""
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "test_image", {}, id_=2)
        blocks = [c for c in out["content"] if c.get("type") == "image"]
        assert len(blocks) == 1
        import base64 as _b64
        assert _b64.b64decode(blocks[0]["data"]).startswith(b"\x89PNG")


def test_render_image_block_cap(monkeypatch, tmp_path):
    """MCP_IMAGE_BLOCK_KB：超阈值图不附 image block（防响应预算截断）。"""
    big = b"\x89PNG\r\n\x1a\n" + b"B" * (128 * 1024)          # 128KB > 64KB
    monkeypatch.setattr("utils.latex_compile.compile_tikz_to_png",
                        lambda code, title="", **kw: big)
    with _mount(monkeypatch, tmp_path) as client:
        _init(client)
        out = _call(client, "render_chemistry", {"text": "[STRUCT:c1ccccc1]"})
        assert out["structuredContent"]["figures"][0]["size_bytes"] == len(big)
        assert not [c for c in out["content"] if c.get("type") == "image"]
        assert any(c.get("type") == "text" for c in out["content"])


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
    monkeypatch.setattr("utils.latex_compile.compile_tikz_to_png",
                        lambda code, title="", **kw: None)
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
