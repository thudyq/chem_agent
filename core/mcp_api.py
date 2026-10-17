# -*- coding: utf-8 -*-
"""core/mcp_api.py — ChatGPT（MCP）工具层：无 LLM 的化学渲染能力即服务。

分工（对应 README 的管线拆分）：ChatGPT 当大脑（理解、推理、决定何时调用、
组织回答）；本模块只做确定性工作——SMILES 验证、名称→SMILES、标记解析 →
契约校验 → 渲染 → PNG 托管。"失败回传修正"闭环在 MCP 语境下由 ChatGPT
依据 render_chemistry 返回的结构化错误自行完成（等价于 app.py 把失败清单
回传 LLM 修正，只是大脑换成了 ChatGPT）。全程**零 LLM 依赖**，所有工具
均为只读（readOnlyHint）。

- 挂载：api.py 启动时调用 mount_mcp(app)；mcp 包未安装则跳过并打警告，
  绝不拖垮 /v1 与网页（fastapi 路径不依赖本模块）。
- 传输：Streamable HTTP，stateless + JSON 响应（不用 SSE 帧——避开 Nginx
  `proxy_buffering` 这类流式反代坑，见 docs/deploy.md §2.3）。
- 鉴权：环境变量 MCP_API_KEY 设置后，所有 /mcp 请求必须带
  `Authorization: Bearer <MCP_API_KEY>`（常量时间比较）；未设置 = 无鉴权，
  仅供本机/内测，启动横幅明确警告。密钥只在内存比对，不落日志。
- 附件：PNG 落在独立目录 data/mcp_attachments/（与 /v1、网页**各占一份**
  配额、互不挤占——理由与 core/web_api.py 同款：共用会让一方的滚动回收
  删掉另一方的热链图），经 /mcp/files/{name} 下载（挂在 MCP 子应用内，
  自动吃到同一道 Bearer 闸门）。
- 不写 diaglog 诊断日志：那是面向 BYOK/清小搭真人提问的隐私边界（见
  docs/deploy.md §4）；MCP 的校验/渲染错误直接结构化回传 ChatGPT 可见，
  服务端只打 [mcp] 前缀的运行日志。
- DNS rebinding 防护保持开启：Host 白名单 = 本机回环 + PUBLIC_BASE_URL
  的主机名 + MCP_ALLOWED_HOSTS（逗号分隔补充）。
"""

import base64
import hashlib
import hmac
import html
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse

from starlette.responses import JSONResponse

from app import (_RENDER_ERROR_PREFIX,
                 _partial_render_composite_without_mecharrows)
from core.attachments import _prune_attachments, extract_code_blocks
from core.config import settings
from core.tag_injector import inject_tags_into_text
from core.tag_parser import parse_tags
from core.tag_validator import (autofix_balance_gap, autofix_mech_bond_endpoint,
                                autofix_stereo_label, degrade_text_friendly,
                                validate_tags)
from renderers.registry import render_tag

try:                                    # mcp 是可选依赖：未安装时 /v1 与网页照常
    from mcp.server import MCPServer
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.types import (CallToolResult, ImageContent, TextContent,
                           ToolAnnotations)
    _MCP_AVAILABLE = True
except ImportError:                     # pragma: no cover - 环境差异路径
    _MCP_AVAILABLE = False

_MCP_DIR = settings.project_root / "data" / "mcp_attachments"
# 与 /v1、网页同口径的独立配额（docs/deploy.md §4 附件目录三条目）
MCP_ATTACHMENT_MAX_BYTES = 2 * 1024 * 1024 * 1024      # 2GB
MCP_ATTACHMENT_MAX_FILES = 50000

# ---------------------------------------------------------------- Widget
# 图以 data URI 内嵌进 widget HTML，经 resources/read（JSON-RPC，走隧道）
# 交付给 ChatGPT 的 MCP Apps iframe；字节**不进** structuredContent（那是
# 模型可见文本，base64 会撑爆上下文）。iframe 默认 CSP 为 MCP Apps 规范的
# restrictive default（img-src 'self' data:），内嵌图默认可渲染，无需声明
# 外部域——也正因 iframe 在 oaiusercontent 沙箱源上，本地/隧道地址永远
# 不可达，data URI 是零托管下唯一取图方式。
# OpenAI 官方 Python 示例（kitchen_sink_server_python）的组件 MIME 口径；
# MCP Apps 规范的 text/html;profile=mcp-app 与之并存，ChatGPT 宿主按
# 此标记识别"这是一个 app 组件"。
_WIDGET_MIME = "text/html+skybridge"
WIDGET_URI_TEMPLATE = "ui://chem-agent/render/{render_id}.html"
WIDGET_FALLBACK_URI = "ui://chem-agent/render.html"   # 工具定义级静态兜底
_WIDGET_KEEP = 128                     # 内存 widget 上限（FIFO，超配额淘汰）
_widgets: dict[str, str] = {}
_widget_order: list[str] = []

# 输入上限（防滥用；与 web_api 的口径对齐）
_MAX_TEXT_CHARS = 8000
_MAX_TAGS = 12
_MAX_SMILES_CHARS = 2000
_MAX_NAME_CHARS = 200

# 附件文件名白名单（uuid hex + .png，防路径穿越；与 web_api._PNG_RE 同式）
_PNG_RE = re.compile(r"^[0-9a-f]{32}\.png$")

_mcp = None                             # MCPServer 实例（mount_mcp 成功后非 None）


# ---------------------------------------------------------------- 工具结果契约
# TypedDict 会自动生成 outputSchema，ChatGPT 按 structuredContent 消费；
# 字段即契约：改名 = 破坏调用方，需同步 docs/deploy.md 的 MCP 章节。

class FigureInfo(TypedDict):
    index: int                          # 第几张图（1 起，与 injected_text 中编号一致）
    kind: str                           # STRUCT / COMPOSITE / ENERGY
    label: str                          # 结构标注（STRUCT 的 label；可为空串）
    size_bytes: int                     # PNG 字节数（仅元数据，字节本体走 widget）
    url: str | None                     # 公网 PNG 地址；仅配置 MCP_PUBLIC_BASE_URL 时非空


class TagError(TypedDict):
    tag: str                            # 标记原文（前 120 字符，定位用）
    kind: str                           # 标记类型
    reason: str                         # 失败原因——回传修正的依据（给 ChatGPT 看）
    hint: str                           # 友好降级文案（可直接展示给最终用户）


class RenderResult(TypedDict):
    figures: list                        # list[FigureInfo]
    injected_text: str                   # 标记已替换为图片引用/（图 N）/降级说明的文本
    errors: list                         # list[TagError]
    tikz_sources: list                   # list[str]；include_tikz=True 时为全部 TikZ 源码
    widget_uri: str | None               # 图示组件地址（ui://…，与结果 _meta.resourceUri 一致）
    note: str                            # 补充说明（如"未发现标记"），可为空串


class ValidateSmilesResult(TypedDict):
    valid: bool
    canonical_smiles: str                # RDKit 规范化后的 SMILES；非法为空串
    formula: str                         # 分子式；含 R/Ar 等通用基团时为空串
    molecular_weight: float | None       # 分子量；含通用基团时为 None
    note: str                            # 规范化/失败原因说明


class ResolveNameResult(TypedDict):
    name: str
    smiles: str                          # 未命中为空串
    found: bool


class SyntaxResult(TypedDict):
    syntax_reference: str                # 标记语法规范（prompts/system_prompt.txt 前段）
    examples: str                        # 完整示例段（include_examples=True 时非空）


# ---------------------------------------------------------------- 工具实现
# 纯函数（便于单测直调）；MCPServer 注册在 _build_server 里包一层。

def _name_to_smiles(name: str) -> str | None:
    """间接层：便于测试 monkeypatch（PubChem 是外网依赖）。"""
    from utils.name_resolver import name_to_smiles
    return name_to_smiles(name)


def _mcp_public_base() -> str:
    """MCP 专用公网前缀（动态读取，便于测试）。

    与 PUBLIC_BASE_URL 分离：隧道模式下 PUBLIC_BASE_URL 指向的地址对
    ChatGPT/用户皆不可达，拼出来的 URL 是死链——**不配置本变量就不返回
    URL**，图一律走 widget；仅当服务自身跑在公网机器上（如云服务器部署）
    时才配置它。
    """
    return (os.environ.get("MCP_PUBLIC_BASE_URL") or "").strip().rstrip("/")


def _image_max_kb() -> int | None:
    """保险压缩阈值（动态读取）：None = 不控制尺寸（300dpi 原样）。"""
    raw = (os.environ.get("MCP_IMAGE_MAX_KB") or "").strip()
    if not raw:
        return None
    try:
        val = int(raw)
    except ValueError:
        return None
    return val if val > 0 else None


def _image_block_kb() -> int | None:
    """图像块阈值（动态读取）：单图 PNG ≤ 此值（KB）才随结果附 image
    content block（A′：复刻 2026-09 社区成功报告的形态——当时图以 image
    block 进 content[]，连接器为其渲染了 widget）。超过阈值的图不附
    （base64 全量进结果有触发"工具响应预算"截断的风险）。0 = 关闭。"""
    raw = (os.environ.get("MCP_IMAGE_BLOCK_KB") or "64").strip()
    try:
        val = int(raw)
    except ValueError:
        return 64
    return val if val > 0 else None


def _auto_open() -> bool:
    """渲染完成后是否自动用系统默认程序弹出 PNG（B′：本机即时看图）。"""
    return (os.environ.get("MCP_AUTO_OPEN") or "").strip().lower() in (
        "1", "true", "yes")


def _startfile(path: str) -> None:
    """系统默认程序打开文件。os.startfile 仅 Windows 存在；其它平台跳过
    （云服务器部署下本就没有桌面会话，静默跳过是正确语义）。"""
    opener = getattr(os, "startfile", None)
    if opener is None:
        print(f"[mcp] 自动打开跳过（非 Windows 平台）: {path}")
        return
    try:
        opener(path)
    except OSError as e:
        print(f"[mcp] 自动打开失败: {e}")


def _store_widget(render_id: str, html_doc: str) -> None:
    _widgets[render_id] = html_doc
    _widget_order.append(render_id)
    while len(_widget_order) > _WIDGET_KEEP:
        old = _widget_order.pop(0)
        _widgets.pop(old, None)


def _widget_uri(render_id: str) -> str:
    return WIDGET_URI_TEMPLATE.format(render_id=render_id)


def _widget_html(items: list[tuple[str, str]]) -> str:
    """图项 [(b64, caption)] → 单页 widget HTML（全部图内嵌 data URI）。

    不声明 _meta.ui.csp：MCP Apps 规范的 restrictive default 已含
    img-src 'self' data:，正是本组件所需；显式声明反而易画蛇添足。
    尾部附 bridge 诊断条（A 实验：探测宿主桥接能力，结果以文字显示在
    组件内——用户截图后经视觉链路读回，精确定位断点）。
    """
    figs = []
    for i, (b64, label) in enumerate(items, 1):
        cap = html.escape(label) if label else f"化学图示-{i}"
        figs.append(
            f'<figure><img src="data:image/png;base64,{b64}" '
            f'alt="{cap}"><figcaption>{cap}</figcaption></figure>')
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<style>body{margin:0;padding:8px;display:flex;flex-direction:"
            "column;gap:10px;align-items:center;background:#fff}"
            "figure{margin:0;max-width:100%}"
            "img{max-width:100%;height:auto;border:1px solid #eee}"
            "figcaption{font-size:12px;color:#666;text-align:center}"
            "#bridgeDiag{font:11px/1.5 monospace;color:#999;align-self:"
            "flex-start;white-space:pre-wrap}</style></head><body>"
            + "".join(figs)
            + "<div id='bridgeDiag'>bridge诊断 diag-v2</div>"
            + _BRIDGE_PROBE_JS
            + "</body></html>")


# 桥接探针（A 实验）：探测 window.openai（ChatGPT 原生桥）、宿主源、
# MCP Apps 标准 postMessage 握手（ui/initialize）与宿主消息；结果逐行
# 写入 #bridgeDiag。纯探测不改状态，实验期临时存在。
_BRIDGE_PROBE_JS = (
    "<script>(function(){"
    "var out=document.getElementById('bridgeDiag');"
    "var lines=['bridge诊断 diag-v2'];"
    "function line(t){lines.push(t);out.textContent=lines.join('\\n');}"
    "try{line('openai API: '+(typeof window.openai!=='undefined'?"
    "'存在['+Object.keys(window.openai).slice(0,10).join(',')+']':'无'));"
    "}catch(e){line('openai 探测异常: '+e.message);}"
    "try{var ao=window.location.ancestorOrigins;"
    "line('宿主源: '+(ao&&ao.length?Array.prototype.join.call(ao,' '):'未知'));"
    "}catch(e){}"
    "var got=false;"
    "window.addEventListener('message',function(ev){"
    "try{var d=typeof ev.data==='string'?JSON.parse(ev.data):ev.data;"
    "var m=d&&(d.method||(d.result!==undefined?'响应:id'+d.id:''));"
    "if(m){got=true;line('宿主消息: '+m);}}catch(e){}});"
    "try{window.parent.postMessage(JSON.stringify({jsonrpc:'2.0',id:1,"
    "method:'ui/initialize',params:{protocolVersion:'1.0',"
    "appInfo:{name:'chem-agent',version:'1.0.0'},capabilities:{}}}),'*');"
    "line('ui/initialize 已发送');"
    "}catch(e){line('postMessage 失败: '+e.message);}"
    "setTimeout(function(){if(!got)line('8s 无宿主消息');},8000);"
    "})();</script>"
)


def tool_validate_smiles(smiles: str) -> ValidateSmilesResult:
    """RDKit 校验 SMILES，返回合法性 + 规范形式 + 分子式/分子量。"""
    smiles = (smiles or "").strip()
    if not smiles:
        return {"valid": False, "canonical_smiles": "", "formula": "",
                "molecular_weight": None, "note": "SMILES 为空"}
    if len(smiles) > _MAX_SMILES_CHARS:
        return {"valid": False, "canonical_smiles": "", "formula": "",
                "molecular_weight": None,
                "note": f"SMILES 超长（>{_MAX_SMILES_CHARS} 字符）"}
    from rdkit import Chem
    from rdkit.Chem.Descriptors import MolWt
    from utils.rdkit_utils import (expand_group_abbrevs, mol_to_formula,
                                   normalize_h_prefix_smiles, parse_smiles)
    # 与校验层/渲染层同口径的预处理：[H3O+]→[OH3+]、R/Ph/Ac 等缩写展开
    norm, abbrs = expand_group_abbrevs(normalize_h_prefix_smiles(smiles))
    mol = parse_smiles(norm)
    if mol is None:
        return {"valid": False, "canonical_smiles": "", "formula": "",
                "molecular_weight": None,
                "note": "SMILES 解析失败（请检查原子/环/电荷语法；"
                        "H3O+ 这类氢前缀与 R/Ph 等缩写已自动规范化）"}
    note = ""
    if abbrs:
        note = "通用基团缩写已展开为占位原子：" + "、".join(abbrs.values())
    has_dummy = any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
    formula = "" if has_dummy else mol_to_formula(norm)
    mw = None
    if not has_dummy:
        mw = round(MolWt(mol), 2)
    return {"valid": True, "canonical_smiles": Chem.MolToSmiles(mol),
            "formula": formula, "molecular_weight": mw, "note": note}


def tool_resolve_chemical_name(name: str) -> ResolveNameResult:
    """化学名称 → SMILES（PubChem 主路径，内置超时与负缓存）。"""
    name = (name or "").strip()
    if not name:
        return {"name": "", "smiles": "", "found": False}
    if len(name) > _MAX_NAME_CHARS:
        return {"name": name[:_MAX_NAME_CHARS], "smiles": "", "found": False}
    try:
        smi = _name_to_smiles(name)
    except Exception as e:              # 网络层异常也不抛出——工具失败要结构化
        print(f"[mcp] name_to_smiles 异常：{e}")
        smi = None
    return {"name": name, "smiles": smi or "", "found": bool(smi)}


def tool_render_chemistry(text: str, include_tikz: bool = False) -> RenderResult:
    """渲染含标记的化学文本 → PNG 图（URL）+ 结构化错误（供修正回传）。

    流程与 app.py 的管线一致，但**不含 LLM 环节**：解析 → 契约校验 →
    确定性自动修复（一轮，不经 LLM）→ 逐标记渲染 → 注入 → TikZ 编译为
    PNG 并托管。校验/渲染/编译失败都不抛异常，以 errors 结构化返回。
    """
    text = text or ""
    if not text.strip():
        return {"figures": [], "injected_text": "", "errors": [],
                "tikz_sources": [], "widget_uri": None, "note": "输入为空"}
    if len(text) > _MAX_TEXT_CHARS:
        return {"figures": [], "injected_text": "",
                "errors": [{"tag": "", "kind": "INPUT",
                            "reason": f"文本超长（>{_MAX_TEXT_CHARS} 字符）",
                            "hint": "请缩短后重试"}],
                "tikz_sources": [], "widget_uri": None, "note": ""}

    tags = parse_tags(text)
    if not tags:
        return {"figures": [], "injected_text": text, "errors": [],
                "tikz_sources": [], "widget_uri": None,
                "note": "未发现渲染标记：请先用 get_tag_syntax 学习标记语法，"
                        "把结构/反应/势能面写成 [STRUCT]/[COMPOSITE]/[ENERGY] "
                        "标记后再调用本工具"}
    if len(tags) > _MAX_TAGS:
        return {"figures": [], "injected_text": text,
                "errors": [{"tag": "", "kind": "INPUT",
                            "reason": f"标记数超上限（{len(tags)} > {_MAX_TAGS}）",
                            "hint": "请拆分为多次调用"}],
                "tikz_sources": [], "widget_uri": None, "note": ""}

    # 契约校验 + 一轮确定性自动修复（与 app.py::_generate_inner 同款三级
    # autofix：机理箭头端点 → 立体标注枚举 → 守恒缺口；均不经 LLM）
    valid_tags, invalid = validate_tags(tags)
    if invalid:
        for r in invalid:
            fix = autofix_mech_bond_endpoint(r.tag)
            if fix is None:
                fix = autofix_stereo_label(r.tag)
            if fix is None and "不守恒" in (r.reason or ""):
                fix = autofix_balance_gap(r.tag, r.reason)
            if fix is None or r.tag.raw not in text:
                continue
            new_raw, _note = fix
            _, bad = validate_tags(parse_tags(new_raw))
            if bad:
                continue
            text = text.replace(r.tag.raw, new_raw, 1)
            print(f"[mcp] 端点自动修复生效：{_note}")
        tags = parse_tags(text)
        valid_tags, invalid = validate_tags(tags)

    # 校验失败降级：COMPOSITE 仅 MECHARROW 报错时剔除箭头重渲染（骨架保留）
    degraded = {}
    for r in invalid:
        partial = None
        if r.tag.type == "COMPOSITE" and "MECHARROW" in (r.reason or ""):
            partial = _partial_render_composite_without_mecharrows(r.tag)
        if partial is not None:
            degraded[r.tag.raw] = (partial + "\n\n> 反应箭头无法渲染，已省略"
                                   "；机理电子流向请以文字说明为准")
        else:
            degraded[r.tag.raw] = degrade_text_friendly(r.tag)

    # 逐标记渲染（REASONING 无渲染器，注入器会整体剥离）
    rendered, failures = {}, []
    for tag in valid_tags:
        if tag.type == "REASONING":
            continue
        try:
            out = render_tag(tag)
        except Exception as e:
            out = f"（{tag.type} 渲染失败：{e}）"
        if out is None:
            continue                    # 未注册类型，注入时保留原标记
        if out.startswith(_RENDER_ERROR_PREFIX):
            failures.append((tag, out))
        else:
            rendered[tag.raw] = out

    final_map = dict(rendered)
    final_map.update(degraded)
    for tag, err in failures:
        final_map.setdefault(tag.raw, err)
    injected = inject_tags_into_text(text, tags, final_map)

    # 错误清单：校验失败 + 渲染失败（结构化回传 ChatGPT 修正）
    errors = []
    for r in invalid:
        errors.append({"tag": r.tag.raw[:120], "kind": r.tag.type,
                       "reason": r.reason or "校验失败",
                       "hint": degrade_text_friendly(r.tag)})
    for tag, err in failures:
        errors.append({"tag": tag.raw[:120], "kind": tag.type,
                       "reason": err, "hint": err})

    # TikZ 代码块 → PNG（独立附件池，配额互不挤占）。交付形态：
    #   * 默认（隧道模式）：图以 data URI 进 widget（resources/read 穿隧道），
    #     **不返回 URL**——PUBLIC_BASE_URL 指向的地址在隧道模式下无人可达，
    #     返回它只会制造死链（实测踩到）；
    #   * 显式配置 MCP_PUBLIC_BASE_URL（服务本身跑在公网机器上）：额外返回
    #     /mcp/files 下载 URL，模型可写进回答。
    # 两种模式都落盘 data/mcp_attachments，便于运维检查与手动取用。
    blocks = extract_code_blocks(injected)
    figures: list = []
    widget_items: list[tuple[str, str]] = []
    written: list[str] = []
    base = _mcp_public_base()
    if blocks:
        # 块归属：按解析顺序，替换文本含代码块的标记依次对应各块
        owners = []
        for tag in tags:
            rep = final_map.get(tag.raw)
            if rep:
                for code in extract_code_blocks(rep):
                    label = tag.args[1] if (
                            tag.type == "STRUCT" and len(tag.args) > 1
                            and isinstance(tag.args[1], str)) else ""
                    owners.append((tag.type, label, code))

        max_kb = _image_max_kb()
        from utils.latex_compile import compile_tikz_to_png_adaptive
        pngs: list = [None] * len(owners)
        if len(owners) > 1:
            with ThreadPoolExecutor(max_workers=min(len(owners), 3)) as ex:
                futs = {i: ex.submit(compile_tikz_to_png_adaptive, code, "",
                                     max_kb)
                        for i, (_, _, code) in enumerate(owners)}
                for i, fut in futs.items():
                    try:
                        pngs[i] = fut.result()
                    except Exception as e:  # 单块编译异常不拖垮整轮
                        print(f"[mcp] 图 {i + 1} 编译异常，作失败块: {e}")
        elif owners:
            try:
                pngs[0] = compile_tikz_to_png_adaptive(owners[0][2], "",
                                                       max_kb)
            except Exception as e:
                print(f"[mcp] 图 1 编译异常，作失败块: {e}")

        _MCP_DIR.mkdir(parents=True, exist_ok=True)
        urls: list = [None] * len(blocks)
        for i, (kind, label, _code) in enumerate(owners):
            png = pngs[i]
            if not png:
                errors.append({"tag": (blocks[i][:80] if i < len(blocks) else ""),
                               "kind": kind,
                               "reason": "LaTeX 编译失败（服务器缺 xelatex/"
                                         "poppler-utils，或源码闸门拦截）",
                               "hint": "（图示未能渲染）"})
                continue
            widget_items.append((base64.b64encode(png).decode("ascii"), label))
            figures.append({"index": i + 1, "kind": kind, "label": label,
                            "size_bytes": len(png), "url": None})
            # 两种模式都落盘（运维检查/手动取用）——widget 模式只是不拼 URL
            name = f"{uuid.uuid4().hex}.png"
            (_MCP_DIR / name).write_bytes(png)
            written.append(name)
            if base:
                urls[i] = f"{base}/mcp/files/{name}"
                figures[-1]["url"] = urls[i]
        if written:
            _prune_attachments(_MCP_DIR, MCP_ATTACHMENT_MAX_BYTES,
                               MCP_ATTACHMENT_MAX_FILES,
                               keep=frozenset(written))
        if _auto_open():
            for name in written:
                _startfile(str(_MCP_DIR / name))

        # 注入文本替换：URL 模式 → 行内 markdown 图（模型可直接引用）；
        # widget 模式 → （图 N）占位（真正的图由 widget 组件显示）
        for i, code in enumerate(blocks):
            repl = (f"![化学图示-{i + 1}]({urls[i]})" if urls[i]
                    else f"（图 {i + 1}）")
            injected = injected.replace(code, repl, 1)

    # widget：全部图内嵌 data URI，按内容哈希寻址（同输入同 URI → 缓存命中）
    widget_uri = None
    if widget_items:
        html_doc = _widget_html(widget_items)
        render_id = hashlib.sha256(html_doc.encode("utf-8")).hexdigest()[:16]
        _store_widget(render_id, html_doc)
        widget_uri = _widget_uri(render_id)

    # 模型可读摘要（content 文本）：官方 workaround——除 structuredContent
    # 外同时返回简短文本，规避连接器对纯 structuredContent 的消化问题
    lines = [f"已渲染 {len(figures)} 张化学图示。"]
    for e in errors[:5]:
        lines.append(f"未通过[{e['kind']}]: {str(e['reason'])[:100]}")
    if len(errors) > 5:
        lines.append(f"…共 {len(errors)} 个未通过项（详见 structuredContent）")
    if widget_uri:
        lines.append(f"图示组件: {widget_uri}")
    if base:
        # B′：显式给出可点击链接（本机/局域网可达），ChatGPT 稳定写进回答
        for fig in figures:
            if fig.get("url"):
                lines.append(f"图示链接[图 {fig['index']}]: {fig['url']}")

    # A′：image content block 随结果返回（2026-09 社区成功报告的形态——
    # 图像块经桥推给 iframe 显示，宿主不应将其计入模型上下文；阈值防
    # 超大图触发响应预算截断）
    content: list = [TextContent(type="text", text="\n".join(lines))]
    block_cap = _image_block_kb()
    if block_cap:
        for b64, _label in widget_items:
            if len(b64) * 3 // 4 <= block_cap * 1024:
                content.append(ImageContent(type="image", data=b64,
                                            mime_type="image/png"))

    result: RenderResult = {"figures": figures, "injected_text": injected,
                            "errors": errors, "tikz_sources": [],
                            "widget_uri": widget_uri, "note": ""}
    if include_tikz:
        # 从**替换前**的注入文本取源码（替换后已是图片引用/占位）；供 Overleaf 编译
        result["tikz_sources"] = list(blocks)

    # 结果级 _meta = 官方 kitchen_sink Python 示例的精确字段集
    # （openai/outputTemplate + widgetAccessible + toolInvocation 文案），
    # 附加 ui.resourceUri（MCP Apps 规范字段，无害）。A″ 实验：此前缺失
    # widgetAccessible 与 invoking/invoked 文案，宿主可能据此不渲染。
    result_meta = None
    if widget_uri:
        result_meta = {
            "openai/outputTemplate": widget_uri,
            "openai/toolInvocation/invoking": "正在渲染化学图示…",
            "openai/toolInvocation/invoked": "图示已生成",
            "openai/widgetAccessible": True,
            "ui": {"resourceUri": widget_uri},
        }

    return CallToolResult(
        content=content,
        structured_content=dict(result),
        meta=result_meta)


def tool_test_image() -> CallToolResult:
    """A′ 补充实验：纯 image content block（**无 structuredContent**）。

    背景：render_chemistry 的 image block 与 structuredContent 同结果返回，
    而 OpenAI Codex 有"structuredContent 存在时丢弃 content[]"的确认行为
    （issue #10334）——ChatGPT 是否同款未知。本工具把 image block 单独
    摘出来测：能显示 → 丢弃行为在，render_chemistry 接入需改结构；
    仍不显示 → ImageContent 通道在 ChatGPT 连接器彻底判死。
    固定读 data/mcp_attachments/test.png（运维可自行替换图片）；缺失时
    用内嵌生成的兜底图，工具永不失败。
    """
    path = _MCP_DIR / "test.png"
    png = path.read_bytes() if path.is_file() else None
    if not png:
        png = _fallback_test_png()
    return CallToolResult(
        content=[ImageContent(type="image",
                              data=base64.b64encode(png).decode("ascii"),
                              mime_type="image/png")],
        structured_content=None,
        meta=None)


def _fallback_test_png() -> bytes:
    """生成肉眼可见的兜底测试图（红框黄底 + 文字）；PyMuPDF 不可用时
    退回 1x1 PNG（工具不失败，但可能不显眼）。"""
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page(width=220, height=160)
        page.draw_rect(fitz.Rect(10, 10, 210, 150),
                       color=(0.85, 0.1, 0.1), fill=(0.95, 0.9, 0.3),
                       width=2)
        page.insert_text((30, 90), "Chem_Agent test image",
                         fontname="helv", fontsize=13)
        pix = page.get_pixmap(dpi=72)
        doc.close()
        return pix.tobytes("png")
    except Exception as e:
        print(f"[mcp] 兜底测试图生成失败，退回 1x1: {e}")
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
            "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def tool_get_tag_syntax(include_examples: bool = False) -> SyntaxResult:
    """返回标记语法规范（单一事实源 = prompts/system_prompt.txt）。

    默认不含示例段（控制响应体积）；include_examples=True 时附带全部
    示例（含原子编号推导示范——复杂机理强烈建议带上）。
    """
    path = settings.project_root / "prompts" / "system_prompt.txt"
    raw = path.read_text(encoding="utf-8")
    m = re.search(r"^七、示例\s*$", raw, re.MULTILINE)
    if m:
        return {"syntax_reference": raw[:m.start()].rstrip(),
                "examples": raw[m.start():] if include_examples else ""}
    return {"syntax_reference": raw, "examples": ""}


# ---------------------------------------------------------------- 服务组装

_INSTRUCTIONS = (
    "Chem_Agent 化学渲染工具集。分工：你负责化学推理与组织回答；本服务负责"
    "把化学图示**画**出来（SMILES 验证、反应式/机理图/势能面渲染）。"
    "典型流程：① 写标记前可先 validate_smiles 核对 SMILES、"
    "resolve_chemical_name 由名称取 SMILES；② 不确定标记语法时调用 "
    "get_tag_syntax（含原子编号推导规则）；③ 把嵌入 [STRUCT]/[COMPOSITE]/"
    "[ENERGY] 标记的文本交给 render_chemistry——图示画面由返回结果中的 "
    "ui:// 组件地址（图示组件）内联展示，结构化数据供你组织回答。"
    "若返回 errors，请按 reason 修正标记后重试——校验失败不会静默出图。"
)

_TOOL_ANNOTATIONS = None
if _MCP_AVAILABLE:
    _TOOL_ANNOTATIONS = ToolAnnotations(read_only_hint=True,
                                        destructive_hint=False,
                                        idempotent_hint=True,
                                        open_world_hint=True)


def _build_server() -> "MCPServer":
    mcp = MCPServer("chem-agent", title="Chem_Agent 化学渲染",
                    instructions=_INSTRUCTIONS, version="1.0.0")

    @mcp.tool(name="validate_smiles",
              description="校验 SMILES 是否合法（RDKit），返回规范化 SMILES、"
                          "分子式与分子量。支持 [H3O+]→[OH3+] 氢前缀规范化与 "
                          "R/Ph/Ac/Me 等通用基团缩写。写标记前用它核对 SMILES。",
              annotations=_TOOL_ANNOTATIONS)
    def validate_smiles(smiles: str) -> ValidateSmilesResult:
        return tool_validate_smiles(smiles)

    @mcp.tool(name="resolve_chemical_name",
              description="化学名称（俗名/IUPAC/中文名）→ 标准 SMILES（PubChem "
                          "兜底）。用户只给了名称没给 SMILES 时先用它换取结构。",
              annotations=_TOOL_ANNOTATIONS)
    def resolve_chemical_name(name: str) -> ResolveNameResult:
        return tool_resolve_chemical_name(name)

    @mcp.tool(name="render_chemistry",
              description="渲染含 Chem_Agent 标记的化学文本为化学图示组件"
                          "（结构式/反应式/机理图/势能面）。输入是嵌入 "
                          "[STRUCT]/[COMPOSITE]/[ENERGY] 等标记的文本（语法"
                          "先用 get_tag_syntax 获取），返回图示元数据、标记"
                          "替换后的文本，以及校验/渲染失败的结构化错误。"
                          "figures[].url 配置了公网/本机地址时为可点击下载"
                          "链接（请原样写进回答供用户点击查看）；为 null "
                          "表示图示由 ui:// 组件与随结果返回的图像块承载。"
                          "有 errors 时请按 reason 修正标记后重新调用。",
              annotations=_TOOL_ANNOTATIONS)
    def render_chemistry(text: str, include_tikz: bool = False) -> RenderResult:
        return tool_render_chemistry(text, include_tikz)

    @mcp.tool(name="get_tag_syntax",
              description="返回 Chem_Agent 渲染标记的完整语法规范（STRUCT 画法"
                          "变体、COMPOSITE 组装、MECHARROW 电子箭头原子编号规则、"
                          "输出前自检）。首次使用 render_chemistry 前必读；复杂"
                          "机理建议 include_examples=true 取回编号推导示例。",
              annotations=_TOOL_ANNOTATIONS)
    def get_tag_syntax(include_examples: bool = False) -> SyntaxResult:
        return tool_get_tag_syntax(include_examples)

    @mcp.tool(name="test_image",
              description="诊断工具：返回一张固定测试图（纯 MCP image 内容块，"
                          "无 structuredContent）。用于验证宿主是否显示 MCP "
                          "原生图片内容。",
              annotations=_TOOL_ANNOTATIONS)
    def test_image():
        return tool_test_image()

    @mcp.tool(name="test_image_content",
              description="实验工具：直接返回固定苯环 PNG 的标准 MCP ImageContent"
                          "（Base64 图片字节），用于测试连接器是否显示图片。",
              annotations=_TOOL_ANNOTATIONS)
    def test_image_content():
        png = Path(__file__).with_name("test_image.png").read_bytes()
        return CallToolResult(content=[ImageContent(
            type="image", mime_type="image/png",
            data=base64.b64encode(png).decode("ascii"))])

    # /mcp/files/{name}：MCP 附件下载。挂在 MCP 子应用内 → 自动吃到同一道
    # Bearer 闸门；白名单正则防路径穿越（与 web_api 同式）。
    @mcp.custom_route("/files/{name}", methods=["GET"])
    async def _serve_mcp_attachment(request):
        name = request.path_params.get("name", "")
        if not _PNG_RE.fullmatch(name):
            return JSONResponse({"error": "not found"}, status_code=404)
        path = _MCP_DIR / name
        if not path.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        from starlette.responses import FileResponse
        return FileResponse(path, media_type="image/png", filename=name)

    # MCP Apps 图示组件：每次 render_chemistry 按内容哈希生成新
    # ui://…/render-<id>.html（缓存键），经 resources/read（JSON-RPC，
    # 走隧道）取回 HTML，内嵌 data URI 图片在 ChatGPT 沙箱 iframe 渲染。
    @mcp.resource(WIDGET_URI_TEMPLATE,
                  mime_type=_WIDGET_MIME,
                  description="Chem_Agent 图示组件（本次渲染的图，内嵌 data URI）")
    def _render_widget(render_id: str) -> str:
        print(f"[mcp] 宿主读取图示组件: render_id={render_id}"
              f"（{'命中' if render_id in _widgets else '未命中内存'}）")
        html_doc = _widgets.get(render_id)
        if html_doc is None:
            from mcp.server.mcpserver.exceptions import ResourceNotFoundError
            raise ResourceNotFoundError(
                f"图示组件不存在或已被回收: {render_id}（请重新调用 "
                f"render_chemistry 生成）")
        return html_doc

    # 工具定义级 resourceUri 的静态兜底：若宿主只认 tools/list 里的地址，
    # 也能渲染出一个说明页（真实图示仍以结果级 _meta 的动态地址为准）。
    # A 实验版 = 完整桥接诊断页：iframe 是否被渲染、宿主桥有哪些能力，
    # 全部以文字显示在组件内，供截图回读定位断点。
    @mcp.resource(WIDGET_FALLBACK_URI,
                  mime_type=_WIDGET_MIME,
                  description="Chem_Agent 图示组件占位页")
    def _render_widget_fallback() -> str:
        print("[mcp] 宿主读取静态兜底组件（定义级 resourceUri 被消费）")
        return ("<!doctype html><html><head><meta charset='utf-8'>"
                "<style>body{font-family:monospace;font-size:13px;line-height:"
                "1.6;padding:12px;background:#111;color:#7f7;margin:0}"
                "h3{color:#fff;margin:0 0 8px;font-size:14px}</style></head>"
                "<body><h3>Chem_Agent bridge 诊断 diag-v2</h3>"
                "<div id='diag'>初始化…</div>"
                "<script>(function(){"
                "var out=document.getElementById('diag');"
                "var lines=[];"
                "function line(t){lines.push(t);out.textContent=lines.join('\\n');}"
                "line('① iframe 已渲染 ' + new Date().toLocaleTimeString());"
                "line('② 处于 iframe 内: ' + (window.parent !== window));"
                "try{line('③ openai API: '+(typeof window.openai!=='undefined'?"
                "'存在['+Object.keys(window.openai).slice(0,12).join(',')+']':'无'));"
                "}catch(e){line('③ openai 探测异常: '+e.message);}"
                "try{line('③ chatgpt API: '+(typeof window.chatgpt!=='undefined'?"
                "'存在':'无'));}catch(e){}"
                "try{var ao=window.location.ancestorOrigins;"
                "line('④ 宿主源: '+(ao&&ao.length?"
                "Array.prototype.join.call(ao,' '):'未知'));}catch(e){}"
                "var got=false;"
                "window.addEventListener('message',function(ev){"
                "try{var d=typeof ev.data==='string'?JSON.parse(ev.data):ev.data;"
                "var m=d&&(d.method||(d.result!==undefined?'响应:id'+d.id:''));"
                "if(m){got=true;line('⑤ 宿主消息: '+m);}}catch(e){}});"
                "try{window.parent.postMessage(JSON.stringify({jsonrpc:'2.0',id:1,"
                "method:'ui/initialize',params:{protocolVersion:'1.0',"
                "appInfo:{name:'chem-agent',version:'1.0.0'},"
                "capabilities:{}}}),'*');"
                "line('⑤ ui/initialize 已发送（等待宿主响应…）');"
                "}catch(e){line('⑤ postMessage 失败: '+e.message);}"
                "setTimeout(function(){if(!got)"
                "line('⏳ 8s 内无宿主消息——宿主未参与桥接');},8000);"
                "})();</script></body></html>")

    return mcp


def _api_key() -> str:
    """动态读取（而非挂载时快照）：测试可 monkeypatch 环境变量。"""
    return (os.environ.get("MCP_API_KEY") or "").strip()


class _BearerMiddleware:
    """MCP 子应用的 Bearer 闸门：MCP_API_KEY 未设置时放行（内测模式）。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            key = _api_key()
            if key:
                headers = {k.lower(): v for k, v in scope.get("headers") or []}
                auth = headers.get(b"authorization", b"").decode("latin-1")
                if not hmac.compare_digest(auth, f"Bearer {key}"):
                    await JSONResponse({"error": "unauthorized",
                                        "detail": "需要 Authorization: Bearer "
                                                  "<MCP_API_KEY>"},
                                       status_code=401)(scope, receive, send)
                    return
        await self.app(scope, receive, send)


def _transport_security() -> "TransportSecuritySettings":
    """DNS rebinding 防护保持开启；Host 白名单 = 回环 + 公网域名 + 环境补充。

    allowed_origins 传 None：服务端调用方（ChatGPT）不带 Origin 头，不受影响。
    """
    hosts = {"localhost", "127.0.0.1", "0.0.0.0", "testserver", "testserver.local"}
    pub = (settings.service.public_base_url or "").strip()
    if pub:
        host = urlparse(pub).hostname
        if host:
            hosts.add(host)
    extra = (os.environ.get("MCP_ALLOWED_HOSTS") or "").strip()
    if extra:
        hosts |= {h.strip() for h in extra.split(",") if h.strip()}
    hosts.discard("")
    # SDK 的 Host 校验按原始头（含端口）精确匹配：为每个主机名补一条
    # "host:*" 通配端口形态（本地 :8123 调试、非常规端口部署都能过）；
    # rebinding 防护不受影响——攻击者仍须精确持有白名单内的主机名。
    wildcard = {h + ":*" for h in hosts}
    # allowed_origins=[]：无 Origin 头的服务端调用（ChatGPT）照常放行；
    # 带跨源 Origin 的浏览器请求一律 403（DNS rebinding 防护的一部分）
    return TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                     allowed_hosts=sorted(hosts | wildcard),
                                     allowed_origins=[])


class _MCPSlashMiddleware:
    """裸 `/mcp`（无尾斜杠）重写为 `/mcp/` 再进路由。

    为什么必须：Starlette 的 `Mount("/mcp")` 不匹配无尾斜杠的精确路径，
    Router 兜底 307 → `/mcp/`；而 curl `-s`、部分 HTTP 客户端不跟随重定向
    （307 对 POST 虽然语义是"保留方法"，但跟随与否由客户端决定）——表现为
    "请求无输出"。服务端直接消除 307，不依赖任何客户端的重定向策略。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") == "/mcp":
            scope = dict(scope, path="/mcp/")
            if scope.get("raw_path"):
                scope["raw_path"] = b"/mcp/"
        await self.app(scope, receive, send)


def mount_mcp(app) -> bool:
    """把 MCP 子应用挂到主 app 的 /mcp 下；mcp 包缺失时跳过（返回 False）。

    必须在**首个请求前**调用（api.py 模块导入期）；配套的 session manager
    生命周期由 mcp_session() 在 api._lifespan 内进入。
    """
    global _mcp
    if not _MCP_AVAILABLE:
        print("[startup] MCP：未安装 mcp 包，/mcp 未启用"
              "（pip install mcp 后重启即启用）")
        return False
    _mcp = _build_server()
    sub = _mcp.streamable_http_app(
        streamable_http_path="/",       # 挂到 /mcp 前缀后，端点本身用根路径
        stateless_http=True,            # 无会话状态：多 worker/反代友好
        json_response=True,             # JSON 直返，不开 SSE 流（避开反代缓冲坑）
        transport_security=_transport_security())
    sub.add_middleware(_BearerMiddleware)   # 必须在子应用启动前注册
    app.mount("/mcp", sub)
    # 裸 /mcp 兜底：必须在主 app 启动前 add_middleware（api.py 导入期调用满足）
    app.add_middleware(_MCPSlashMiddleware)
    auth_desc = "Bearer(MCP_API_KEY)" if _api_key() else \
        "无——仅限内测，上线前设置 MCP_API_KEY"
    print(f"[startup] MCP：/mcp 已挂载（Streamable HTTP·stateless·JSON·"
          f"鉴权：{auth_desc}）")
    return True


_session_ran: set = set()               # 已 run() 过的 session manager（按实例 id）


@asynccontextmanager
async def mcp_session():
    """MCP session manager 生命周期：api._lifespan 内 `async with` 进入。

    SDK 限定每个 manager 实例只能 run() 一次，而测试的 TestClient 会对同一
    app 反复进出 lifespan——按实例 id 去重：首次进入启动，重入时跳过（重入
    后 /mcp 不再服务，仅测试场景会发生；生产进程 lifespan 只走一轮）。
    未挂载（mcp 包缺失）时为 no-op——绝不影响 /v1 与网页启动。
    """
    if _mcp is None:
        yield
        return
    mgr = _mcp.session_manager
    if id(mgr) in _session_ran:
        yield
        return
    _session_ran.add(id(mgr))
    async with mgr.run():
        yield
