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

import hmac
import os
import re
from contextlib import asynccontextmanager
from typing import TypedDict
from urllib.parse import urlparse

from starlette.responses import JSONResponse

from app import (_RENDER_ERROR_PREFIX,
                 _partial_render_composite_without_mecharrows)
from core.attachments import (build_attachments, extract_code_blocks,
                              replace_code_blocks_with_images)
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
    from mcp.types import ToolAnnotations
    _MCP_AVAILABLE = True
except ImportError:                     # pragma: no cover - 环境差异路径
    _MCP_AVAILABLE = False

_MCP_DIR = settings.project_root / "data" / "mcp_attachments"
# 与 /v1、网页同口径的独立配额（docs/deploy.md §4 附件目录三条目）
MCP_ATTACHMENT_MAX_BYTES = 2 * 1024 * 1024 * 1024      # 2GB
MCP_ATTACHMENT_MAX_FILES = 50000

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
    url: str                            # 公网可下载的 PNG 地址
    label: str                          # 结构标注（STRUCT 的 label；可为空串）


class TagError(TypedDict):
    tag: str                            # 标记原文（前 120 字符，定位用）
    kind: str                           # 标记类型
    reason: str                         # 失败原因——回传修正的依据（给 ChatGPT 看）
    hint: str                           # 友好降级文案（可直接展示给最终用户）


class RenderResult(TypedDict):
    figures: list                        # list[FigureInfo]
    injected_text: str                   # 标记已替换为 markdown 图片/降级说明的文本
    errors: list                         # list[TagError]
    tikz_sources: list                   # list[str]；include_tikz=True 时为全部 TikZ 源码
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
                "tikz_sources": [], "note": "输入为空"}
    if len(text) > _MAX_TEXT_CHARS:
        return {"figures": [], "injected_text": "",
                "errors": [{"tag": "", "kind": "INPUT",
                            "reason": f"文本超长（>{_MAX_TEXT_CHARS} 字符）",
                            "hint": "请缩短后重试"}],
                "tikz_sources": [], "note": ""}

    tags = parse_tags(text)
    if not tags:
        return {"figures": [], "injected_text": text, "errors": [],
                "tikz_sources": [],
                "note": "未发现渲染标记：请先用 get_tag_syntax 学习标记语法，"
                        "把结构/反应/势能面写成 [STRUCT]/[COMPOSITE]/[ENERGY] "
                        "标记后再调用本工具"}
    if len(tags) > _MAX_TAGS:
        return {"figures": [], "injected_text": text,
                "errors": [{"tag": "", "kind": "INPUT",
                            "reason": f"标记数超上限（{len(tags)} > {_MAX_TAGS}）",
                            "hint": "请拆分为多次调用"}],
                "tikz_sources": [], "note": ""}

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

    # TikZ 代码块 → PNG（独立附件池，配额互不挤占）
    blocks = extract_code_blocks(injected)
    figures: list = []
    if blocks:
        base = (settings.service.public_base_url or "").rstrip("/")
        attachments = build_attachments(
            injected, base, dir_path=_MCP_DIR,
            max_bytes=MCP_ATTACHMENT_MAX_BYTES,
            max_files=MCP_ATTACHMENT_MAX_FILES,
            url_prefix="/mcp/files")     # 与上方 custom_route 成对
        # 块归属：按解析顺序，替换文本含代码块的标记依次对应各块
        owners = []
        for tag in tags:
            rep = final_map.get(tag.raw)
            if rep:
                for _ in extract_code_blocks(rep):
                    label = tag.args[1] if (
                            tag.type == "STRUCT" and len(tag.args) > 1
                            and isinstance(tag.args[1], str)) else ""
                    owners.append((tag.type, label))
        for i, att in enumerate(attachments):
            kind, label = owners[i] if i < len(owners) else ("FIGURE", "")
            if att and att.get("fileUrl"):
                figures.append({"index": i + 1, "kind": kind,
                                "url": att["fileUrl"], "label": label})
            else:
                errors.append({"tag": (blocks[i][:80] if i < len(blocks) else ""),
                               "kind": kind,
                               "reason": "LaTeX 编译失败（服务器缺 xelatex/"
                                         "poppler-utils，或源码闸门拦截）",
                               "hint": "（图示未能渲染）"})
        injected = replace_code_blocks_with_images(
            injected, [a["fileUrl"] if a else None for a in attachments])

    result: RenderResult = {"figures": figures, "injected_text": injected,
                            "errors": errors, "tikz_sources": [],
                            "note": ""}
    if include_tikz:
        # 从**替换前**的注入文本取源码（替换后已是图片引用）；供 Overleaf 编译
        result["tikz_sources"] = list(blocks)
    return result


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
    "把化学图示**画**出来（SMILES 验证、反应式/机理图/势能面渲染为 PNG）。"
    "典型流程：① 写标记前可先 validate_smiles 核对 SMILES、"
    "resolve_chemical_name 由名称取 SMILES；② 不确定标记语法时调用 "
    "get_tag_syntax（含原子编号推导规则）；③ 把嵌入 [STRUCT]/[COMPOSITE]/"
    "[ENERGY] 标记的文本交给 render_chemistry，得到 PNG 图 URL 与校验结果。"
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
              description="渲染含 Chem_Agent 标记的化学文本为 PNG 图示。输入是"
                          "嵌入 [STRUCT]/[COMPOSITE]/[ENERGY] 等标记的文本（语法"
                          "先用 get_tag_syntax 获取），返回每张图的公网 PNG URL、"
                          "标记替换后的文本，以及校验/渲染失败的结构化错误——"
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
