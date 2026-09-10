# -*- coding: utf-8 -*-
"""core/web_api.py — 公开网页（BYOK）后端：任何用户填自己的 API Key 即可用。

与清小搭接入层（`api.py` 的 `/v1/*`）**完全分离**，互不影响：

* `/v1/*`   清小搭线上契约（服务端密钥），行为一字不改；
* `/api/*`  本模块——**纯 BYOK**：凭证从请求头进入，只活在本次请求的
  contextvar 里（`core.credentials`），服务端**不存、不落盘、不回显**；
* `/`       自包含单页聊天界面（`web/index.html`）。

端点
----
    GET  /                    单页界面（自包含 HTML）
    GET  /api/web-config      前端需要的服务端公共配置（无敏感信息）
    POST /api/chat            对话：SSE 流式（stream=true）或 JSON（非流式）
    GET  /api/session/{sid}/{name}.png   会话级图示附件下载

凭证请求头
----------
    X-Chem-Api-Key        必填，用户自己的 LLM API Key
    X-Chem-Base-Url       选填，OpenAI 兼容端点（默认 https://api.deepseek.com/v1）
    X-Chem-Model          选填，主模型名（默认 deepseek-flash）
    X-Chem-Fallback-Model 选填，回退模型（主模型思考过长时用）
    X-Chem-Upgrade-Model  选填，升级模型（难题直用）
    X-Chem-Vision-Model   选填，视觉模型（不填则图片输入不可用）
    X-Chem-Thinking       选填，enabled/disabled

安全
----
* 密钥只从**请求头**读取（不走 URL —— 避免落入访问日志/Referer/浏览器历史）；
* `base_url` 做 SSRF 校验（拒绝内网/回环/云元数据地址，见
  `core.credentials.client_host_allowed`）——服务端会带用户密钥请求该地址，
  放任内网等于把服务器当跳板；
* 日志一律经 `credentials.redact()` 打指纹，绝不打印完整密钥；
* 每 IP 限流（默认 20 次/分钟，可配），防脚本刷对话端点。
"""

from __future__ import annotations

import base64
import contextvars
import ipaddress
import json
import queue
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from app import process_question
from core import answer_cache, credentials, diaglog
from core.attachments import build_attachments, replace_code_blocks_with_images
from core.config import settings

router = APIRouter()

# ---------------------------------------------------------------- 常量

_HEADER_KEY = "X-Chem-Api-Key"
_HEADER_BASE = "X-Chem-Base-Url"
_HEADER_MODEL = "X-Chem-Model"
_HEADER_THINKING = "X-Chem-Thinking"          # 思考开关 on/off
_HEADER_EFFORT = "X-Chem-Effort"              # 思考强度 low/medium/high/max
_HEADER_MAX_TOKENS = "X-Chem-Max-Tokens"      # 最大输出上限
_HEADER_VISION = "X-Chem-Vision-Model"
_HEADER_VISION_BASE = "X-Chem-Vision-Base-Url"
_HEADER_VISION_KEY = "X-Chem-Vision-Api-Key"
_HEADER_VISION_THINKING = "X-Chem-Vision-Thinking"
_HEADER_VISION_EFFORT = "X-Chem-Vision-Effort"

# 默认端点/模型：前端设置面板的预填值（用户可改）。服务端 .env 的配置
# **不参与** BYOK 默认值推导——避免把服务器配置静默施加到用户自己的 key。
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-flash"

_WEB_DIR = settings.project_root / "web"
_SESSIONS_DIR = settings.project_root / "data" / "web_sessions"

# 会话 id / 文件名白名单（防路径穿越）
_SID_RE = re.compile(r"^[0-9a-f]{8,64}$")
_PNG_RE = re.compile(r"^[0-9a-f]{32}\.png$")

# 输入上限（防滥用）
_MAX_QUESTION_CHARS = 8000
_MAX_IMAGES = 4
_MAX_HISTORY_ITEMS = 10

# 网页附件的**全局**配额。语义与 /v1（core/attachments.py）完全一致：
# **不按时间删，只在超配额时删最旧的**。差别只在作用域——/v1 是一个扁平目录，
# 配额天然就是全局；网页是"每会话一个目录"，所以必须在 `web_sessions/` 这一层
# 再设一道全局上限，否则总磁盘 = 每会话额度 × 会话数，没有上界（网页端点匿名，
# 点一次"新对话"就多一个目录）。
#
# 与 /v1 **各占一份**、互不挤占：共用会让网页的匿名使用把清小搭热链的图删掉，
# 反之亦然。默认值与 `ServiceConfig.attachment_max_*` 同口径（2GB / 50000）。
WEB_ATTACHMENT_MAX_BYTES = 2 * 1024 * 1024 * 1024      # 2GB
WEB_ATTACHMENT_MAX_FILES = 50000

# 清理时**不动**刚写入的文件：清理器是后台线程，可能和一个正在落盘的请求撞上。
# 这不是"按时间删除"，只是给新文件一道保护窗（超配额时宁超额也不删新图，
# 与 /v1 的 keep 保护同一取舍）。
_PRUNE_MIN_AGE = 10 * 60

# 每 IP 限流（滑动窗口）
RATE_LIMIT_PER_MINUTE = 20

_SSE_HEARTBEAT = 10.0
_ANSWER_CHUNK = 20


# ---------------------------------------------------------------- 限流

_rate_lock = threading.Lock()
_rate_hits: dict = {}


def _rate_limit(ip: str, limit: int | None = None) -> tuple[bool, int]:
    """滑动窗口限流：返回 (是否放行, 建议 Retry-After 秒数)。limit<=0 关闭。

    `limit` 默认在**调用时**从模块常量读取（不能写成默认参数——默认值在函数
    定义时求值，会绑死旧值，导致运行时改配置/测试 monkeypatch 全都不生效）。
    """
    if limit is None:
        limit = RATE_LIMIT_PER_MINUTE
    if limit <= 0:
        return True, 0
    now = time.time()
    with _rate_lock:
        hits = [t for t in _rate_hits.get(ip, ()) if now - t < 60.0]
        if len(hits) >= limit:
            _rate_hits[ip] = hits
            return False, max(1, int(60 - (now - hits[0])) + 1)
        hits.append(now)
        _rate_hits[ip] = hits
        if len(_rate_hits) > 4096:      # 防内存无界增长（清理已空桶）
            for key in [k for k, v in _rate_hits.items() if not v]:
                _rate_hits.pop(key, None)
    return True, 0


def _reset_rate_limit_for_tests() -> None:
    with _rate_lock:
        _rate_hits.clear()


def _client_ip(request: Request) -> str:
    """客户端 IP（优先取反代链首个地址）。"""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    try:
        return ipaddress.ip_address(request.client.host).compressed \
            if request.client else "unknown"
    except ValueError:
        return "unknown"


# ---------------------------------------------------------------- 凭证

# 请求头 → 凭证字段名。**必须显式映射**：`credentials.normalize` 只认
# `api_key` 这类字段名，直接拿 HTTP 头名（X-Chem-Api-Key）当键会被它当未知键
# 静默丢弃 —— 表现为"填了 Key 仍提示缺少 Key"。
# 三列同名同义的登记表见 instructions/Model-Config-Refactor.md 附录 D。
_HEADER_TO_FIELD = {
    _HEADER_KEY: "api_key",
    _HEADER_BASE: "base_url",
    _HEADER_MODEL: "model",
    _HEADER_THINKING: "thinking",
    _HEADER_EFFORT: "effort",
    _HEADER_MAX_TOKENS: "max_tokens",
    _HEADER_VISION: "vision_model",
    _HEADER_VISION_BASE: "vision_base_url",
    _HEADER_VISION_KEY: "vision_api_key",
    _HEADER_VISION_THINKING: "vision_thinking",
    _HEADER_VISION_EFFORT: "vision_effort",
}


def _extract_credentials(headers: dict) -> dict:
    """{请求头名: 值} → 凭证 dict（未提供的项不出现，由 credentials 层决定回退）。"""
    raw = {}
    for header, field in _HEADER_TO_FIELD.items():
        value = headers.get(header)
        if value:
            raw[field] = value
    return credentials.normalize(raw)


def _credentials_from_headers(kwargs: dict) -> dict:
    """由 FastAPI 解析出的 Header 参数构造凭证。

    **不预填视觉凭证**：`credentials.vision_config()` 只在用户**显式**给出
    `X-Chem-Vision-Model` 时才认为视觉可用（`_FIELDS` 注释里的安全约定：
    绝不把服务器 .env 的视觉模型/计费施加到用户请求上）。若在这里顺手把
    主 Key 填进 `vision_api_key`，用户没配视觉模型时也会被判为"已配置"，
    进而在 `settings.vision.model_name` 兜底（很可能是服务器 .env 的视觉
    模型名）下**拿用户 Key 去调一个他并没有指定的模型**——20260830 实测
    踩到（真实网络 401 泄漏进单元测试）。用户单独填了视觉 Key/端点时，
    下面的映射照常带上。
    """
    return _extract_credentials({k: v for k, v in kwargs.items() if v})


def _require_credentials(kwargs: dict) -> dict:
    """校验并返回凭证；缺失/非法直接抛 HTTPException（400）。"""
    creds = _credentials_from_headers(kwargs)
    if not creds.get("api_key"):
        raise HTTPException(
            status_code=400,
            detail=f"缺少 {_HEADER_KEY} 请求头：请先在界面设置里填写你自己的 "
                   f"API Key（本服务为 BYOK —— 服务端不提供共享密钥）")
    if not creds.get("model"):
        creds["model"] = DEFAULT_MODEL
    if not creds.get("base_url"):
        creds["base_url"] = DEFAULT_BASE_URL
    ok, reason = credentials.client_host_allowed(creds.get("base_url", ""))
    if not ok:
        raise HTTPException(status_code=400, detail=f"端点地址不可用：{reason}")
    return creds


# ---------------------------------------------------------------- 会话目录

def _sessions_root() -> Path:
    root = _SESSIONS_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _session_dir(session_id: str) -> Path:
    """会话附件目录（惰性创建）。"""
    d = _sessions_root() / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _within_web_quota(total_bytes: int, total_files: int,
                      max_bytes: int, max_files: int) -> bool:
    if max_bytes > 0 and total_bytes > max_bytes:
        return False
    if max_files > 0 and total_files > max_files:
        return False
    return True


def prune_web_attachments(max_bytes: int = WEB_ATTACHMENT_MAX_BYTES,
                          max_files: int = WEB_ATTACHMENT_MAX_FILES) -> int:
    """按**全局配额**回收网页附件，返回删除的文件数。

    只在超配额时删最旧的（按文件 mtime），**不按时间 TTL 删**——理由：
    对话文字存在浏览器里、是永久的，若图先按时间消失，用户看到的是
    "昨天还好好的图今天裂了"，而且完全无法预期。改成配额后，只有真的顶到
    磁盘上限才开始回收最冷的图（与 /v1 的取舍一致：宁超额，不删新图）。

    与 `_prune_attachments`（/v1）**不能直接复用**：那个只处理单个扁平目录，
    这里是每会话一个目录，必须在 `web_sessions/` 这一层跨目录收集。

    * 先统计全部 `*.png` 的字节数与个数，未超配额直接返回 0（常态路径，不删任何东西）；
    * 超配额 → 按 mtime 升序删，直到两项都达标；`_PRUNE_MIN_AGE` 内的新文件
      一律跳过（可能正被某个请求写入/引用）；
    * 顺手删掉因此变空的会话目录（避免 `web_sessions/` 里堆一堆空目录）。
    """
    root = _SESSIONS_DIR
    if not root.is_dir() or (max_bytes <= 0 and max_files <= 0):
        return 0

    files = []
    total_bytes = 0
    for p in root.rglob("*.png"):
        try:
            if not p.is_file():
                continue
            sz = p.stat().st_size
        except OSError:
            continue
        files.append(p)
        total_bytes += sz

    if _within_web_quota(total_bytes, len(files), max_bytes, max_files):
        return 0

    cutoff = time.time() - _PRUNE_MIN_AGE
    removed, freed = 0, 0
    for p in sorted(files, key=lambda x: x.stat().st_mtime):
        if _within_web_quota(total_bytes, len(files), max_bytes, max_files):
            break
        try:
            if p.stat().st_mtime > cutoff:      # 新文件不删
                continue
            sz = p.stat().st_size
            p.unlink()
        except OSError:
            continue
        files.remove(p)
        total_bytes -= sz
        removed += 1
        freed += sz

    if removed:
        _drop_empty_session_dirs(root)
        print(f"[web] 附件超配额，回收最旧图 {removed} 个（约 {freed // 1024} KB）；"
              f"剩余 {len(files)} 个 / {total_bytes // 1024 // 1024} MB")
    return removed


def _drop_empty_session_dirs(root: Path) -> int:
    """删掉 `web_sessions/` 下的空会话目录（清理后收尾）。"""
    dropped = 0
    try:
        children = list(root.iterdir())
    except OSError:
        return 0
    for child in children:
        try:
            if child.is_dir() and not any(child.iterdir()):
                child.rmdir()
                dropped += 1
        except OSError:
            continue
    return dropped


# ---------------------------------------------------------------- 问题构造

def _extract_question(messages) -> tuple[str, list]:
    """取最后一条 user 消息 → (文本, 图片引用列表)。

    图片引用支持 `data:image/...;base64,...`（前端读文件后内联）与 http(s)
    URL（做 SSRF 校验）。与 api.py 的 /v1 路径口径一致但更窄：网页端
    不需要音频/文件（前端未提供入口），出现时按文本忽略。
    """
    if not isinstance(messages, list):
        return "", []
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content, []
        if isinstance(content, list):
            texts, images = [], []
            for part in content:
                if isinstance(part, str):
                    texts.append(part)
                    continue
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype in ("text", "input_text"):
                    texts.append(part.get("text", ""))
                elif ptype in ("image_url", "input_image", "image"):
                    ref = part.get("image_url") or part.get("url") or ""
                    if isinstance(ref, dict):
                        ref = ref.get("url", "")
                    if ref:
                        images.append(ref)
            return "\n".join(t for t in texts if t), images[:_MAX_IMAGES]
    return "", []


def _extract_history(messages) -> list:
    """多轮历史：user 原文保留、assistant 去锚定摘要（与 /v1 路径同一策略）。

    复用 `api.py` 的实现——同一套"绝不回喂上一版标记"规则，避免网页端
    出现"让模型重画它却照抄上一版错误图"的问题。
    """
    from api import _extract_history as _impl
    return _impl(messages, max_items=_MAX_HISTORY_ITEMS)


_DATA_URL_RE = re.compile(r"data:image/(\w+);base64,(.*)", re.DOTALL)


def _save_upload_image(ref: str, index: int, upload_dir: Path) -> str | None:
    """把一条图片引用落盘为会话上传目录下的独立文件，返回路径；失败 None。

    单独实现（不复用 api.py 的 `_fetch_image_to_temp`）：那个函数写死文件名
    `upload.png`，同请求多图会互相覆盖。这里每图一个 uuid 文件名，支持并发。
    data: base64 直接解码（不发网络请求）；http(s) 走 SSRF 校验后下载。
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    if ref.startswith("data:"):
        m = _DATA_URL_RE.match(ref)
        if not m:
            return None
        ext = "jpg" if m.group(1).lower() in ("jpg", "jpeg") else "png"
        try:
            data = base64.b64decode(m.group(2))
        except Exception:
            return None
        path = upload_dir / f"up{index}-{uuid.uuid4().hex[:8]}.{ext}"
        path.write_bytes(data)
        return str(path)
    if ref.startswith(("http://", "https://")):
        if not _validate_download_url(ref):
            print(f"[web] 拒绝下载图片（SSRF 防护）: {ref[:80]}")
            return None
        import requests
        try:
            resp = requests.get(ref, timeout=20)
        except requests.exceptions.RequestException:
            return None
        if resp.status_code != 200:
            return None
        path = upload_dir / f"up{index}-{uuid.uuid4().hex[:8]}.png"
        path.write_bytes(resp.content)
        return str(path)
    return None


def _build_question(text: str, images: list, session_dir: Path) -> str:
    """文本 + 图片识别结果拼成最终问题（视觉模型可用时）。

    图片落盘到**会话目录的 uploads 子目录**（不是系统 Temp——沙箱/容器里
    系统 Temp 常不可写，且会话目录便于 TTL 清理；独立子目录避免与图示
    PNG 混在一起）；识别完即删原图，不留用户数据。识别失败明确告知主模型，
    不静默丢弃。
    """
    parts = [text] if text else []
    if not images:
        return "\n".join(parts) or "（空消息）"

    vconf = credentials.vision_config()
    if not vconf.is_configured:
        parts.append(
            "（用户上传了图片，但未配置视觉模型（设置面板里的「视觉模型」），"
            "无法识别图片内容。请在回答中提示用户：填写视觉模型后即可识别图片，"
            "或先用文字描述结构）")
        return "\n".join(parts)

    from utils.ocr_utils import describe_image
    upload_dir = session_dir / "uploads"
    for i, ref in enumerate(images, 1):
        path = _save_upload_image(ref, i, upload_dir)
        if not path:
            parts.append(f"（第 {i} 张图片下载/解码失败，已忽略）")
            continue
        desc = describe_image(path)
        try:
            Path(path).unlink(missing_ok=True)   # 识别完即删原图
        except OSError:
            pass
        if desc and desc.get("content"):
            parts.append(f"（用户上传的图片 {i} 的内容（{desc['type']}）："
                         f"{desc['content']}）")
            if desc.get("smiles_ok") is False:
                parts.append(f"（提示：图片 {i} 识别出的结构式无法解析，识别可能"
                             "有误；请在回答中提醒用户核对，必要时请用户用文字描述）")
            elif desc.get("downgraded"):
                parts.append(f"（提示：图片 {i} 识别受限，以上为尽力提取的片段，"
                             "可能不完整；请提醒用户以原图为准）")
            else:
                parts.append(f"（提示：图片 {i} 为视觉模型自动识别，识别可能有误；"
                             "请在回答中提醒用户以原图为准、核对识别内容）")
        else:
            parts.append(f"（用户上传的图片 {i} 识别失败：视觉模型未能理解图片内容。"
                         "请基于文字作答，并提示用户重新上传或用文字描述）")
    return "\n".join(parts)


# ---------------------------------------------------------------- 结果整备

def _prepare_answer(answer: str, session_id: str) -> str:
    """TikZ 代码块 → 编译为 PNG → 就地替换为行内图片引用，返回展示文本。

    会话级附件目录（`data/web_sessions/<sid>/`）+ `/api/session/<sid>/<name>.png`
    下载路由：不同用户的图片互不可见（文件名随机 + 会话隔离），且可随会话
    TTL 清理。编译失败降级为"（图示未能渲染）"，不影响文字。

    ★ 图片 URL 用**相对路径**，**故意不拼 `PUBLIC_BASE_URL`**。网页是浏览器
    直接打开的，相对路径必然解析到"用户此刻正在访问的那个源"，因此：

    * 不会把服务器 `.env` 的公网地址施加给访客——那个地址上既没有这条会话
      路由、也没有这张图（图落盘在本机 `_SESSIONS_DIR`）。20260910 实测：
      本地 `uvicorn` 起服务时所有图示全裂，浏览器按 `.env` 的
      `PUBLIC_BASE_URL=https://60.205.181.60` 去请求
      `https://60.205.181.60/api/session/<sid>/<name>.png`，而那台机器跑的是
      旧版应用（`/api/session/*` 与 `/api/web-config` 均 404），必然加载失败；
    * 反代（Nginx + https）下不会因为推断出的 scheme 是 http 而触发浏览器的
      混合内容拦截——这正是当初引入绝对 URL 想解决的问题，相对路径让它从
      根上不存在。

    注意与 `/v1` 的分工：那条路径的图是**清小搭在服务端抓取**的，必须要绝对
    URL，故 `api.py::_public_base` 保持原样（用 `.env` 的 `PUBLIC_BASE_URL`
    兜底、缺省按请求 Host 推导）。
    """
    answer = answer or ""
    if not answer:
        return ""
    try:
        # public_base 传空串：本路径**不使用** build_attachments 拼出的绝对前缀，
        # 只借用它落盘并返回文件名（见下方 ★ 与函数 docstring）
        attachments = build_attachments(
            answer, "",
            dir_path=_session_dir(session_id),
            max_bytes=200 * 1024 * 1024, max_files=2000)
    except Exception as e:      # 编译异常不拖垮已生成的文字
        print(f"[web] 附件编译异常，降级为无图: {e}")
        return answer
    if not attachments:
        return answer
    urls = []
    for a in attachments:
        if not a:
            urls.append(None)
            continue
        # build_attachments 给的是 {base}/files/<name>：只取文件名，丢弃它拼的
        # 绝对前缀（那前缀来自服务器 .env，见本函数 ★），换成**同源相对**路由
        name = a["fileUrl"].rsplit("/", 1)[-1]
        urls.append(f"/api/session/{session_id}/{name}")
    return replace_code_blocks_with_images(answer, urls)


def _sse_frame(delta: dict, finish: str | None = None,
               error: dict | None = None) -> str:
    chunk = {"object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if error is not None:
        chunk["error"] = error
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------- 路由

@router.get("/api/web-config")
def web_config():
    """前端需要的公共配置（**无任何敏感信息**：服务器密钥/端点不外泄）。"""
    import os
    return {
        "service": "chem_agent-web",
        "byok": True,
        "default_base_url": DEFAULT_BASE_URL,
        "default_model": DEFAULT_MODEL,
        # 思考参数默认值：**只暴露常量**，不读服务器 .env（避免把服务器配置
        # 静默施加到用户自己的 key；`allow_server_defaults=False` 即此约定）
        "default_thinking": "on",
        "default_effort": "low",
        "default_max_tokens": 32768,
        "thinking_choices": ["on", "off"],
        "effort_choices": ["low", "medium", "high", "max"],
        "allow_private_base_url": os.environ.get(
            "WEB_ALLOW_PRIVATE_BASE_URL", "").strip() in ("1", "true", "yes"),
        # 附件配额：网页路径与 /v1 同口径（不按时间删），但作用域是全局
        "attachment_max_bytes": WEB_ATTACHMENT_MAX_BYTES,
        "attachment_max_files": WEB_ATTACHMENT_MAX_FILES,
        "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
        "max_images": _MAX_IMAGES,
        "max_question_chars": _MAX_QUESTION_CHARS,
    }


@router.get("/api/session/{sid}/{name}")
def serve_session_attachment(sid: str, name: str):
    """会话级图示附件下载（会话 id + 随机文件名，不可猜测）。

    严格白名单正则（路径段，无 `/`）防路径穿越；文件不存在返回 404。
    """
    if not _SID_RE.fullmatch(sid) or not _PNG_RE.fullmatch(name):
        raise HTTPException(status_code=404, detail="not found")
    path = _SESSIONS_DIR / sid / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="image/png", filename=name)


@router.get("/web", response_class=HTMLResponse, include_in_schema=False)
@router.get("/chat", response_class=HTMLResponse, include_in_schema=False)
def index():
    """自包含单页界面（单文件、零构建、零 CDN 依赖）。

    同时挂 `/chat`（便于直接发给用户）与 `/web`（稳定别名）。**不动 `/`**
    ——那是清小搭接入的既有 JSON 端点（`{"service": "chem_agent", ...}`），
    保留原样以免影响任何外部探测。
    """
    page = _WEB_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="web/index.html 缺失")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@router.post("/api/chat")
async def chat(request: Request,
               x_chem_api_key: str | None = Header(None),
               x_chem_base_url: str | None = Header(None),
               x_chem_model: str | None = Header(None),
               x_chem_thinking: str | None = Header(None),
               x_chem_effort: str | None = Header(None),
               x_chem_max_tokens: str | None = Header(None),
               x_chem_vision_model: str | None = Header(None),
               x_chem_vision_base_url: str | None = Header(None),
               x_chem_vision_api_key: str | None = Header(None),
               x_chem_vision_thinking: str | None = Header(None),
               x_chem_vision_effort: str | None = Header(None)):
    """对话端点：`stream=true` 走 SSE，否则 JSON。

    凭证来自请求头 → `credentials.user_credentials()` 包住**整个**请求处理
    （含 SSE 的子线程，contextvars 自动继承），管线全程用用户自己的 key。
    """
    allowed, retry_after = _rate_limit(_client_ip(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁，请 {retry_after} 秒后重试",
            headers={"Retry-After": str(retry_after)})

    creds = _require_credentials({
        _HEADER_KEY: x_chem_api_key,
        _HEADER_BASE: x_chem_base_url,
        _HEADER_MODEL: x_chem_model,
        _HEADER_THINKING: x_chem_thinking,
        _HEADER_EFFORT: x_chem_effort,
        _HEADER_MAX_TOKENS: x_chem_max_tokens,
        _HEADER_VISION: x_chem_vision_model,
        _HEADER_VISION_BASE: x_chem_vision_base_url,
        _HEADER_VISION_KEY: x_chem_vision_api_key,
        _HEADER_VISION_THINKING: x_chem_vision_thinking,
        _HEADER_VISION_EFFORT: x_chem_vision_effort,
    })
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体不是合法 JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象")

    stream = body.get("stream", False)
    stream = stream if isinstance(stream, bool) else False

    messages = body.get("messages") or []
    text, images = _extract_question(messages)
    if len(text) > _MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"问题过长（{len(text)} 字符，上限 {_MAX_QUESTION_CHARS}）")

    session_id = str(body.get("session_id") or "").strip().lower()
    if not _SID_RE.fullmatch(session_id):
        # 前端未给或格式非法 → 服务端生成一个（前端回存并复用）
        session_id = uuid.uuid4().hex
    session_dir = _session_dir(session_id)
    history = _extract_history(messages)

    # 整个请求（含后续 SSE 子线程）都在用户凭证作用域内
    with credentials.user_credentials(creds):
        print(f"[web] ip={_client_ip(request)} key={credentials.redact(creds['api_key'])} "
              f"model={creds.get('model')} base={creds.get('base_url')} "
              f"session={session_id[:8]} images={len(images)} "
              f"stream={stream} history={len(history)}")
        question = _build_question(text, images, session_dir)

        if stream:
            # ★ 必须把**凭证 dict 本身**传进生成器：`_sse_stream` 是生成器，
            # 它的函数体要到**被迭代时**才执行——那时本 `with` 块已退出、
            # contextvar 已 reset。若只在外面 set，生成器内（及其 SSE 工作线程）
            # 看到的 `credentials.current()` 是空的 → 模型/端点/密钥全部回退
            # 服务器 `.env`（20260830 实测：网页填 deepseek-flash、日志却是
            # .env 的 gemini-3.7-flash，根因即此）。由生成器自己重新建立作用域。
            return StreamingResponse(
                _sse_stream(question, history, session_id,
                            thinking=_thinking_of(creds),
                            effort=creds.get("effort"),
                            max_tokens=_max_tokens_of(creds),
                            creds=creds),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        diag, responses = [], []
        answer = process_question(question, history=history,
                                  diagnostics=diag, responses=responses,
                                  thinking=_thinking_of(creds),
                                  effort=creds.get("effort"),
                                  max_tokens=_max_tokens_of(creds)) \
            or "（未能生成回答）"
        answer = _polish_answer(answer)
        diaglog.log_request(question, f"web-{session_id[:8]}", diag,
                            responses[-1] if responses else None,
                            credential=credentials.fingerprint())
        display = _prepare_answer(answer, session_id)
        if responses:
            answer_cache.store(display, responses[-1],
                               meta=_diag_meta(diag))
        return JSONResponse({
            "session_id": session_id,
            "content": display,
            "diagnostics": [{"resolved": d.get("resolved"),
                             "type": d.get("type")} for d in diag],
        })


def _diag_meta(diag: list) -> dict:
    """会话去锚定摘要所需的 meta（是否仍有未解决失败 + 原因）。"""
    unresolved = [d for d in (diag or []) if d.get("resolved") is False]
    if not unresolved:
        return {"failed": False, "reason": ""}
    return {"failed": True,
            "reason": "; ".join((d.get("reason") or "")
                                for d in unresolved)[:500]}


def _sse_stream(question: str, history: list, session_id: str,
                thinking: str = None, effort: str = None,
                max_tokens: int = None, creds: dict = None):
    """SSE 帧序列：role → reasoning 心跳 → content 增量 → stop(+session_id)。

    与 `/v1` 的帧序同构（文本先行：文字就绪即发，PNG 编译在 worker 线程
    并行完成），差异只在 stop 帧带回 `session_id` 与 `x_soda` 不再使用。

    ★ `creds`：用户凭证 dict **必须由调用方传入**（不能只在外层 `with
    user_credentials(...)`）。原因：本函数是**生成器**，函数体在被迭代时才执行，
    那时外层 `with` 已退出、contextvar 已 reset——生成器内会读不到凭证，
    从而静默回退服务器 `.env` 的模型/端点/密钥（20260830 实测病例）。

    ★ 但**不要在本函数体里设置凭证**：本函数是生成器，它的每一段都在
    线程池里被独立 `next()`；Starlette 每次迭代都从请求任务上下文**重新拷贝**
    一份 Context，因此在第 1 次 `next()` 里 `set` 的值，到第 N 次 `next()` 里
    已经不存在了。凭证必须在 **`work()` 内部**设置——那个函数整体跑在同一个
    `ctx.run(...)` 里，一次 `set` 对该线程后续全部管线代码（及其 `copy_context()`
    子线程）都可见（20260830 实测：只在生成器里 set，流式仍回退服务器 `.env`
    的 gemini-3.7-flash）。

    同理不能用 `with user_credentials(...)`：`reset` 要求 set/reset 同 Context，
    跨迭代会抛 `ValueError: Token was created in a different Context`。
    """
    yield from _sse_stream_inner(question, history, session_id,
                                 thinking, effort, max_tokens, creds)


def _sse_stream_inner(question: str, history: list, session_id: str,
                      thinking: str = None,
                      effort: str = None, max_tokens: int = None,
                      creds: dict = None):
    """`_sse_stream` 的主体；用户凭证在 `work()` 内落地（见 `_sse_stream`）。"""
    yield _sse_frame({"role": "assistant"})
    yield _sse_frame({"reasoning": "正在思考并绘制化学图示…"})

    answer_q: queue.Queue = queue.Queue(maxsize=1)
    progress_q: queue.Queue = queue.Queue(maxsize=200)
    correction_mark = object()

    def _safe_put(item) -> None:
        try:
            progress_q.put_nowait(item)
        except queue.Full:
            pass

    def work():
        # ★ 凭证必须在这里 set：work() 整体跑在下面那一个 `ctx.run(...)` 里，
        # 上下文只有一个，set 之后本线程内所有管线代码（以及它们用
        # copy_context() 拉起的子线程）都能读到用户凭证。
        credentials.apply_credentials(creds)
        diag, responses = [], []
        try:
            answer = process_question(
                question, history=history,
                progress_callback=progress_q.put,
                correction_callback=lambda: _safe_put(correction_mark),
                diagnostics=diag, responses=responses,
                thinking=thinking, effort=effort, max_tokens=max_tokens)
        except Exception as e:                       # 管线异常兜底为 error 帧
            answer_q.put(e)
            return
        diaglog.log_request(question, f"web-{session_id[:8]}", diag,
                            responses[-1] if responses else None,
                            credential=credentials.fingerprint())
        display = _polish_answer(_prepare_answer(answer or "", session_id))
        if responses:
            answer_cache.store(display, responses[-1], meta=_diag_meta(diag))
        answer_q.put(display)

    # 子线程**不自动继承** contextvars（新线程拿到的是创建时的空 context），
    # 必须显式 copy_context().run() 把当前请求的用户凭证带进工作线程——
    # 否则线程里的管线会退回服务器 .env 配置（BYOK 失效）。
    ctx = contextvars.copy_context()
    threading.Thread(target=lambda: ctx.run(work), daemon=True).start()
    last_flush = time.time()
    while True:
        try:
            answer = answer_q.get_nowait()
            break
        except queue.Empty:
            pass
        correction = False
        while True:
            try:
                p = progress_q.get_nowait()
            except queue.Empty:
                break
            if p is correction_mark:
                correction = True
        if correction:
            yield _sse_frame({"reasoning": "正在修正回答…"})
            last_flush = time.time()
        elif time.time() - last_flush >= _SSE_HEARTBEAT:
            yield _sse_frame({"reasoning": "正在思考并绘制化学图示…"})
            last_flush = time.time()
        else:
            time.sleep(0.2)

    # 收尾排空（保留修正提示）
    correction = False
    while True:
        try:
            p = progress_q.get_nowait()
        except queue.Empty:
            break
        if p is correction_mark:
            correction = True
    if correction:
        yield _sse_frame({"reasoning": "正在修正回答…"})

    if isinstance(answer, Exception):
        yield _sse_frame({}, finish="stop",
                         error={"type": "upstream_error",
                                "message": _friendly_error(str(answer))})
        yield "data: [DONE]\n\n"
        return

    text = answer or "（未能生成回答）"
    for i in range(0, len(text), _ANSWER_CHUNK):
        yield _sse_frame({"content": text[i:i + _ANSWER_CHUNK]})
    yield _sse_frame({"session_id": session_id}, finish="stop")
    yield "data: [DONE]\n\n"


def _friendly_error(raw: str) -> str:
    """把管线异常翻译成用户能看懂的话（不泄漏服务端路径/堆栈）。"""
    low = (raw or "").lower()
    if "401" in low or "invalid_api_key" in low or "unauthorized" in low:
        return "API Key 无效或已过期，请在设置里检查后重试"
    if "404" in low or "model_not_found" in low or "does not exist" in low:
        return "模型名或端点地址不正确（模型不存在），请在设置里核对"
    if "402" in low or "insufficient" in low or "quota" in low or "balance" in low:
        return "账户额度不足或欠费，请检查你的 API 账户"
    if "429" in low or "rate limit" in low:
        return "上游接口限流，请稍后重试"
    if "timeout" in low or "timed out" in low:
        return "上游接口超时，请稍后重试"
    return "生成回答时出现错误，请稍后重试或检查设置"


# 管线内部"调用失败"文案（app.py 生成）——那是给 CLI/.env 场景写的，对
# 网页用户是误导（他会去翻服务器 .env，而问题在自己的设置面板里）。命中即
# 换成面向网页用户的说明。
_PIPELINE_FAIL_TEXT = "（LLM 调用失败，请检查 .env 配置与网络）"
_WEB_FAIL_TEXT = (
    "（未能生成回答：请检查设置里的 API Key、模型名与接口地址是否正确，"
    "以及账户余额是否充足；填错端点或模型时上游会直接拒绝请求。）"
)


def _thinking_of(creds: dict) -> str | None:
    """用户请求头里的思考开关（空 → 由服务端默认决定）。"""
    return (creds.get("thinking") or "").strip() or None


def _max_tokens_of(creds: dict) -> int | None:
    """用户请求头里的最大输出上限（非法值忽略，由调用点默认兜底）。"""
    raw = (creds.get("max_tokens") or "").strip()
    if not raw:
        return None
    try:
        val = int(raw)
    except ValueError:
        return None
    return val if 1 <= val <= 1_000_000 else None


def _polish_answer(answer: str) -> str:
    """把管线内部面向运维的文案换成面向网页用户的说明。"""
    if not answer:
        return answer
    return answer.replace(_PIPELINE_FAIL_TEXT, _WEB_FAIL_TEXT)


def _validate_download_url(url: str) -> bool:
    """图片 URL 下载的 SSRF 校验（复用 api.py 的实现，保持同一口径：
    仅公网 http(s)，拒绝内网/回环/链路本地/云元数据地址）。"""
    from api import _validate_download_url as _impl
    return _impl(url)


__all__ = ["router", "prune_web_attachments", "WEB_ATTACHMENT_MAX_BYTES",
           "WEB_ATTACHMENT_MAX_FILES"]
