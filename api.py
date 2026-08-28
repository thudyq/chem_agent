# -*- coding: utf-8 -*-
"""api.py — 清小搭接入服务（OpenAI 兼容协议）。

端点：
    GET  /v1/models            连通性与凭证校验
    POST /v1/chat/completions  对话（非流式 JSON + 流式 SSE）

鉴权：Authorization: Bearer <SERVICE_API_KEY>（在 .env 中配置，
接入清小搭向导时「API 密钥」填同一个值）。

图片输入：messages 的 content 数组中可含 image_url（data: base64 或
http(s) URL），经视觉模型识别为 SMILES 后并入问题文本。

运行：
    uvicorn api:app --host 0.0.0.0 --port 8000
    baseUrl 填 http://<host>:8000/v1
"""

import base64
import ipaddress
import json
import queue
import re
import socket
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app import process_question
from core.attachments import build_attachments, replace_code_blocks_with_images
from core.config import settings

app = FastAPI(title="Chem_Agent", version="1.0.0")

# 清小搭网关为服务端调用，CORS 仅供浏览器直连调试；密钥保护下放开无妨
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVICE_KEY = settings.service.api_key

# SSRF 防护：禁止下载的内网/保留地址网段（RFC1918 + 回环 + 链路本地 +
# 云元数据地址 + 组播/保留）。元数据地址 169.254.169.254（AWS/GCP/Azure）
# 与 100.100.100.200（阿里云）分别落在 169.254.0.0/16 与 100.64.0.0/10 内。
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("255.255.255.255/32"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]
_DANGEROUS_HOST_SUFFIX = (".local", ".internal", ".localhost", ".lan",
                          ".corp", ".home", ".intranet")


def _is_blocked_ip(ip_str: str) -> bool:
    """IP 是否命中内网/保留网段；无法解析为合法 IP 一律拒绝。"""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return any(ip in net for net in _BLOCKED_NETWORKS)


def _validate_download_url(url: str) -> bool:
    """SSRF 防护：仅允许公网 http/https 下载。

    拒绝：非 http/https、无 host、内网/保留 IP（含云元数据地址）、
    localhost/.local 等危险主机名、DNS 解析到内网 IP、解析失败（保守拒绝）。
    """
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    host = host.rstrip(".").lower()
    if host.startswith("localhost") or host.endswith(_DANGEROUS_HOST_SUFFIX):
        return False
    try:
        ip = ipaddress.ip_address(host)
        return not _is_blocked_ip(str(ip))
    except ValueError:
        pass  # 域名，需 DNS 解析后校验
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    addrs = {info[4][0] for info in infos}
    if not addrs:
        return False
    return all(not _is_blocked_ip(a) for a in addrs)


def _check_auth(authorization: str | None) -> None:
    """Bearer 鉴权；未配置服务密钥返回 500，凭证缺失/错误返回 401。"""
    if not SERVICE_KEY:
        raise HTTPException(status_code=500, detail="服务未配置 SERVICE_API_KEY")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing credential")
    if authorization[len("Bearer "):] != SERVICE_KEY:
        raise HTTPException(status_code=401, detail="invalid credential")


def _usage(prompt_text: str, answer: str) -> dict:
    """token 用量（无真实统计时按字符数估计，协议允许填 0）。"""
    p, c = len(prompt_text), len(answer)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _extract_question(messages: list) -> tuple[str, list, list, list]:
    """从 messages 取最后一个 user 消息，返回 (文本, 图片, 音频, 文件)。

    兼容多种多模态格式：content 为字符串 / 数组；part 类型
    text / input_text / image_url / input_image / image / input_audio / file；
    image_url 取值可以是 {"url": ...} 字典或直接是字符串。
    音频返回 [(url, format)]；文件返回 [(url, file_id, filename)]。
    """
    if not isinstance(messages, list):
        return "", [], [], []
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content, [], [], []
        if isinstance(content, list):
            texts, images, audios, files = [], [], [], []
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
                elif ptype == "input_audio":
                    ref = part.get("input_audio") or {}
                    if isinstance(ref, dict) and ref.get("url"):
                        audios.append((ref["url"], ref.get("format", "")))
                elif ptype == "file":
                    ref = part.get("file") or {}
                    if isinstance(ref, dict):
                        files.append((ref.get("url", ""),
                                      ref.get("file_id", ""),
                                      ref.get("filename", "")))
            return "\n".join(t for t in texts if t), images, audios, files
    return "", [], [], []


def _log_diagnostics(diag: list) -> None:
    """后端日志：打印回答中未渲染标记的诊断（含修正救回/最终失败）。

    前端只看到注入的友好降级文本（"图示无法渲染，已省略"），技术细节
    （reason）仅在此处输出，供质量分析与修正闭环改进。diag 为空（无失败）
    时静默。
    """
    if not diag:
        return
    unresolved = sum(1 for d in diag if d.get("resolved") is False)
    print(f"[api] 回答含 {len(diag)} 个未渲染标记诊断"
          f"（修正后仍失败 {unresolved} 个）：")
    for d in diag:
        status = "已修正救回" if d.get("resolved") else "未解决"
        stage = d.get("stage", "main")
        print(f"  - [round {d.get('round')}][{stage}][{status}] "
              f"{d.get('raw', '')[:60]} → {d.get('reason', '')[:120]}")


_TIKZ_RE = re.compile(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
                      re.DOTALL)
_CHEMFIG_RE = re.compile(r"\\chemfig\{[^}]*\}")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)\s]*\)")


def _strip_render_code(text: str) -> str:
    """剥离 assistant 历史消息中的渲染产物（TikZ/chemfig/图片 markdown）。

    多轮对话时，assistant 历史消息是我们返回的**渲染后**文本（含 TikZ 代码）。
    这些代码不能回传给 LLM（会污染上下文、浪费 token），需剥离。
    20260828：行内图片 markdown（![化学图示-N](fileUrl)）一并剥离——保留
    会让 LLM 在下轮模仿手写图片链接（编造不存在的 fileUrl，前端 404 空白，
    que_test_retry 实测：同一对话第二轮图片全丢、URL 为递增规律假 uuid）。
    """
    text = _TIKZ_RE.sub("", text)
    text = _CHEMFIG_RE.sub("", text)
    text = _MD_IMAGE_RE.sub("[化学图示]", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_md_images(text: str) -> str:
    """剥离 LLM 输出中手写的 markdown 图片链接（![...](...)）。

    真实图片只经附件通道注入（build_attachments 编译 + 随后
    replace_code_blocks_with_images 替换）；原始输出中出现的图片链接只可能
    是 LLM 模仿历史消息编造的假链接（文件不存在，前端 404 空白）——删除。
    """
    text = _MD_IMAGE_RE.sub("", text or "")
    return re.sub(r"\n{3,}", "\n\n", text)


def _extract_history(messages: list, max_items: int = 10) -> list:
    """提取最后一条 user 消息之前的对话历史（A3 多轮对话）。

    返回 [{"role": "user"/"assistant", "content": 文本}, ...]（最近 max_items 条）。
    - 当前问题 = 最后一条 user 消息（由 _extract_question 处理），其本身不在此处；
    - assistant 历史剥离渲染代码（TikZ/chemfig）；
    - 多模态 content 数组只取文本部分。
    """
    if not isinstance(messages, list):
        return []
    last_user_idx = -1
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            last_user_idx = i
    history = []
    for m in messages[:last_user_idx][-max_items:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict) and part.get("type") in (
                        "text", "input_text"):
                    texts.append(part.get("text", ""))
            text = "\n".join(t for t in texts if t)
        else:
            continue
        if role == "assistant":
            text = _strip_render_code(text)
        if text.strip():
            history.append({"role": role, "content": text.strip()})
    return history


def _fetch_image_to_temp(url: str, tmp_dir: str) -> str | None:
    """把 data: base64 或 http(s) 图片 URL 存为临时文件，返回路径；失败 None。"""
    if url.startswith("data:"):
        m = re.match(r"data:image/(\w+);base64,(.*)", url, re.DOTALL)
        if not m:
            return None
        ext = "jpg" if m.group(1).lower() in ("jpg", "jpeg") else "png"
        try:
            data = base64.b64decode(m.group(2))
        except Exception:
            return None
        path = Path(tmp_dir) / f"upload.{ext}"
        path.write_bytes(data)
        return str(path)
    if url.startswith(("http://", "https://")):
        if not _validate_download_url(url):
            print(f"[api] 拒绝下载图片（SSRF 防护）: {url[:80]}")
            return None
        try:
            resp = requests.get(url, timeout=20)
        except requests.exceptions.RequestException:
            return None
        if resp.status_code != 200:
            return None
        path = Path(tmp_dir) / "upload.png"
        path.write_bytes(resp.content)
        return str(path)
    return None


def _download_text(url: str, limit: int = 4000) -> str | None:
    """按 URL 下载文本文件内容（截断 limit 字符）；失败返回 None。"""
    if not _validate_download_url(url):
        print(f"[api] 拒绝下载文件（SSRF 防护）: {url[:80]}")
        return None
    try:
        resp = requests.get(url, timeout=20)
    except requests.exceptions.RequestException:
        return None
    if resp.status_code != 200:
        return None
    return resp.content.decode("utf-8", errors="replace")[:limit]


def _build_question(text: str, images: list, audios: list, files: list,
                    tmp_dir: str) -> str:
    """文本 + 各模态附件处理结果拼成最终问题。

    图片经视觉模型理解为结构化描述（文字转录/结构 SMILES/机理描述），
    与用户文字合并为同一问题；音频暂不支持（显式提示）；
    文本类文件（txt/md/csv）下载后内联，其余文件类型显式提示不支持。
    所有失败都给出明确原因并打日志，不静默丢弃。
    """
    parts = [text] if text else []
    if images:
        if not settings.vision.is_configured:
            print("[api] 收到图片但未配置 VISION_MODEL/VISION_BASE_URL/VISION_API_KEY")
            parts.append(
                "（用户上传了图片，但服务未配置视觉模型（VISION_MODEL 等），"
                "无法识别图片内容，请提示用户先描述结构或联系管理员配置视觉模型）"
            )
        else:
            from utils.ocr_utils import describe_image
            for i, url in enumerate(images, 1):
                path = _fetch_image_to_temp(url, tmp_dir)
                if not path:
                    print(f"[api] 图片下载/解码失败: {url[:80]}")
                    parts.append("（一张图片下载失败，已忽略）")
                    continue
                desc = describe_image(path)
                if desc and desc.get("content"):
                    # 图片描述与用户文字合并为同一问题（多模态两段式，B 方案）
                    parts.append(
                        f"（用户上传的图片 {i} 的内容（{desc['type']}）："
                        f"{desc['content']}）"
                    )
                    # B3（20260826）：图像为视觉模型自动识别，提示用户识别可能有误
                    if desc.get("smiles_ok") is False:
                        parts.append(
                            f"（提示：图片 {i} 识别出的结构式 SMILES 经校验"
                            "无法解析，识别可能有误；请在回答中提醒用户核对"
                            "图片/结构，必要时请用户用文字描述该结构）")
                    else:
                        parts.append(
                            f"（提示：图片 {i} 为视觉模型自动识别，识别可能"
                            "有误；请在回答中提醒用户以图片为准、核对识别内容，"
                            "如有出入可请用户用文字补充说明）")
                else:
                    print(f"[api] 视觉模型未能理解图片: {url[:80]}")
                    # 识别失败：空内容 + 用户文字照常传给主 LLM，并明确
                    # 告知"图片识别失败"（视觉重试已耗尽，仍继续问答流程）
                    parts.append(
                        f"（用户上传的图片 {i} 识别失败：视觉模型多次尝试仍"
                        "无法理解图片内容，图片内容不可用。请基于文字内容"
                        "作答，并提示用户重新上传图片或改用文字描述）"
                    )
    for url, fmt in audios:
        print(f"[api] 收到音频输入（{fmt or '未知格式'}），暂不支持: {url[:80]}")
        parts.append("（用户上传了一段音频，本服务暂不支持音频输入，请改用文字描述）")
    for url, file_id, filename in files:
        name = filename or "未知文件"
        if not url:
            print(f"[api] 文件仅有 file_id 无 URL，无法拉取: {name}")
            parts.append(f"（用户上传了文件「{name}」，但服务无法按 file_id 拉取，已忽略）")
            continue
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext in ("txt", "md", "markdown", "csv"):
            content = _download_text(url)
            if content:
                parts.append(f"（用户上传的文件「{name}」，内容如下：\n{content}）")
            else:
                print(f"[api] 文件下载失败: {name} {url[:80]}")
                parts.append(f"（文件「{name}」下载失败，已忽略）")
        else:
            print(f"[api] 暂不支持的文件类型: {name}")
            parts.append(f"（用户上传了文件「{name}」，本服务暂不支持解析该类型文件，已忽略）")
    return "\n".join(parts) or "（空消息）"


def _public_base(request: Request) -> str:
    """附件下载 URL 的公网前缀：优先 PUBLIC_BASE_URL 配置，否则按请求 Host 推导
    （Nginx 反代需 proxy_set_header Host $host;）。"""
    if settings.service.public_base_url:
        return settings.service.public_base_url
    return str(request.base_url).rstrip("/")


def _sse_frame(cid: str, created: int, delta: dict,
               finish: str | None = None, usage: dict | None = None,
               error: dict | None = None, x_soda: dict | None = None) -> str:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    chunk = {"id": cid, "object": "chat.completion.chunk",
             "created": created, "choices": [choice]}
    if usage is not None:
        chunk["usage"] = usage
    if error is not None:
        chunk["error"] = error
    if x_soda is not None:
        chunk["x_soda"] = x_soda
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# P2 修正触发标记：correction_callback 经 progress_q 传给主循环的哨兵（非文本片段）
_CORRECTION_MARK = object()

# 思考期心跳间隔（秒）：SSE 长时间无数据帧时 Nginx 等网关可能断开
# （proxy_read_timeout 默认 60s），需要保活帧；但清小搭等前端对 reasoning
# 帧是逐条追加显示，心跳太频繁会刷屏——10s 一次是保活与观感的折中。
_HEARTBEAT_INTERVAL = 10.0


def _sse_stream(question: str, history: list, cid: str, created: int,
                public_base: str):
    """SSE 帧序列：role 帧 → 思考帧（固定提示 + 低频心跳 + P2 修正提示；不再
    转发草稿增量——清小搭等前端对 reasoning 帧逐条追加显示，草稿帧会造成
    "正在生成… 草稿"无限叠加错乱）→ content 增量（文本里的 TikZ 已替换为
    行内图片引用 ![化学图示-N](fileUrl)）→ stop 帧（usage）→ [DONE]。

    文本先行：process_question 一返回立即发 content 帧，附件 PNG 编译在
    work 线程与 content 发送并行、完成后挂 stop 帧——用户先读到完整文字
    回答，图示随后到达（与本地界面"文本先行、图片回填"同构）。
    """
    yield _sse_frame(cid, created, {"role": "assistant"})
    yield _sse_frame(cid, created, {"reasoning": "正在思考并绘制化学图示…"})

    answer_q = queue.Queue(maxsize=1)    # process_question 完成（文本就绪）
    result_q = queue.Queue(maxsize=1)    # attachments 编译完成
    progress_q = queue.Queue(maxsize=200)

    def _safe_put(item) -> None:
        # 修正标记是装饰性提示，队列满时丢弃即可，不阻塞管线
        try:
            progress_q.put_nowait(item)
        except queue.Full:
            pass

    def work():
        diag = []
        try:
            answer = process_question(
                question, history=history,
                progress_callback=progress_q.put,
                correction_callback=lambda: _safe_put(_CORRECTION_MARK),
                diagnostics=diag)
        except Exception as e:  # 管线异常兜底为 stop 帧 + error 字段
            answer_q.put(e)
            return
        answer = _strip_md_images(answer)
        _log_diagnostics(diag)
        try:
            attachments = build_attachments(answer or "", public_base) \
                if answer else []
            display = replace_code_blocks_with_images(
                answer or "",
                [a["fileUrl"] if a else None for a in attachments])
        except Exception as e:  # 编译异常不拖垮已生成的文本回答
            print(f"[api] 附件编译异常，降级为无附件: {e}")
            attachments, display = [], answer or ""
        answer_q.put(display)
        result_q.put(attachments)

    threading.Thread(target=work, daemon=True).start()
    last_flush = time.time()
    while True:
        try:
            answer = answer_q.get_nowait()
            break
        except queue.Empty:
            pass
        # 消费 progress_q（防止队列满阻塞 process_question 的 on_piece 回调）；
        # 不把草稿增量作为 reasoning 帧转发（清小搭对 reasoning 帧逐条追加，
        # 草稿帧会无限叠加错乱），只发低频状态帧避免长时间只有 "..."。
        correction = False
        while True:
            try:
                p = progress_q.get_nowait()
            except queue.Empty:
                break
            if p is _CORRECTION_MARK:
                correction = True
        if correction:
            yield _sse_frame(cid, created, {"reasoning": "正在修正回答…"})
            last_flush = time.time()
        elif time.time() - last_flush >= _HEARTBEAT_INTERVAL:
            yield _sse_frame(cid, created, {"reasoning": "正在思考并绘制化学图示…"})
            last_flush = time.time()
        else:
            time.sleep(0.2)

    if isinstance(answer, Exception):
        yield _sse_frame(cid, created, {}, finish="stop",
                         usage=_usage(question, ""),
                         error={"type": "upstream_error", "message": str(answer)})
        yield "data: [DONE]\n\n"
        return

    # 收尾排空：answer 就绪时队列里可能仍有未消费的修正标记（快速回答时
    # 主循环来不及取出）——只保留修正提示；草稿片段一律丢弃（不再转发）。
    correction = False
    while True:
        try:
            p = progress_q.get_nowait()
        except queue.Empty:
            break
        if p is _CORRECTION_MARK:
            correction = True
    if correction:
        yield _sse_frame(cid, created, {"reasoning": "正在修正回答…"})

    answer = answer or "（未能生成回答）"   # answer_q 已是 display（含行内图片）
    step = 20
    for i in range(0, len(answer), step):
        yield _sse_frame(cid, created, {"content": answer[i:i + step]})

    # 图已通过行内 markdown 引用（content 里的 ![化学图示-N](fileUrl)）展示，
    # 且内容帧在编译完成后才发出（文件已落盘），无需等待/挂 x_soda。
    yield _sse_frame(cid, created, {}, finish="stop",
                     usage=_usage(question, answer))
    yield "data: [DONE]\n\n"


@app.get("/")
def root():
    return {"service": "chem_agent", "endpoints": ["/v1/models", "/v1/chat/completions"]}


@app.get("/v1/models")
def list_models(authorization: str | None = Header(None)):
    _check_auth(authorization)
    return {"object": "list", "data": [
        {"id": "chem-agent", "object": "model", "owned_by": "chem_agent"},
    ]}


@app.get("/files/{filename}")
def serve_attachment(filename: str):
    """图片附件下载（清小搭拉取用）。

    无鉴权：文件名为随机 UUID 不可猜测；仅允许十六进制 .png 名，防路径穿越。
    """
    if not re.fullmatch(r"[0-9a-f]{32}\.png", filename):
        raise HTTPException(status_code=404, detail="not found")
    path = settings.service.attachment_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="image/png", filename=filename)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(None)):
    _check_auth(authorization)
    body = await request.json()
    # 严格按 JSON 布尔解析 stream（字符串 "false" 视为非流式）
    stream = body.get("stream", False)
    stream = stream if isinstance(stream, bool) else False

    text, images, audios, files = _extract_question(body.get("messages") or [])
    history = _extract_history(body.get("messages") or [])
    cid = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())
    public_base = _public_base(request)

    with tempfile.TemporaryDirectory() as tmp_dir:
        question = _build_question(text, images, audios, files, tmp_dir)

    if stream:
        return StreamingResponse(
            _sse_stream(question, history, cid, created, public_base),
            media_type="text/event-stream",
        )

    diag = []
    answer = process_question(question, history=history,
                              diagnostics=diag) or "（未能生成回答）"
    answer = _strip_md_images(answer)
    _log_diagnostics(diag)
    try:
        attachments = build_attachments(answer, public_base)
    except Exception as e:  # 编译异常不拖垮已生成的文本回答（与流式路径一致）
        print(f"[api] 附件编译异常，降级为无附件: {e}")
        attachments = []
    content = replace_code_blocks_with_images(
        answer, [a["fileUrl"] if a else None for a in attachments])
    payload = {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": _usage(question, answer),
    }
    # 图已通过行内 markdown 引用（content 里的 ![化学图示-N](fileUrl)）展示，
    # 不再挂 x_soda.attachments，避免文末再出现一排缩略图（20260826）。
    return JSONResponse(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
