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
from core.attachments import build_attachments
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


_TIKZ_RE = re.compile(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
                      re.DOTALL)
_CHEMFIG_RE = re.compile(r"\\chemfig\{[^}]*\}")


def _strip_render_code(text: str) -> str:
    """剥离 assistant 历史消息中的渲染代码（TikZ/chemfig），只留纯文本。

    多轮对话时，assistant 历史消息是我们返回的**渲染后**文本（含 TikZ 代码）。
    这些代码不能回传给 LLM（会污染上下文、浪费 token），需剥离。
    """
    text = _TIKZ_RE.sub("", text)
    text = _CHEMFIG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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

    图片经视觉模型识别为 SMILES；音频暂不支持（显式提示）；
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
            from utils.ocr_utils import image_to_smiles
            for url in images:
                path = _fetch_image_to_temp(url, tmp_dir)
                if not path:
                    print(f"[api] 图片下载/解码失败: {url[:80]}")
                    parts.append("（一张图片下载失败，已忽略）")
                    continue
                smiles = image_to_smiles(path)
                if smiles:
                    parts.append(f"（上传的结构式图片识别为 SMILES：{smiles}）")
                else:
                    print(f"[api] 视觉模型未能识别图片: {url[:80]}")
                    parts.append("（一张结构式图片识别失败，已忽略）")
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


def _sse_stream(question: str, history: list, cid: str, created: int,
                public_base: str):
    """SSE 帧序列：role 帧 → 思考帧（B2：实时转发 LLM 生成草稿）→
    content 增量 → stop 帧（usage + x_soda.attachments）→ [DONE]。"""
    yield _sse_frame(cid, created, {"role": "assistant"})
    yield _sse_frame(cid, created, {"reasoning": "正在思考并绘制化学图示…"})

    result_q = queue.Queue(maxsize=1)
    progress_q = queue.Queue(maxsize=200)

    def work():
        try:
            answer = process_question(question, history=history,
                                      progress_callback=progress_q.put)
            attachments = build_attachments(answer or "", public_base) \
                if answer else []
            result_q.put((answer, attachments))
        except Exception as e:  # 渲染管线异常兜底为 stop 帧 + error 字段
            result_q.put(e)

    threading.Thread(target=work, daemon=True).start()
    draft = []          # LLM 生成草稿（限长保留，reasoning 帧覆盖式显示）
    last_flush = time.time()
    while True:
        try:
            result = result_q.get_nowait()
            break
        except queue.Empty:
            pass
        # 批量取出 LLM 增量（限频，避免每 chunk 一帧刷屏）
        pieces = []
        while True:
            try:
                pieces.append(progress_q.get_nowait())
            except queue.Empty:
                break
        if pieces:
            draft.append("".join(pieces))
            if len(draft) > 4:
                draft.pop(0)
            yield _sse_frame(cid, created,
                             {"reasoning": f"正在生成… {''.join(draft)}"})
            last_flush = time.time()
        elif time.time() - last_flush >= 3.0:
            yield _sse_frame(cid, created, {"reasoning": "仍在思考…"})
            last_flush = time.time()
        else:
            time.sleep(0.2)

    if isinstance(result, Exception):
        yield _sse_frame(cid, created, {}, finish="stop",
                         usage=_usage(question, ""),
                         error={"type": "upstream_error", "message": str(result)})
        yield "data: [DONE]\n\n"
        return

    answer, attachments = result
    answer = answer or "（未能生成回答）"
    step = 20
    for i in range(0, len(answer), step):
        yield _sse_frame(cid, created, {"content": answer[i:i + step]})
    yield _sse_frame(cid, created, {}, finish="stop",
                     usage=_usage(question, answer),
                     x_soda={"attachments": attachments} if attachments else None)
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

    answer = process_question(question, history=history) or "（未能生成回答）"
    attachments = build_attachments(answer, public_base)
    payload = {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": _usage(question, answer),
    }
    if attachments:
        payload["x_soda"] = {"attachments": attachments}
    return JSONResponse(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
