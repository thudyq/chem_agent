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
import json
import queue
import re
import tempfile
import threading
import time
from pathlib import Path

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app import process_question
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


def _extract_question(messages: list) -> tuple[str, list]:
    """从 messages 取最后一个 user 消息，返回 (文本, 图片 URL 列表)。"""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content, []
        if isinstance(content, list):
            texts, images = [], []
            for part in content:
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url:
                        images.append(url)
            return "\n".join(t for t in texts if t), images
    return "", []


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


def _build_question(text: str, images: list, tmp_dir: str) -> str:
    """文本 + 图片识别结果拼成最终问题。"""
    parts = [text] if text else []
    if images:
        from utils.ocr_utils import image_to_smiles
        for url in images:
            path = _fetch_image_to_temp(url, tmp_dir)
            if not path:
                parts.append("（一张图片下载失败，已忽略）")
                continue
            smiles = image_to_smiles(path)
            if smiles:
                parts.append(f"（上传的结构式图片识别为 SMILES：{smiles}）")
            else:
                parts.append("（一张结构式图片识别失败，已忽略）")
    return "\n".join(parts) or "（空消息）"


def _sse_frame(cid: str, created: int, delta: dict,
               finish: str | None = None, usage: dict | None = None,
               error: dict | None = None) -> str:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    chunk = {"id": cid, "object": "chat.completion.chunk",
             "created": created, "choices": [choice]}
    if usage is not None:
        chunk["usage"] = usage
    if error is not None:
        chunk["error"] = error
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _sse_stream(question: str, cid: str, created: int):
    """SSE 帧序列：role 帧 → 思考帧（每 3s 心跳保活）→ content 增量 → stop 帧 → [DONE]。"""
    yield _sse_frame(cid, created, {"role": "assistant"})
    yield _sse_frame(cid, created, {"reasoning": "正在思考并绘制化学图示…"})

    result_q = queue.Queue(maxsize=1)

    def work():
        try:
            result_q.put(process_question(question))
        except Exception as e:  # 渲染管线异常兜底为 stop 帧 + error 字段
            result_q.put(e)

    threading.Thread(target=work, daemon=True).start()
    while True:
        try:
            result = result_q.get(timeout=3.0)
            break
        except queue.Empty:
            yield _sse_frame(cid, created, {"reasoning": "仍在思考…"})

    if isinstance(result, Exception):
        yield _sse_frame(cid, created, {}, finish="stop",
                         usage=_usage(question, ""),
                         error={"type": "upstream_error", "message": str(result)})
        yield "data: [DONE]\n\n"
        return

    answer = result or "（未能生成回答）"
    step = 20
    for i in range(0, len(answer), step):
        yield _sse_frame(cid, created, {"content": answer[i:i + step]})
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(None)):
    _check_auth(authorization)
    body = await request.json()
    # 严格按 JSON 布尔解析 stream（字符串 "false" 视为非流式）
    stream = body.get("stream", False)
    stream = stream if isinstance(stream, bool) else False

    text, images = _extract_question(body.get("messages") or [])
    cid = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    with tempfile.TemporaryDirectory() as tmp_dir:
        question = _build_question(text, images, tmp_dir)

    if stream:
        return StreamingResponse(
            _sse_stream(question, cid, created),
            media_type="text/event-stream",
        )

    answer = process_question(question) or "（未能生成回答）"
    return JSONResponse({
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": _usage(question, answer),
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
