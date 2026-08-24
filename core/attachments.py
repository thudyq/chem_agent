# -*- coding: utf-8 -*-
"""core/attachments.py — 回答文本中的 TikZ 代码 → PNG 图片附件。

把 process_question 输出中的 tikzpicture / schemestart / chemfig 代码块
编译为 PNG，写入附件目录，生成清小搭 L2 扩展字段 x_soda.attachments
所需的条目（fileUrl/fileName/fileType/mimeType/fileSize）。

附件为随机 UUID 文件名，供 /files/{name} 无鉴权下载（不可猜测）；
清小搭收到响应后会立即转存到自己的 OSS，本地附件按 TTL 定期清理。
"""

import re
import time
import uuid
from pathlib import Path

from core.config import settings
from utils.latex_compile import compile_tikz_to_png

# 代码块：tikzpicture 整块 | schemestart 反应式 | 单个 \chemfig{...}
_CODE_RE = re.compile(
    r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}"
    r"|\\schemestart.*?\\schemestop"
    r"|\\chemfig\{(?:[^{}]|\{[^{}]*\})*\}",
    re.DOTALL,
)

_MIME_PNG = "image/png"
_FILE_TYPE_IMAGE = "image"


def extract_code_blocks(text: str) -> list:
    """提取文本中所有可编译的 TikZ/chemfig 代码块。"""
    return _CODE_RE.findall(text or "")


def strip_code_blocks(text: str) -> str:
    """移除文本中的 TikZ/chemfig 代码块（只留说明文字）。

    这些代码块会被编译为 PNG 附件（x_soda.attachments），不应以裸 LaTeX
    出现在最终回答里——清小搭端用附件承载图片，文本里保留裸 TikZ 会被当作
    普通文本显示（20260826 实测）。失败未编译的块一并移除，宁可不显示也不
    露源码。
    """
    return _CODE_RE.sub("", text or "")


def _cleanup_old_files(dir_path: Path, ttl: int) -> None:
    if ttl <= 0 or not dir_path.is_dir():
        return
    cutoff = time.time() - ttl
    for f in dir_path.glob("*.png"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def build_attachments(answer: str, public_base: str,
                      dir_path: Path | None = None,
                      ttl: int | None = None) -> list:
    """编译回答中的 TikZ 代码为 PNG，返回 x_soda.attachments 条目列表。

    参数:
        answer: process_question 的最终文本（含内联 TikZ 代码）。
        public_base: 服务公网地址（不含尾部 /），用于拼接 fileUrl。
        dir_path: 附件存放目录（默认 settings.service.attachment_dir）。
        ttl: 附件保留秒数（默认 settings.service.attachment_ttl）。

    返回:
        attachments 列表；无代码块或全部编译失败返回空列表。
    """
    blocks = extract_code_blocks(answer)
    if not blocks:
        return []
    if dir_path is None:
        dir_path = settings.service.attachment_dir
    if ttl is None:
        ttl = settings.service.attachment_ttl
    dir_path.mkdir(parents=True, exist_ok=True)
    _cleanup_old_files(dir_path, ttl)

    base = (public_base or "").rstrip("/")
    attachments = []
    for i, code in enumerate(blocks, 1):
        png = compile_tikz_to_png(code)
        if not png:
            continue
        name = f"{uuid.uuid4().hex}.png"
        (dir_path / name).write_bytes(png)
        attachments.append({
            "fileUrl": f"{base}/files/{name}",
            "fileName": f"化学图示-{i}.png",
            "fileType": _FILE_TYPE_IMAGE,
            "mimeType": _MIME_PNG,
            "fileSize": len(png),
        })
    return attachments
