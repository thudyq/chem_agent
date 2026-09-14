# -*- coding: utf-8 -*-
"""core/attachments.py — 回答文本中的 TikZ 代码 → PNG 图片附件。

把 process_question 输出中的 tikzpicture / schemestart / chemfig 代码块
编译为 PNG，写入附件目录，生成清小搭 L2 扩展字段 x_soda.attachments
所需的条目（fileUrl/fileName/fileType/mimeType/fileSize）。

附件为随机 UUID 文件名，供 /files/{name} 无鉴权下载（不可猜测）；
清小搭对 /files 是热链、未必转存到自己的 OSS，图片在本地长期保留、仅在
磁盘超配额（max_bytes/max_files）时才删最旧的——不按时间 TTL 硬删，
否则历史对话的图会因文件被清而 404。
"""

import re
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


def replace_code_blocks_with_images(text: str, urls: list) -> str:
    """把文本中的 TikZ/chemfig 代码块替换为行内 markdown 图片引用。

    urls 与代码块**按原始顺序一一对应**（长度 == 代码块数，由
    build_attachments 保证）；元素为可用 url 的块 → ![化学图示-N](url)，
    元素为空/None 的块（编译失败）→ 该位置原地替换为友好提示
    "（图示未能渲染）"，**不会把后面的块往前挤**——否则多块时部分编译
    失败会让图片与编号错位、且真实成功图被误删（20260827 修复：曾因
    只按"成功块序号"填 url，导致中间失败块被赋予下一张图的 url、后面的
    成功图被挤掉）。无代码块时原样返回；urls 少于块数（异常兜底）时，
    多余块按失败处理。

    用于让清小搭在正文内联显示图片（20260826：平台把 attachments 渲染成
    文末缩略图，行内引用才可能内联）。
    """
    blocks = extract_code_blocks(text)
    if not blocks:
        return text or ""
    display = text or ""
    urls = urls or [None] * len(blocks)
    # 补齐到块数（异常兜底：urls 短于块数时，缺的按编译失败处理）
    if len(urls) < len(blocks):
        urls = urls + [None] * (len(blocks) - len(urls))
    for i, code in enumerate(blocks):
        url = urls[i]
        if url:
            display = display.replace(
                code, f"![化学图示-{i + 1}]({url})", 1)
        else:
            display = display.replace(code, "(图示未能渲染)", 1)
    return display


def _prune_attachments(dir_path: Path, max_bytes: int, max_files: int,
                       keep: frozenset = frozenset()) -> None:
    """配额滚动删：目录超过 max_bytes 或 max_files 时删最旧的 PNG，直到达标。

    不再按"超过 TTL 就删"——清小搭对 /files 是热链、未必转存到自己的 OSS，
    图片必须在本地长期保留（否则历史对话的图会因文件被清而 404）；
    只在磁盘/数量超配额时才回滚删最旧的，保证历史图长期可看、磁盘有上限。
    max_bytes/max_files <= 0 表示该项不限（无限增长，需自行定期清理）。
    keep：受保护文件名集（本批新写入的图）——配额统计含全部文件，但 keep
    中的文件不删（新图不被本次请求清掉）；非保护文件删完仍超配额则保持
    现状（宁超额，不删新图）。
    """
    if not dir_path.is_dir():
        return
    if max_bytes <= 0 and max_files <= 0:
        return
    pngs = [p for p in dir_path.glob("*.png")]
    if not pngs:
        return
    total = sum(p.stat().st_size for p in pngs if p.is_file())
    if _within_quota(total, len(pngs), max_bytes, max_files):
        return
    # 按 mtime 升序（最旧在前），逐个删直到两项都达标；keep 保护的不删
    for p in sorted(pngs, key=lambda x: x.stat().st_mtime):
        if p.name in keep:
            continue
        try:
            p.unlink()
            pngs.remove(p)
            total = sum(q.stat().st_size for q in pngs if q.is_file())
        except OSError:
            continue
        if _within_quota(total, len(pngs), max_bytes, max_files):
            break


def _within_quota(total: int, count: int, max_bytes: int, max_files: int) -> bool:
    """total/count 是否都在配额内（max_bytes/max_files <=0 表示该项不限）。"""
    if max_bytes > 0 and total > max_bytes:
        return False
    if max_files > 0 and count > max_files:
        return False
    return True


def build_attachments(answer: str, public_base: str,
                      dir_path: Path | None = None,
                      max_bytes: int | None = None,
                      max_files: int | None = None) -> list:
    """编译回答中的 TikZ 代码为 PNG，返回行内图片引用所需的 url 列表。

    参数:
        answer: process_question 的最终文本（含内联 TikZ 代码）。
        public_base: 服务公网地址（不含尾部 /），用于拼接 fileUrl。
        dir_path: 附件存放目录（默认 settings.service.attachment_dir）。
        max_bytes: 目录总字节上限，超出删最旧（默认 settings.service.
            attachment_max_bytes）；0/负数=不限。
        max_files: 目录最多文件数，超出删最旧（默认 attachment_max_files）；
            0/负数=不限。

    返回:
        与正文代码块**一一对应**的列表（长度 == 块数，含按原始顺序的
        {fileUrl,...} 或 None（该块编译失败，无图））。供调用方用
        `[a["fileUrl"] if a else None for a in attachments]` 传给
        replace_code_blocks_with_images，保证图片与编号不错位、
        失败块不影响后续块。无代码块或全部编译失败时长度为块数（元素为
        None）或空（无块）。
    """
    blocks = extract_code_blocks(answer)
    if not blocks:
        return []
    if dir_path is None:
        dir_path = settings.service.attachment_dir
    if max_bytes is None:
        max_bytes = settings.service.attachment_max_bytes
    if max_files is None:
        max_files = settings.service.attachment_max_files
    dir_path.mkdir(parents=True, exist_ok=True)

    base = (public_base or "").rstrip("/")
    # 与 blocks 对齐；成功块 {fileUrl,...}，失败块 None（占位，保持位置对应）
    attachments = [None] * len(blocks)
    # 并行编译各块（LaTeX 子进程各自独立临时目录）：3 图并发，把串行 ~10s 压到 ~3~4s
    from concurrent.futures import ThreadPoolExecutor
    pngs = [None] * len(blocks)
    if len(blocks) > 1:
        with ThreadPoolExecutor(max_workers=min(len(blocks), 3)) as ex:
            futs = {i: ex.submit(compile_tikz_to_png, code)
                    for i, code in enumerate(blocks)}
            for i, fut in futs.items():
                try:
                    pngs[i] = fut.result()
                except Exception as e:  # 单块编译异常不拖垮整轮（其余块照常）
                    print(f"[attachments] 图 {i + 1} 编译异常，作失败块: {e}")
                    pngs[i] = None
    else:
        try:
            pngs[0] = compile_tikz_to_png(blocks[0])
        except Exception as e:
            print(f"[attachments] 图 1 编译异常，作失败块: {e}")
            pngs[0] = None
    written = []
    for i, (code, png) in enumerate(zip(blocks, pngs)):
        if not png:
            continue
        name = f"{uuid.uuid4().hex}.png"
        (dir_path / name).write_bytes(png)
        written.append(name)
        attachments[i] = {
            "fileUrl": f"{base}/files/{name}",
            "fileName": f"化学图示-{i + 1}.png",
            "fileType": _FILE_TYPE_IMAGE,
            "mimeType": _MIME_PNG,
            "fileSize": len(png),
        }
    # 配额滚动在写入后执行（统计含本批新图，最终状态不超配额）；本批新图
    # 列入 keep 保护——新图不被本次请求清掉（若在写入前清理，配额会被
    # 本批突破：max_files=2 时 3 旧图只清到 2，写入后变 3）
    if written:
        _prune_attachments(dir_path, max_bytes, max_files,
                           keep=frozenset(written))
    return attachments
