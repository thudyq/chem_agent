# -*- coding: utf-8 -*-
"""utils/latex_compile.py — 把 TikZ/chemfig 代码片段编译成 PNG 字节流。

设计目标
--------
*renderers/* 各渲染器只产出裸 TikZ / chemfig 片段（无导言区）。本模块负责：
1. 包装成可编译的 standalone 文档（含 CJK 支持）。
2. 自动探测 LaTeX 引擎与 PDF→PNG 转换器，兼容：
   - 原生 Linux/macOS 的 ``xelatex`` / ``pdflatex``；
   - WSL 下通过 interop 调用 Windows MiKTeX 的 ``xelatex.exe`` / ``pdflatex.exe``；
   - PDF→PNG：优先 PyMuPDF(``fitz``)，其次 ``pdftoppm``（原生或 .exe）。
3. 两级重试：先 CJK 导言区，失败则回退纯英文导言区（保住结构图）。
4. 任一环节失败返回 ``None``，由调用方优雅降级到 ``st.code``。

公共 API：``compile_tikz_to_png(code, title="", dpi=200) -> bytes | None``
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from typing import Optional

__all__ = ["compile_tikz_to_png", "detect_backends"]


# ---------------------------------------------------------------------------
# 后端探测
# ---------------------------------------------------------------------------

# 常见 Windows MiKTeX 安装路径（仅用于 .exe interop 兜底）
_MIKTEX_CANDIDATES = [
    r"C:\Program Files\MiKTeX\miktex\bin\x64",
    r"C:\Users\{}\AppData\Local\Programs\MiKTeX\miktex\bin\x64".format(
        os.environ.get("USER", "public")
    ),
]


def _is_windows_exe(path: str) -> bool:
    return bool(path) and path.lower().endswith(".exe")


def _try_exec(path: str, timeout: int = 15) -> bool:
    """执行验证：返回 True 当且仅当二进制存在且能启动（rc 可能非 0）。

    用执行而非 os.path.isfile 判断存在性——WSL 的 /mnt/c 9p 挂载下 stat
    调用偶发失败，而 posix_spawn 在挂载可用时能正常启动 .exe。
    """
    try:
        subprocess.run([path, "--version"], capture_output=True, timeout=timeout)
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


def _engine_dirs() -> list[str]:
    """收集可能装了 LaTeX 的目录：PATH 中含 miktex/texlive/tex 字样的目录 + 兜底路径。

    从 PATH 取目录可绕过 WSL 用户名 ≠ Windows 用户名 的问题（PATH 条目本身
    就含正确路径），比硬编码 os.environ['USER'] 可靠。
    """
    dirs, seen = [], set()
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d or d in seen:
            continue
        dl = d.lower()
        if "miktex" in dl or "texlive" in dl or "texmf" in dl:
            seen.add(d)
            dirs.append(d)
    for base in _MIKTEX_CANDIDATES:
        wsl_base = base.replace(r"C:", "/mnt/c").replace("\\", "/")
        if wsl_base not in seen:
            seen.add(wsl_base)
            dirs.append(wsl_base)
    return dirs


@lru_cache(maxsize=1)
def _find_latex_engine() -> Optional[str]:
    """返回首个可启动的 LaTeX 引擎路径。优先 xelatex（CJK 友好）。"""
    names = ["xelatex", "xelatex.exe", "pdflatex", "pdflatex.exe"]
    for n in names:
        p = shutil.which(n)
        if p and _try_exec(p):
            return p
    for d in _engine_dirs():
        for n in names:
            cand = os.path.join(d, n)
            if _try_exec(cand):
                return cand
    for n in names:
        if _try_exec(n):
            return n
    return None


def _pdf_to_png_pymupdf(pdf_path: str, dpi: int) -> Optional[bytes]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")
    except Exception:
        return None


def _find_pdftoppm() -> Optional[str]:
    """探测 pdftoppm：PATH 原生/.exe → latex 引擎同目录（执行验证）。

    引擎同目录最可靠：MiKTeX 把 pdftoppm.exe 和 xelatex.exe 放一起。
    """
    for n in ("pdftoppm", "pdftoppm.exe"):
        p = shutil.which(n)
        if p and _try_exec(p):
            return p
    eng = _find_latex_engine()
    if eng:
        for cand in ("pdftoppm.exe", "pdftoppm"):
            sibling = os.path.join(os.path.dirname(eng), cand)
            if _try_exec(sibling):
                return sibling
    return None


def _pdf_to_png_pdftoppm(pdf_path: str, dpi: int, engine_is_exe: bool) -> Optional[bytes]:
    """用 pdftoppm（原生或 .exe）转 PNG，返回首个 PNG 字节。"""
    conv = _find_pdftoppm()
    if not conv:
        return None

    work = os.path.dirname(os.path.abspath(pdf_path))
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_prefix = os.path.join(work, f"{base}_png")
    need_win = _is_windows_exe(conv) and sys.platform != "win32"

    def _w(p: str) -> str:
        return _to_win_path(p) if need_win else p

    try:
        subprocess.run(
            [conv, "-png", "-r", str(dpi), _w(pdf_path), _w(out_prefix)],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except Exception:
        return None
    # pdftoppm 输出 <prefix>-1.png（多页时 -2, -3...）
    for i in range(1, 10):
        cand = f"{out_prefix}-{i}.png"
        if os.path.isfile(cand):
            with open(cand, "rb") as f:
                return f.read()
    return None


def _to_win_path(linux_path: str) -> str:
    """WSL 路径 → Windows 路径（供 .exe interop 使用）。"""
    try:
        out = subprocess.run(
            ["wslpath", "-w", linux_path],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        return out
    except Exception:
        return linux_path


@lru_cache(maxsize=1)
def _have_pymupdf() -> bool:
    try:
        import fitz  # noqa: F401
        return True
    except ImportError:
        return False


def detect_backends() -> dict:
    """诊断用：返回当前探测到的后端。"""
    eng = _find_latex_engine()
    return {
        "latex_engine": eng,
        "engine_is_exe": _is_windows_exe(eng) if eng else None,
        "pymupdf_available": _have_pymupdf(),
        "pdftoppm": _find_pdftoppm(),
        "platform": platform.system(),
        "wslpath": shutil.which("wslpath"),
    }


# ---------------------------------------------------------------------------
# 编译核心
# ---------------------------------------------------------------------------

def _preamble(cjk: bool) -> str:
    """生成导言区。cjk=True 时加 ctex（Windows 用 fontset=windows）。"""
    engine = _find_latex_engine() or ""
    on_windows_tex = _is_windows_exe(engine)
    parts = [
        r"\documentclass[border=10pt]{standalone}",
        r"\usepackage{amsmath}",
        r"\usepackage{tikz}",
        r"\usetikzlibrary{arrows.meta}",
        r"\usepackage{chemfig}",
        r"\usepackage{mol2chemfig}",
    ]
    if cjk:
        if on_windows_tex:
            parts.append(r"\usepackage[fontset=windows]{ctex}")
        else:
            parts.append(r"\usepackage{ctex}")
    parts.append(r"\begin{document}")
    return "\n".join(parts) + "\n"


_POSTAMBLE = "\n\\end{document}\n"


def _run_latex(engine: str, tex_path: str) -> Optional[str]:
    """编译 .tex → PDF，返回 PDF 路径或 None。"""
    need_win = _is_windows_exe(engine) and sys.platform != "win32"
    work = os.path.dirname(os.path.abspath(tex_path))
    job = os.path.splitext(os.path.basename(tex_path))[0]

    def _w(p: str) -> str:
        return _to_win_path(p) if need_win else p

    try:
        subprocess.run(
            [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={_w(work)}",
                _w(tex_path),
            ],
            cwd=work,
            capture_output=True,
            timeout=120,
        )
    except Exception:
        return None
    pdf = os.path.join(work, f"{job}.pdf")
    return pdf if os.path.isfile(pdf) else None


def _compile_doc_to_png(latex_doc: str, dpi: int) -> Optional[bytes]:
    """编译完整 LaTeX 文档 → PNG 字节。"""
    engine = _find_latex_engine()
    if not engine:
        return None
    with tempfile.TemporaryDirectory(prefix="chemtex_") as work:
        tex_path = os.path.join(work, "fig.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_doc)
        pdf_path = _run_latex(engine, tex_path)
        if not pdf_path:
            return None
        # PDF → PNG：先 PyMuPDF（无外部依赖），再 pdftoppm
        png = _pdf_to_png_pymupdf(pdf_path, dpi)
        if png:
            return png
        return _pdf_to_png_pdftoppm(pdf_path, dpi, _is_windows_exe(engine))


def compile_tikz_to_png(code: str, title: str = "", dpi: int = 200) -> Optional[bytes]:
    """把 TikZ/chemfig 代码片段编译成 PNG。

    参数:
        code: chemfig 或 ``\\begin{tikzpicture}...`` 片段。
        title: 可选标题（图上方）。
        dpi: 输出分辨率。

    返回:
        PNG 字节流；任一环节失败返回 ``None``（调用方应降级为纯文本/代码块）。
    """
    if not code or not code.strip():
        return None
    body = (f"\\textbf{{{title}}}\\par\n" if title else "") + code

    # 两级重试：CJK 导言区 → 纯英文导言区
    for cjk in (True, False):
        doc = _preamble(cjk) + body + _POSTAMBLE
        png = _compile_doc_to_png(doc, dpi)
        if png:
            return png
    return None


if __name__ == "__main__":
    # 自检：探测后端 + 编译一个苯环 + CJK 能量图
    print("=== backends ===")
    for k, v in detect_backends().items():
        print(f"  {k}: {v}")
    print("\n=== compile benzene (chemfig) ===")
    png1 = compile_tikz_to_png(r"\chemfig{*6(------)}")
    print(f"  benzene PNG: {len(png1) if png1 else None} bytes")
    print("\n=== compile CJK tikz ===")
    png2 = compile_tikz_to_png(
        "\\begin{tikzpicture}\n"
        "  \\draw[->] (0,0) -- (5,0);\n"
        "  \\node[font=\\small] at (2.5,-0.3) {反应进程};\n"
        "  \\node at (2,2) {过渡态};\n"
        "\\end{tikzpicture}"
    )
    print(f"  CJK PNG: {len(png2) if png2 else None} bytes")
