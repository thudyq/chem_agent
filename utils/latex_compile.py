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

安全（安全审查 R1，20260911）
----------------------------
编译的是**模型生成**的片段，而模型输出受用户提问影响。TeX 的
``\\input`` / ``\\openin`` / ``\\read`` 能读**任意可读文件**（实测：相对 ``..``
与绝对路径都能读），内容会排版进图里、随图片回传给访客——项目根目录的
``.env`` 正是最值钱的目标。本模块加两道防线：

1. **源码闸门**：片段里出现文件 I/O / 包加载原语（``\\input`` ``\\openin``
   ``\\write`` ``\\catcode`` ``\\csname`` …）直接拒绝编译（`forbidden_token`）。
   拦得住现实攻击，但拦不住刻意构造的技巧，**不是**唯一防线。
2. **引擎级加固**（`security_status`）：``-no-shell-escape`` / ``-disable-write18``
   关掉 ``\\write18``；``--cnf-line=openin_any=p`` 把文件读取限制在工作目录内
   （kpathsea 认，**MiKTeX 不认**）。这些 flag 只有在**实测**"正常文档仍能编译"
   之后才会启用，并且会实测"越界读文件是否真的被拒"，把结论写进启动日志。

★ 唯一完整的解法是**操作系统级隔离**（独立用户 / 容器 / 只读挂载，让编译进程
读不到 ``.env``）——那是部署要求，代码里做不到。详见
``instructions/Security-Review.md`` R1。

公共 API：``compile_tikz_to_png(code, title="", dpi=200) -> bytes | None``
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import Optional

from utils.tempdir import work_dir

__all__ = ["compile_tikz_to_png", "detect_backends", "forbidden_token",
           "hardening_flags", "security_status"]


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
    """返回首个可启动的 LaTeX 引擎路径。优先 xelatex（CJK 友好）。

    不依赖 PATH 的兜底：systemd 服务环境 PATH 可能不含 /usr/bin，导致
    shutil.which 找不到 xelatex——最后再试几个常见固定绝对路径（Linux/
    macOS 的 texlive）与 MiKTeX 路径，保证服务进程也能编译。
    """
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
    # PATH 兜底：常见绝对路径（服务进程 PATH 受限时仍能命中）
    fixed = ["/usr/bin", "/usr/local/bin", "/opt/local/bin", "/opt/homebrew/bin",
             "/bin", "/sbin"]
    for d in fixed:
        for n in names:
            cand = os.path.join(d, n)
            if _try_exec(cand):
                return cand
    return None


# ---------------------------------------------------------------------------
# 安全加固（安全审查 R1）：源码闸门 + 引擎级文件访问限制
# ---------------------------------------------------------------------------

_CS_RE = re.compile(r"\\([A-Za-z]+)")

# 片段（模型生成的图代码 + 图标题）里**不允许**出现的控制序列名。
# 覆盖三类：
#   * 读文件：input / include / openin / read / readline / verbatiminput /
#     lstinputlisting / inputminted / subfile / InputIfFileExists
#   * 写文件与改 catcode（用来绕过上面的黑名单）：write（含 \write18）/
#     openout / newwrite / catcode / csname / makeatletter / everyeof
#   * 包加载与其它读取通道：usepackage / RequirePackage / documentclass /
#     special / directlua / latelua
# 注意 `\inputencoding` 这类**仅前缀相同**的宏不会命中（按整名匹配）。
_FORBIDDEN_CS = frozenset({
    "input", "include", "includeonly", "InputIfFileExists", "subfile",
    "openin", "openout", "read", "readline", "newread", "newwrite",
    "write", "catcode", "csname", "endcsname", "makeatletter",
    "makeatother", "everyeof",
    "usepackage", "RequirePackage", "documentclass",
    "special", "directlua", "latelua", "luadirect",
    "verbatiminput", "lstinputlisting", "inputminted",
})


def forbidden_token(code: Optional[str]) -> Optional[str]:
    """返回片段里第一个被禁止的控制序列名；没有则返回 ``None``。

    为什么按"控制序列整名"匹配：``\\input`` 后面直接跟 ``{``，用词边界正则会把
    ``\\write18`` 漏掉（``18`` 是词字符），而 ``\\inputencoding`` 又会被误伤。
    这里先把 ``\\name`` 里的字母整段取出来再查集合，两个问题一起解决。
    """
    if not code:
        return None
    for m in _CS_RE.finditer(code):
        name = m.group(1)
        if name in _FORBIDDEN_CS:
            return name
    return None


# 引擎级加固候选：不同引擎认的写法不一样（MiKTeX 只认 -disable-write18，
# kpathsea/TeX Live 才认 --cnf-line），所以逐个探测后再用。
_HARDENING_CANDIDATES = (
    "-no-shell-escape",           # web2c / TeX Live：关掉 \write18
    "-disable-write18",           # MiKTeX：同上
    "--cnf-line=openin_any=p",    # kpathsea：读文件只允许工作目录树内
)

# ★ 与 openin_any=p 配套的硬约束：命令行里出现**绝对路径**时，kpathsea 会拒绝
# 读入主文件（TeX Live 邮件列表 2020-09 确认："latex full_path_of_sample2e.tex
# does not work"，而 kpse 自己找到的系统宏包仍可读）。所以 `_run_latex` 必须用
# **相对文件名 + cwd=工作目录**调用；否则加固的结果是"什么都编译不出来"。
# https://tug.org/pipermail/tex-live/2020-September/046128.html


@lru_cache(maxsize=8)
def _engine_accepts_flag(engine: str, flag: str) -> bool:
    """引擎是否接受该命令行选项（用 --version 探，未知选项会返回非 0）。"""
    try:
        r = subprocess.run([engine, flag, "--version"],
                           capture_output=True, timeout=20)
    except Exception:
        return False
    return r.returncode == 0


@lru_cache(maxsize=8)
def hardening_flags(engine: Optional[str] = None) -> tuple:
    """当前引擎实测可用的加固选项（**热路径**用：只做廉价的选项探测）。

    探不到任何选项就返回空元组——此时只剩源码闸门，`security_status()` 会把
    这件事说清楚。
    """
    engine = engine or _find_latex_engine()
    if not engine:
        return ()
    return tuple(f for f in _HARDENING_CANDIDATES
                 if _engine_accepts_flag(engine, f))


def _verify_hardening(engine: str, flags: tuple) -> dict:
    """**慢**（要跑 3 次真实编译）：实测加固到底有没有生效。

    只由 `--security-check` / `security_status(verify=True)` 触发，**不进热路径**
    （本机 MiKTeX 实测：用项目导言区编译一次约 5~16 秒）。三份文档：
    1. 正常小图（项目自己的导言区）带 flag → 必须能编译；
    2. 对照：``\\input{../canary.tex}`` 不带 flag → 必须能编译（证明 canary 可读）；
    3. 同样文档带 flag → 必须**失败**才算"文件读取被限制"。
    """
    body = "\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}"
    normal = _preamble(True, False) + body + _POSTAMBLE
    tag = os.urandom(6).hex()
    try:
        with work_dir(prefix="chemsec_work_") as work, \
                work_dir(prefix="chemsec_key_") as keydir:
            with open(os.path.join(keydir, "canary.tex"), "w",
                      encoding="utf-8") as f:
                f.write("\\def\\CANARYTOKEN{OK}\n")
            rel = "../" + os.path.basename(keydir) + "/canary.tex"
            attack = (_preamble(True, False)
                      + "\\input{" + rel + "}\n"
                      + "\\begin{tikzpicture}\\draw (0,0)--(1,0);"
                        "\\node at (0.5,0.4) {\\CANARYTOKEN};"
                        "\\end{tikzpicture}"
                      + _POSTAMBLE)

            def run(doc: str, fl: tuple, job: str) -> bool:
                tex = os.path.join(work, job + ".tex")
                with open(tex, "w", encoding="utf-8") as f:
                    f.write(doc)
                return _run_latex(engine, tex, fl) is not None

            ok_normal = run(normal, flags, "normal" + tag)
            ok_plain = run(attack, (), "plain" + tag)
            ok_flagged = run(attack, flags, "flag" + tag)
    except OSError as e:
        return {"file_read_restricted": None,
                "note": f"无法自检（临时目录不可用：{e}）"}

    if not ok_plain:
        return {"file_read_restricted": None,
                "note": "无法自检（对照文档本身就编译不过）"}
    if not ok_normal:
        return {"file_read_restricted": None,
                "note": "★ 加固选项会让正常文档编译失败 —— 已放弃这些选项，"
                        "只剩源码闸门"}
    return {"file_read_restricted": not ok_flagged,
            "note": ("越界读文件已实测被拒"
                     if not ok_flagged else
                     "★ 引擎不支持文件访问限制，越界读文件仍可成功 —— 请用"
                     "独立用户/容器运行编译，使其读不到 .env")}


@lru_cache(maxsize=8)
def security_status(engine: Optional[str] = None, verify: bool = False) -> dict:
    """当前进程的 LaTeX 加固状态，可直接打进日志。

    返回 ``{"engine", "flags", "file_read_restricted", "note"}``；
    ``file_read_restricted`` 的 ``None`` 表示"没做实测"或"无法自检"——只有
    ``verify=True``（或跑 ``python -m utils.latex_compile --security-check``）
    才会真的去读一个越界文件来验证。
    """
    engine = engine or _find_latex_engine()
    flags = list(hardening_flags(engine))
    if not engine:
        return {"engine": None, "flags": flags, "file_read_restricted": None,
                "note": "未找到 LaTeX 引擎"}
    if not verify:
        note = ("已启用: " + " ".join(flags)) if flags else "引擎不接受任何加固选项"
        note += "；确认文件读取是否真被拒请跑 --security-check"
        return {"engine": engine, "flags": flags,
                "file_read_restricted": None, "note": note}
    return {"engine": engine, "flags": flags, **_verify_hardening(engine, tuple(flags))}


def _import_fitz():
    """PyMuPDF 模块导入：pymupdf（>=1.28）优先，回退旧别名 fitz。"""
    try:
        import pymupdf as fitz
        return fitz
    except ImportError:
        pass
    try:
        import fitz
        return fitz
    except ImportError:
        return None


def _pdf_to_png_pymupdf(pdf_path: str, dpi: int) -> Optional[bytes]:
    fitz = _import_fitz()
    if fitz is None:
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


def _pdf_to_png_pdftoppm(pdf_path: str, dpi: int) -> Optional[bytes]:
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
    return _import_fitz() is not None


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

def _preamble(cjk: bool, uses_chemfig: bool = False) -> str:
    """生成导言区。cjk=True 时加 ctex（Windows 用 fontset=windows）。

    uses_chemfig：代码块是否用 \\chemfig{...}/\\schemestart（本项目的结构式/
    机理渲染器输出的是纯 tikzpicture，不需要 chemfig；只有 legacy chemfig
    块才引入）。mol2chemfig 包已随 mol2chemfigPy3 删除，不再引用——否则
    服务器 TeX 无该包时每次编译都失败（附件为空、文本留裸 TikZ）。
    """
    engine = _find_latex_engine() or ""
    on_windows_tex = _is_windows_exe(engine)
    parts = [
        r"\documentclass[border=20pt]{standalone}",
        r"\usepackage{amsmath}",
        r"\usepackage{tikz}",
        r"\usetikzlibrary{arrows.meta}",
    ]
    if uses_chemfig:
        parts.append(r"\usepackage{chemfig}")
    if cjk:
        if on_windows_tex:
            parts.append(r"\usepackage[fontset=windows]{ctex}")
        else:
            # Linux/macOS：ctex 默认 fandol，冷僻字缺字形。见到有全量覆盖的
            # Noto CJK 就优先用（缺字体则静默回退，不影响编译）。
            parts.append(r"\usepackage{ctex}")
            parts.append(_CJK_FULL_COVERAGE)
    parts.append(r"\begin{document}")
    return "\n".join(parts) + "\n"


_POSTAMBLE = "\n\\end{document}\n"

# 非 Windows 下 ctex 默认 fontset=fandol，Fandol 字库覆盖不全，冷僻字
# （如“鎓”U+93D3）会缺字形、被 xelatex 渲染成占位符。这里在系统装有
# 覆盖 GB18030 全量的 Noto CJK 时改用它；\IfFontExistsTF 守卫保证字体
# 不存在时静默回退到 ctex 默认字体，绝不因此让编译失败。
_CJK_FULL_COVERAGE = (
    "\\IfFontExistsTF{Noto Serif CJK SC}{\\setCJKmainfont{Noto Serif CJK SC}}{}\n"
    "\\IfFontExistsTF{Noto Sans CJK SC}{\\setCJKsansfont{Noto Sans CJK SC}}{}\n"
)


def _run_latex(engine: str, tex_path: str,
               flags: tuple = ()) -> Optional[str]:
    """编译 .tex → PDF，返回 PDF 路径或 None。

    `flags` 来自 `hardening_flags()`：只有引擎实测接受的加固选项才传进来。

    ★ 调用形态有硬约束：**原生引擎必须给相对文件名 + `cwd=工作目录`**。
    因为 `--cnf-line=openin_any=p` 会让 kpathsea 拒绝读取命令行上的绝对路径
    主文件（系统宏包仍由 kpse 找到、不受影响）。写成绝对路径的话，加固的结果
    就是"每张图都编译不出来"。
    WSL 里调 Windows 的 `.exe` 是例外（interop 的 cwd 不可靠），仍给 Windows
    绝对路径——那条路走不到 kpathsea 的 `openin_any`，由源码闸门兜底。
    """
    need_win = _is_windows_exe(engine) and sys.platform != "win32"
    work = os.path.dirname(os.path.abspath(tex_path))
    job = os.path.splitext(os.path.basename(tex_path))[0]

    if need_win:
        out_arg = f"-output-directory={_to_win_path(work)}"
        tex_arg = _to_win_path(tex_path)
    else:
        out_arg = "-output-directory=."
        tex_arg = os.path.basename(tex_path)

    try:
        subprocess.run(
            [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                *flags,
                out_arg,
                tex_arg,
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
    """编译完整 LaTeX 文档 → PNG 字节。

    ★ 本函数的契约是"失败返回 None"，所以**落盘也要兜住 OSError**：
    临时目录不可写（只读 `/tmp` 的容器、受限身份下的沙箱）时 `open(...,'w')`
    会抛 PermissionError。以前它会一路冒到调用方，Web 路径有 try/except 还好，
    Streamlit 路径直接把整页渲染成一段 traceback（实测踩到）。
    """
    engine = _find_latex_engine()
    if not engine:
        return None
    flags = hardening_flags(engine)
    with work_dir(prefix="chemtex_") as work:
        tex_path = os.path.join(work, "fig.tex")
        try:
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(latex_doc)
        except OSError as e:
            print(f"[latex] 临时目录不可写，跳过编译（可设 CHEM_AGENT_TMPDIR）: {e}")
            return None
        pdf_path = _run_latex(engine, tex_path, flags)
        if not pdf_path:
            return None
        # PDF → PNG：先 PyMuPDF（无外部依赖），再 pdftoppm
        png = _pdf_to_png_pymupdf(pdf_path, dpi)
        if png:
            return png
        return _pdf_to_png_pdftoppm(pdf_path, dpi)


def compile_tikz_to_png(code: str, title: str = "", dpi: int = 300) -> Optional[bytes]:
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
    # ★ 源码闸门：图代码与**图标题**都过一遍（标题会被塞进 \textbf{...}，
    # 同样能读出文件）。命中即拒绝编译，调用方降级为展示 LaTeX 源码。
    bad = forbidden_token(code) or forbidden_token(title)
    if bad:
        print(f"[latex] 拒绝编译：图代码/标题含被禁止的文件 I/O 命令 \\{bad}"
              f"（安全审查 R1；已降级为 LaTeX 源码显示）")
        return None
    body = (f"\\textbf{{{title}}}\\par\n" if title else "") + code
    uses_chemfig = "\\chemfig" in code or "\\schemestart" in code

    # 两级重试：CJK 导言区 → 纯英文导言区
    for cjk in (True, False):
        doc = _preamble(cjk, uses_chemfig) + body + _POSTAMBLE
        png = _compile_doc_to_png(doc, dpi)
        if png:
            return png
    return None


if __name__ == "__main__":
    # 自检：探测后端 + 编译一个苯环 + CJK 能量图
    # `--security-check` 只跑安全自检（部署后用它确认加固到底有没有生效）：
    # 会真的去读一个工作目录外的 canary 文件，所以慢（每次真实编译若干秒）。
    if "--security-check" in sys.argv:
        import json as _json
        st = security_status(verify=True)
        print(_json.dumps(st, ensure_ascii=False, indent=2))
        if "--strict" in sys.argv and st.get("file_read_restricted") is not True:
            raise SystemExit(1)
        raise SystemExit(0)

    print("=== security ===")
    print(" ", security_status(), "\n")
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
