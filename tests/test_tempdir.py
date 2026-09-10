# -*- coding: utf-8 -*-
"""tests/test_tempdir.py — 临时工作目录入口（`utils/tempdir.py`）。

背景：`api.py`（请求级目录）与 `utils/latex_compile.py`（LaTeX 中间文件）
原先直接用 `tempfile.TemporaryDirectory()`。受限环境（只读 /tmp 的容器、
权限受限的 CI）下有两个坑：
1. 系统 temp 根本不可写 → `mkdtemp` 出来的目录写不进文件；
2. 更隐蔽：目录建得出、却在 `__exit__` 清理时被拒 → **回答已经生成好**，
   整请求却因为"删临时文件失败"而 500（20260830 实测，22 个用例因此变红）。
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from utils import tempdir


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(tempdir.ENV_VAR, raising=False)


def _mkdtemp_is_writable(tmp_path) -> bool:
    """本环境里 `mkdtemp` 目录能否写文件？

    极少数受限环境下不能：Windows 上 `os.mkdir(path, 0o700)`（`mkdtemp`
    内部就是这么建的）会建出一个**不继承父级 ACE** 的目录，若进程跑在
    "靠继承拿权限"的受限身份下（本沙箱即如此），连属主都写不进去。
    这是环境特性、不是本项目缺陷——真实机器上 `mkdtemp` 必然可写。
    """
    d = tempfile.mkdtemp(dir=str(tmp_path))
    try:
        Path(d, "probe").write_text("x", encoding="utf-8")
        return True
    except OSError:
        return False


# ---------------------------------------------------------------- 父目录解析

def test_temp_parent_is_none_when_env_unset(clean_env):
    """没配就是 `None` —— 完全沿用系统临时目录（与改动前行为一致）。"""
    assert tempdir.temp_parent() is None


def test_temp_parent_uses_env_when_set(clean_env, tmp_path, monkeypatch):
    target = tmp_path / "chem-tmp"
    monkeypatch.setenv(tempdir.ENV_VAR, str(target))
    assert tempdir.temp_parent() == str(target)
    assert target.is_dir()          # 会自动创建


def test_temp_parent_falls_back_when_path_unusable(clean_env, monkeypatch):
    """配了一个建不出来的路径 → 退回系统默认，而不是让整个服务起不来。"""
    monkeypatch.setenv(tempdir.ENV_VAR, "Z:\\definitely\\not\\here\\chem")
    assert tempdir.temp_parent() is None


def test_temp_parent_ignores_blank(clean_env, monkeypatch):
    monkeypatch.setenv(tempdir.ENV_VAR, "   ")
    assert tempdir.temp_parent() is None


# ---------------------------------------------------------------- 工作目录

def test_work_dir_uses_env_parent(clean_env, tmp_path, monkeypatch):
    parent = tmp_path / "custom"
    monkeypatch.setenv(tempdir.ENV_VAR, str(parent))
    with tempdir.work_dir() as work:
        assert Path(work).parent == parent


def test_work_dir_yields_writable_dir_and_removes_it(clean_env, tmp_path,
                                                     monkeypatch):
    if not _mkdtemp_is_writable(tmp_path):
        pytest.skip("本环境的 mkdtemp 目录不可写（Windows 受限身份），"
                    "无法验证可写性；清理行为仍由其它用例覆盖")
    monkeypatch.setenv(tempdir.ENV_VAR, str(tmp_path))
    with tempdir.work_dir(prefix="chemtex_") as work:
        assert os.path.isdir(work)
        Path(work, "fig.tex").write_text("x", encoding="utf-8")   # 必须可写
        assert Path(work).name.startswith("chemtex_")
    assert not os.path.exists(work), "临时目录用完应删除"


def test_work_dir_cleanup_failure_does_not_raise(clean_env, tmp_path,
                                                 monkeypatch):
    """★ 核心回归：清理失败只记日志，不能让请求失败。

    受限环境里 `shutil.rmtree` 会被拒（只读挂载 / 沙箱 / 杀软占用）。
    调用方要的结果此时**已经算好了**，不该因为删不掉临时文件而整体失败。

    ⚠ 必须自己恢复 `shutil.rmtree`：用 `monkeypatch` 的话，撤销时机可能晚于
    pytest 内置 `tmp_path` 的清理，会把测试自身的收尾也炸掉（实测 teardown
    ERROR）。
    """
    monkeypatch.setenv(tempdir.ENV_VAR, str(tmp_path))
    real = shutil.rmtree
    tempdir.shutil.rmtree = lambda *a, **k: (_ for _ in ()).throw(
        PermissionError(13, "Access is denied"))
    try:
        with tempdir.work_dir(prefix="locked_") as work:
            assert os.path.isdir(work)
        # 走到这里没抛异常即为通过
    finally:
        tempdir.shutil.rmtree = real


def test_work_dir_cleanup_failure_logs(clean_env, tmp_path, monkeypatch,
                                       capsys):
    """清理失败要留痕（否则临时目录悄悄堆积、排查无线索）。"""
    monkeypatch.setenv(tempdir.ENV_VAR, str(tmp_path))
    real = shutil.rmtree
    tempdir.shutil.rmtree = lambda *a, **k: (_ for _ in ()).throw(
        PermissionError(13, "Access is denied"))
    try:
        with tempdir.work_dir(prefix="locked_"):
            pass
    finally:
        tempdir.shutil.rmtree = real
    assert "临时目录清理失败" in capsys.readouterr().out


def test_work_dir_cleans_up_readonly_files(clean_env, tmp_path, monkeypatch):
    """只读文件（MiKTeX 产物常见）也要能删干净，不留垃圾。"""
    if not _mkdtemp_is_writable(tmp_path):
        pytest.skip("本环境的 mkdtemp 目录不可写（Windows 受限身份）")
    monkeypatch.setenv(tempdir.ENV_VAR, str(tmp_path))
    with tempdir.work_dir(prefix="ro_") as work:
        f = Path(work, "fig.log")
        f.write_text("x", encoding="utf-8")
        os.chmod(f, 0o444)
    assert not os.path.exists(work)


# ---------------------------------------------------------------- 可写性探测

def test_probe_returns_none_when_writable(clean_env, tmp_path, monkeypatch):
    if not _mkdtemp_is_writable(tmp_path):
        pytest.skip("本环境的 mkdtemp 目录不可写（Windows 受限身份）")
    monkeypatch.setenv(tempdir.ENV_VAR, str(tmp_path))
    assert tempdir.probe() is None


def test_probe_reports_problem_when_unwritable(clean_env, tmp_path, monkeypatch):
    """★ 探测必须把"目录建得出但写不进"这种情形说清楚。

    这是 20260830 排查中最费时的一类：`mkdtemp` 成功、写入被拒，症状只是
    回答里冒出"（图示未能渲染）"，完全指不到临时目录。探测要给出可行动的
    提示（点名 `CHEM_AGENT_TMPDIR`）。
    """
    monkeypatch.setenv(tempdir.ENV_VAR, str(tmp_path))
    real_rmtree = shutil.rmtree
    tempdir.shutil.rmtree = lambda *a, **k: None        # 别让清理干扰
    try:
        monkeypatch.setattr("builtins.open",
                            lambda *a, **k: (_ for _ in ()).throw(
                                PermissionError(13, "Access is denied")))
        problem = tempdir.probe()
    finally:
        tempdir.shutil.rmtree = real_rmtree
        monkeypatch.undo()
    assert problem is not None
    assert tempdir.ENV_VAR in problem
    assert "不可写" in problem


def test_probe_reports_when_mkdtemp_fails(clean_env, monkeypatch):
    """连目录都建不出来（只读挂载）时也要给出可行动提示。"""
    monkeypatch.setattr(tempfile, "mkdtemp",
                        lambda *a, **k: (_ for _ in ()).throw(
                            OSError(30, "Read-only file system")))
    problem = tempdir.probe()
    assert problem is not None
    assert "无法创建" in problem
    assert tempdir.ENV_VAR in problem


def test_work_dir_warns_once_when_unwritable(clean_env, tmp_path, monkeypatch,
                                             capsys):
    monkeypatch.setenv(tempdir.ENV_VAR, str(tmp_path))
    tempdir.reset_probe_for_tests()
    monkeypatch.setattr(tempdir, "probe", lambda: "临时目录不可写：X")
    with tempdir.work_dir():
        with tempdir.work_dir():
            pass
    out = capsys.readouterr().out
    assert out.count("临时目录不可写") == 1, "告警应只出现一次，别刷屏"


def test_tempdir_logs_are_gbk_encodable(clean_env, tmp_path, monkeypatch):
    """★ 日志文案必须能被 GBK 编码：Windows 默认代码页下 `print` 遇到
    转不了的字符（`⚠` `✓` `⁻` …）会抛 `UnicodeEncodeError`。

    20260830 实测：启动横幅里一个 `⚠` 让 uvicorn 直接
    `Application startup failed. Exiting.` —— **服务完全起不来**。
    这里把该模块会打印的所有文案都过一遍 GBK 编码。
    """
    monkeypatch.setenv(tempdir.ENV_VAR, str(tmp_path))
    texts = [tempdir.describe(), tempdir.probe() or "",
             f"[tempdir] 警告：{tempdir.probe() or ''}"]
    for t in texts:
        t.encode("gbk")           # 编码不了即抛，用例变红


# ---------------------------------------------------------------- 诊断

def test_describe_mentions_default_and_override(clean_env):
    text = tempdir.describe()
    assert tempdir.ENV_VAR in text
    assert "系统默认" in text


def test_describe_shows_configured_path(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv(tempdir.ENV_VAR, str(tmp_path))
    assert str(tmp_path) in tempdir.describe()


# ---------------------------------------------------------------- 接线

def test_api_and_latex_compile_use_work_dir():
    """两个调用点都必须走 `work_dir()`，不能再直接用 `tempfile`。

    直接 `tempfile.TemporaryDirectory()` 会在受限环境下把"清理失败"变成
    请求失败；这是回归防线（改回去就红）。
    """
    import api
    import utils.latex_compile as lc

    assert api.work_dir is tempdir.work_dir
    assert lc.work_dir is tempdir.work_dir
    for mod in (api, lc):
        assert not hasattr(mod, "tempfile"), \
            f"{mod.__name__} 又直接用 tempfile 了"
