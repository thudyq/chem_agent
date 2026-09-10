# -*- coding: utf-8 -*-
"""utils/tempdir.py — 统一的"临时工作目录"入口。

为什么需要它
------------
产品代码里若干处需要真实磁盘临时目录（LaTeX 编译中间文件、请求级工作目录）。
直接用 `tempfile.TemporaryDirectory()` 会用系统临时目录（Windows 的
`%TEMP%`、Linux 的 `/tmp`），而受限环境（容器 / CI / 只读 `/tmp` 的运行时）
下系统临时目录**不可写**，表现为：

    PermissionError: [WinError 5] 拒绝访问:
    'C:\\Users\\...\\AppData\\Local\\Temp\\tmpab12cd'

更隐蔽的是：`mkdtemp()` 有时能建出目录、却在**清理阶段**（`shutil.rmtree`）
才被拒——于是"回答已经生成好"却因为收尾删目录失败而整请求失败。

约定
----
* 设了 `CHEM_AGENT_TMPDIR` → 所有临时工作目录都建在它下面（由管理员指定，
  例如容器里挂一个可写的 `/var/tmp/chem_agent`）。
* 未设 → **完全沿用系统临时目录**，行为与改动前一致。

`CHEM_AGENT_TMPDIR` 只影响本项目的临时工作目录，不改变进程全局 `TMPDIR`
语义（不去动 `tempfile.tempdir`，避免影响第三方库）。

兼容性：本模块只用 Python 3.9 就有的 API（`TemporaryDirectory` 的
`ignore_cleanup_errors` 要 3.10，故不用）。
"""

import os
import shutil
import stat
import tempfile
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

__all__ = ["ENV_VAR", "temp_parent", "work_dir", "describe", "probe",
           "reset_probe_for_tests"]

ENV_VAR = "CHEM_AGENT_TMPDIR"

_probe_lock = threading.Lock()
_probe_done = False


def temp_parent() -> Optional[str]:
    """返回临时目录的父目录；未配置或建不出来时返回 `None`（= 用系统默认）。"""
    raw = (os.environ.get(ENV_VAR) or "").strip()
    if not raw:
        return None
    try:
        os.makedirs(raw, exist_ok=True)
    except OSError as e:                      # 配了但建不出来 → 退回系统默认
        print(f"[tempdir] {ENV_VAR}={raw} 不可用（{e}），改用系统临时目录")
        return None
    return raw


def _rmtree_force(path: str) -> None:
    """删目录树：Windows 上先去掉只读属性再删，失败只记日志不抛。"""
    def _onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    try:
        shutil.rmtree(path, onerror=_onerror)
    except OSError as e:
        print(f"[tempdir] 临时目录清理失败（不影响本次结果）: {path} ({e})")


def probe() -> Optional[str]:
    """探测临时目录是否**真的可写**，返回错误说明；正常返回 `None`。

    为什么值得探测：目录"建得出来"和"写得进去"是两件事。极端情况下
    （例如 Windows 上以受限身份运行、而 `mkdtemp` 建出的 `0o700` 目录不
    继承父级 ACE）创建会成功、写入却 `PermissionError`——此时 LaTeX 编译
    直接失败，用户只看到回答里冒出"（图示未能渲染）"，排查方向完全被带偏。
    宁可在启动/首次使用时给一句明确的话。
    """
    parent = temp_parent()
    try:
        probe_path = tempfile.mkdtemp(prefix="chemprobe_", dir=parent)
    except OSError as e:
        return f"临时目录无法创建：{e}（可用 {ENV_VAR} 指定可写目录）"
    try:
        with open(os.path.join(probe_path, "probe.txt"), "w",
                  encoding="utf-8") as f:
            f.write("ok")
    except OSError as e:
        return (f"临时目录不可写：{probe_path}（{e}）；"
                f"LaTeX 编译与附件落盘会失败，请用 {ENV_VAR} 指定可写目录")
    finally:
        _rmtree_force(probe_path)
    return None


def _warn_once_if_unwritable() -> None:
    """首次调用时探一次可写性；不可写就打印一行明确告警（进程内只报一次）。"""
    global _probe_done
    if _probe_done:
        return
    with _probe_lock:
        if _probe_done:
            return
        _probe_done = True
    problem = probe()
    if problem:
        print(f"[tempdir] 警告：{problem}")


@contextmanager
def work_dir(prefix: str = "chem_") -> Iterator[str]:
    """临时工作目录上下文管理器（用完自动递归删除）。

    参数:
        prefix: 目录名前缀，便于在临时目录里辨认来源（如 `chemtex_`）。

    产出:
        可写的目录绝对路径。

    收尾删除失败（只读挂载 / 杀软占用 / 沙箱限制）**不抛异常**：调用方要的
    结果已经算好了，不该因为删不掉临时文件而整请求失败——只留一行日志。
    """
    _warn_once_if_unwritable()
    path = tempfile.mkdtemp(prefix=prefix, dir=temp_parent())
    try:
        yield path
    finally:
        _rmtree_force(path)


def reset_probe_for_tests() -> None:
    """让可写性探测重新跑一次（测试隔离用）。"""
    global _probe_done
    _probe_done = False


def describe() -> str:
    """一行人类可读的临时目录说明（启动诊断用）。"""
    parent = temp_parent()
    if parent:
        return f"{parent}（来自 {ENV_VAR}）"
    return f"{tempfile.gettempdir()}（系统默认；可设 {ENV_VAR} 覆盖）"
