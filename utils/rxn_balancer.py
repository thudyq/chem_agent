# -*- coding: utf-8 -*-
"""utils/rxn_balancer.py — SynRBL 配平兜底（best-effort 自动补全物种）。

当 REACTION 标记通过校验器化学校验（T2-3）被拦截（两侧原子不守恒）时，
用 SynRBL（J Cheminform 2024, MIT）自动补全缺失物种，把配平后的方程式
注入 P2 修正 prompt——LLM 直接照抄即可，不必自己推导配平。

实测能力边界（20260814，synrbl 1.0.6）：
- 能：补全缺失物种——酯化 ``CC(=O)O.CCO>>CC(=O)OCC`` → 自动补 H2O 得
  ``CC(=O)O.CCO>>CC(=O)OCC.O``；乙醇氧化 ``CCO.O=O>>CC(=O)O`` →
  ``CCO.O=O>>CC(=O)O.O``。这类"漏写 H2O/副产物"正是化学校验拦截的高频失败。
- 不能：修正计量系数错误（``CCO>>CCOCC`` 需 2CCO，返回原样；乙醇+KMnO4
  复杂氧化返回原样）——SynRBL 的 MCS 补全面向"缺物种"，不重写系数。
- 性能：首次 import ~23s（加载模型/数据），懒加载单例只付一次；MCS 搜索
  最坏 ~50s（系数错误场景白耗），快路径 0.1~1.5s——接入带超时与失败静默。
- 输入限制：所有物种须为合法 SMILES（KMnO4/H2SO4 等教科书化学式非 SMILES，
  跳过不喂）；SynRBL 输出仍须通过本项目守恒复核才采用。

设计：
- 懒加载单例 ``Balancer``（线程锁保护初始化与调用，joblib 内部并行不冲突）；
- 调用带超时（``_TIMEOUT``），超时返回 None 并在后台线程完成前置 ``_stuck``
  跳过后续调用（防堆积在 ~50s 的 MCS 搜索上），线程结束后自动复位；
- 任何异常 / 超时 / 不可用（未安装）一律返回 None，不影响原管线。

用法:
    from utils.rxn_balancer import balance_reaction
    hint = balance_reaction("CCO;O=O", "CC(=O)O")   # → 配平后的提示文本或 None
"""

import threading

# SynRBL 可用性：懒加载探测——首次需要配平时才 import（~23s 模型/数据加载），
# 未安装/导入失败则降级为不可用（None），不影响原管线。
_SYNRBL_OK = False
_SynRBLBalancer = None

from core.tag_validator import (
    _balance_reason,
    _parse_mol,
    _split_multi_coeff,
    _sum_species,
)

# 单次 SynRBL 配平的最长等待（秒）。快路径 0.1~1.5s，MCS 搜索最坏 ~50s——
# 系数错误场景（SynRBL 本来就改不了）白耗，超时后放弃并跳过后续调用。
_TIMEOUT = 20.0

_balancer = None
_lock = threading.Lock()
_stuck = False  # 上一次调用超时，后台线程仍在跑：跳过后续调用直至其结束


def _load_synrbl():
    """尝试导入 synrbl Balancer；成功置 _SYNRBL_OK=True。"""
    global _SynRBLBalancer, _SYNRBL_OK
    if _SynRBLBalancer is not None:
        return True
    try:
        from synrbl import Balancer as B
    except Exception:  # pragma: no cover - 取决于部署环境
        _SYNRBL_OK = False
        return False
    _SynRBLBalancer = B
    _SYNRBL_OK = True
    return True


def _get_balancer():
    """懒加载单例 Balancer（线程安全）。"""
    global _balancer
    if _balancer is None:
        with _lock:
            if _balancer is None:
                if not _load_synrbl():
                    return None
                # n_jobs=1：序列化 joblib，避免与 API 并发/流式线程叠加
                _balancer = _SynRBLBalancer(n_jobs=1, confidence_threshold=0)
    return _balancer


def _run_with_timeout(fn, timeout: float):
    """在守护线程中执行 fn，超时返回 None 并标记 _stuck。

    超时后线程继续在后台跑（SynRBL 是 CPU 密集，无法安全强杀），
    期间 _stuck=True 使新调用直接跳过；由独立 reaper 线程等原线程
    结束后复位 _stuck（避免与原线程 finally 的复位竞态）。
    """
    global _stuck
    if _stuck:
        return None
    result = {}
    state = {"done": False}

    def worker():
        try:
            result["v"] = fn()
        except Exception:
            result["e"] = True
        finally:
            state["done"] = True

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if not state["done"]:
        _stuck = True

        def reaper():
            t.join()
            global _stuck
            _stuck = False

        threading.Thread(target=reaper, daemon=True).start()
        return None
    if "e" in result:
        return None
    return result.get("v")


def _single_mol_smiles(smi: str) -> bool:
    """SMILES-only 门槛：合法 SMILES 且为单组分（不含 .）。

    SynRBL 的输入用 . 连接物种，物种内部再含 . 会歧义（如
    [K+].[O-][Mn](=O)(=O)=O）；教科文化学式（KMnO4/H2SO4/H2O）不是合法
    SMILES，同样被门槛拦下——这类物种 SynRBL 也无法处理，跳过不喂。
    """
    return bool(smi) and "." not in smi and _parse_mol(smi) is not None


def balance_reaction(left_seg: str, right_seg: str) -> str | None:
    """对 REACTION 两侧物种段做 SynRBL 补全配平，返回可直接采用的提示文本。

    left_seg / right_seg：REACTION 标记的 args[0]/args[1]（"物种;物种"，
    可带系数前缀，如 "CCO;O=O"）。全部物种须为单组分合法 SMILES 才尝试。
    返回形如：
        SynRBL 自动配平参考（已通过守恒校验）：反应物 CCO;O=O，生成物
        CC(=O)O;O（把 . 分隔的补全物种并入你的 REACTION，保持系数一致）
    失败/超时/门槛不过返回 None（调用方沿用原修正提示）。
    """
    if not _load_synrbl():
        return None
    left = [smi for _, smi in _split_multi_coeff(left_seg or "")]
    right = [smi for _, smi in _split_multi_coeff(right_seg or "")]
    if not left or not right:
        return None
    if not all(_single_mol_smiles(s) for s in left + right):
        return None
    rxn = ".".join(left) + ">>" + ".".join(right)

    def call():
        return _get_balancer().rebalance(rxn)

    res = _run_with_timeout(call, _TIMEOUT)
    if not res:
        return None
    balanced = res[0] if isinstance(res, list) else res
    if not isinstance(balanced, str) or ">>" not in balanced:
        return None
    # 拆回两侧物种，过本项目守恒复核（SynRBL 结果也须守恒才采用）
    try:
        b_left, b_right = balanced.split(">>", 1)
        b_left = [s for s in b_left.split(".") if s]
        b_right = [s for s in b_right.split(".") if s]
        if not b_left or not b_right:
            return None
        if ".".join(b_left) == ".".join(left) and ".".join(b_right) == ".".join(right):
            return None  # 原样返回（系数错误类，无补全信息）
        lsum = _sum_species([(1, s) for s in b_left])
        rsum = _sum_species([(1, s) for s in b_right])
        if _balance_reason(lsum, rsum, strict_h=True, check_charge=True,
                           step="配平参考"):
            return None  # 输出仍不守恒：不采用
    except Exception:
        return None
    return (
        "SynRBL 自动配平参考（已通过守恒校验）：反应物"
        + "；".join(b_left)
        + "，生成物 "
        + "；".join(b_right)
        + "（把补全的物种并入你的 REACTION，保持两侧原子守恒即可）"
    )


def balancer_available() -> bool:
    """SynRBL 是否可用（测试/日志用；触发懒加载探测）。"""
    return _load_synrbl() and not _stuck