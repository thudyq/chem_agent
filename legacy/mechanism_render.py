# -*- coding: utf-8 -*-
"""
legacy/mechanism_render.py
      =========================
反应机理示意图：用 matplotlib 绘制带曲线电子流动箭头的机理图。

2.3 任务：
    - draw_mechanism: 按机理类型（SN2/E2/SN1/E1）绘制参数化示意图。
    - 曲线箭头用 FancyArrowPatch 标注电子对转移（成键/断键）。

设计：只管绘图，不导入 chem_agent（避免循环导入）；matplotlib 懒导入。
labels 由 chem_agent 的 LLM 机理识别提供（用户的化合物角色）。
机理类型暂未内置时返回 None，调用方优雅跳过。
"""

import sys
from io import BytesIO
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# 支持的机理类型
SUPPORTED = {"SN2", "E2", "SN1", "E1"}


def _new_axis(figsize=(9.5, 4.5)):
    """新建无坐标轴的绘图区，返回 (fig, ax)。matplotlib 不可用返回 None。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, None
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    return fig, ax


def _curved_arrow(ax, start, end, rad=-0.35, color="#C62828"):
    """画一条曲线电子流动箭头（红色，表示电子对转移）。"""
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="->,head_width=0.45,head_length=0.6",
        color=color, mutation_scale=14, linewidth=1.8, zorder=4,
    ))


def _to_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def draw_sn2(labels):
    """SN2：亲核试剂背面进攻 + 离去基团协同离去，一步过渡态。"""
    nu = labels.get("nucleophile", "Nu:")
    sub = labels.get("substrate", "R-LG")
    lg = labels.get("leaving_group", "LG")
    prod = labels.get("product", "Nu-R")
    fig, ax = _new_axis()
    if fig is None:
        return None

    ax.text(1.0, 3.2, f"{nu}", ha="center", fontsize=13, fontweight="bold", color="#1565C0")
    ax.text(1.0, 2.5, "亲核试剂", ha="center", fontsize=9, color="#666")
    ax.text(5.0, 3.2, f"{sub}", ha="center", fontsize=13, fontweight="bold")
    ax.text(5.0, 2.5, "底物（C-LG）", ha="center", fontsize=9, color="#666")
    ax.text(9.0, 3.2, f"{prod}  +  :{lg}", ha="center", fontsize=13, fontweight="bold", color="#2E7D32")
    ax.text(9.0, 2.5, "产物 + 离去基", ha="center", fontsize=9, color="#666")

    # 过渡态标注
    ax.text(5.0, 5.2, "[Nu ··· C ··· LG]‡  过渡态（三角双锥）", ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#FB8C00"))
    # 曲线箭头1：亲核试剂 → C（成键）
    _curved_arrow(ax, (1.7, 3.5), (4.3, 3.5), rad=-0.45)
    ax.text(3.0, 4.5, "① 成键", ha="center", fontsize=9, color="#C62828")
    # 曲线箭头2：C-LG 键 → LG（断键）
    _curved_arrow(ax, (5.6, 3.5), (6.3, 4.6), rad=0.4)
    ax.text(6.6, 5.0, "② 断键", ha="left", fontsize=9, color="#C62828")

    ax.text(5.0, 1.2, "特征：背面进攻 · 一步协同 · 构型翻转（Walden 翻转）· 二级反应动力学",
            ha="center", fontsize=10, style="italic", color="#555")
    ax.set_title("SN2 机理（双分子亲核取代）", fontsize=13, fontweight="bold")
    return _to_png(fig)


def draw_e2(labels):
    """E2：碱夺取 β-H + 离去基离去 + 双键形成，一步协同。"""
    base = labels.get("base", "B:")
    sub = labels.get("substrate", "R-CH₂-CH₂-LG")
    lg = labels.get("leaving_group", "LG")
    prod = labels.get("product", "R-CH=CH₂")
    fig, ax = _new_axis()
    if fig is None:
        return None

    ax.text(1.0, 3.2, f"{base}", ha="center", fontsize=13, fontweight="bold", color="#1565C0")
    ax.text(1.0, 2.5, "碱", ha="center", fontsize=9, color="#666")
    ax.text(5.0, 3.2, f"{sub}", ha="center", fontsize=12, fontweight="bold")
    ax.text(5.0, 2.5, "底物（含 β-H）", ha="center", fontsize=9, color="#666")
    ax.text(9.0, 3.2, f"{prod}  +  :{lg}  +  H-B", ha="center", fontsize=12, fontweight="bold", color="#2E7D32")
    ax.text(9.0, 2.5, "烯烃 + 离去基 + 共轭酸", ha="center", fontsize=9, color="#666")

    ax.text(5.0, 5.2, "[B···H···C=C···LG]‡  过渡态", ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#FB8C00"))
    # 箭头1：碱 → β-H
    _curved_arrow(ax, (1.7, 3.5), (4.2, 3.7), rad=-0.4)
    ax.text(2.8, 4.6, "① 夺 H", ha="center", fontsize=9, color="#C62828")
    # 箭头2：C-H 键 → C=C（成 π 键）
    _curved_arrow(ax, (4.8, 3.9), (5.4, 4.6), rad=0.3)
    ax.text(5.0, 5.0, "② 成 π 键", ha="center", fontsize=9, color="#C62828")
    # 箭头3：C-LG → LG
    _curved_arrow(ax, (5.8, 3.5), (6.5, 4.5), rad=0.4)
    ax.text(6.8, 4.9, "③ 断键", ha="left", fontsize=9, color="#C62828")

    ax.text(5.0, 1.2, "特征：一步协同 · 反式共平面消除 · 遵循 Zaitsev 规则（优先生成取代多的烯烃）",
            ha="center", fontsize=10, style="italic", color="#555")
    ax.set_title("E2 机理（双分子消除）", fontsize=13, fontweight="bold")
    return _to_png(fig)


_DRAWERS = {"SN2": draw_sn2, "E2": draw_e2, "SN1": None, "E1": None}


def draw_mechanism(mech_type, labels=None):
    """按机理类型绘制示意图，返回 PNG bytes。

    mech_type: SN2/E2/SN1/E1（SN1/E1 暂未内置绘图，返回 None）
    labels: dict，各机理所需的角色标签（由 LLM 识别提供）
    """
    drawer = _DRAWERS.get(mech_type.upper())
    if drawer is None:
        return None
    return drawer(labels or {})


if __name__ == "__main__":
    # 离线检查：draw_mechanism 路由 + 类型识别（matplotlib 不可用时返回 None，逻辑仍可验证）
    print("支持的机理:", SUPPORTED)
    for t in ("SN2", "E2", "SN1", "E1", "XX"):
        png = draw_mechanism(t, {"nucleophile": "OH-", "substrate": "CH3Cl", "leaving_group": "Cl-", "product": "CH3OH"})
        print(f"  {t}: {'有drawer' if png else ('无drawer' if t.upper() in SUPPORTED else '不支持')}")
