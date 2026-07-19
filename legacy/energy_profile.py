# -*- coding: utf-8 -*-
"""
legacy/energy_profile.py
      ========================
反应能量剖面图（势能面）：解析驻点能量 -> matplotlib 绘图。

2.2 任务：
    - parse_energy_points: 从 LLM 文本解析驻点能量 JSON。
    - plot_energy_profile: 用 matplotlib 绘制反应坐标-能量剖面图，返回 PNG bytes。

设计：本模块只管"数据解析 + 绘图"，不导入 chem_agent（避免循环导入）；
LLM 调用由 chem_agent 完成，把文本传给 parse_energy_points。
matplotlib 懒导入，未安装时绘图优雅返回 None。
"""

import json
import re
import sys
from io import BytesIO
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_energy_points(text):
    """从 LLM 输出解析能量驻点 JSON。

    预期格式：{"points":[{"label":"反应物","energy":0}, ...]}
    兼容 markdown 代码围栏、<think> 思考链、reasoning 文本中夹带的 JSON。

    返回:
        list[dict] | None: [{"label":str, "energy":float}, ...]（至少 2 点）；解析失败返回 None。
    """
    if not text:
        return None
    # 剥 <think> 思考链、去 markdown 代码围栏
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()
    # 定位首个 {...} 块（容错：LLM 可能把 JSON 夹在散文里）
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    points = data.get("points") if isinstance(data, dict) else None
    if not isinstance(points, list) or len(points) < 2:
        return None
    result = []
    for p in points:
        if not isinstance(p, dict):
            continue
        label = p.get("label") or p.get("name") or ""
        try:
            energy = float(p.get("energy"))
        except (TypeError, ValueError):
            continue
        if label:
            result.append({"label": str(label), "energy": energy})
    return result if len(result) >= 2 else None


def plot_energy_profile(points, title="反应能量剖面图"):
    """用 matplotlib 绘制能量剖面图，返回 PNG bytes。

    points: [{"label":str, "energy":float}, ...]
    matplotlib 不可用或点数不足时返回 None。
    """
    if not points or len(points) < 2:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无显示环境的服务器端渲染
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    # CJK 字体回退（Windows: Microsoft YaHei/SimHei；macOS: Arial Unicode MS）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(points)
    x = list(range(n))
    energies = [p["energy"] for p in points]
    labels = [p["label"] for p in points]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, energies, "-o", color="#1976D2", markersize=9, linewidth=2.2, zorder=3)
    # 反应物基线参考
    ax.axhline(y=energies[0], color="gray", linestyle="--", alpha=0.4, linewidth=1)

    e_max = max(energies)
    for xi, e, lab in zip(x, energies, labels):
        yoff = 12 if e == e_max else -22
        ax.annotate(
            f"{lab}\n{e:+.1f} kJ/mol",
            (xi, e), textcoords="offset points", xytext=(0, yoff),
            ha="center", fontsize=9, color="#333",
        )

    # Ea（≈ 最高峰 - 反应物）与 ΔH（产物 - 反应物），均为估算
    ea = e_max - energies[0]
    dh = energies[-1] - energies[0]
    info = f"Ea ≈ {ea:.1f} kJ/mol\nΔH ≈ {dh:+.1f} kJ/mol"
    ax.text(
        0.98, 0.98, info, transform=ax.transAxes, ha="right", va="top",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="#FFFDE7", ec="#FBC02D", alpha=0.9),
    )

    ax.set_xlabel("反应坐标", fontsize=11)
    ax.set_ylabel("相对能量 (kJ/mol)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.margins(y=0.2)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


if __name__ == "__main__":
    # parse_energy_points 离线测试（无需 matplotlib）
    print("=== parse_energy_points 测试 ===")
    cases = [
        ('{"points":[{"label":"反应物","energy":0},{"label":"过渡态","energy":108},{"label":"产物","energy":-20}]}', 3),
        ('```json\n{"points":[{"label":"R","energy":0},{"label":"TS","energy":80},{"label":"P","energy":-15}]}\n```', 3),
        ('<think>分析中...</think>\n{"points":[{"label":"反应物","energy":0},{"label":"产物","energy":-30}]}', 2),
        ("这不是JSON，无法解析", None),
        ("", None),
    ]
    for text, expect_n in cases:
        r = parse_energy_points(text)
        n = len(r) if r else 0
        flag = "✓" if (expect_n is None and r is None) or (expect_n and n == expect_n) else "✗"
        print(f"  {flag} 期望{expect_n}点 -> 实际{n}点: {r}")
