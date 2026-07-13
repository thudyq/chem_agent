# -*- coding: utf-8 -*-
"""renderers/energy.py — [ENERGY] 标记渲染器：反应能量剖面图（势能面）。

输入逗号分隔的相对能量值（kJ/mol，反应物=0），输出纯 TikZ 折线图：
点 + 连线 + 驻点标签 + Ea/ΔH 标注框。无 pgfplots 依赖，与其他渲染器统一。
"""


def render_energy(points_str: str) -> str:
    """[ENERGY] 渲染：能量点序列 → TikZ 势能面图。失败返回错误提示。"""
    if not points_str or not isinstance(points_str, str):
        return "（势能面渲染失败：点序列为空）"
    try:
        values = [float(v.strip()) for v in points_str.split(",") if v.strip()]
    except ValueError:
        return f"（势能面渲染失败：点序列格式错误「{points_str}」）"
    if len(values) < 2:
        return "（势能面渲染失败：至少需要 2 个能量点）"

    n = len(values)
    emax, emin = max(values), min(values)
    erange = (emax - emin) or 1.0
    max_idx = values.index(emax)

    def yp(e):
        return (e - emin) / erange * 3.0

    xstep = 1.5
    x0 = 1.0  # 数据起点右移，给纵轴留空间
    x_last = x0 + (n - 1) * xstep

    lines = ["\\begin{tikzpicture}"]
    # 坐标轴
    lines.append(f"  \\draw[->] (0,0) -- ({x_last + 0.8:.1f},0);")
    lines.append(f"  \\draw[->] (0,0) -- (0,4.0);")
    lines.append(f"  \\node[font=\\small] at ({(x_last + 0.8) / 2:.1f},-0.30) {{反应进程}};")
    lines.append("  \\node[font=\\small, rotate=90, anchor=south] at (-0.10,3.5) {能量 (kJ/mol)};")
    # 反应物基线
    y0 = yp(values[0])
    lines.append(f"  \\draw[gray, dashed] (0,{y0:.2f}) -- ({x_last:.1f},{y0:.2f});")
    # 平滑曲线
    coords = " ".join(f"({x0 + i*xstep:.1f},{yp(v):.2f})" for i, v in enumerate(values))
    lines.append(f"  \\draw[thick, blue, smooth] plot coordinates {{{coords}}};")
    # 点 + 标签
    role_map = {0: "反应物", n - 1: "产物"}
    if max_idx not in role_map:
        role_map[max_idx] = "过渡态"
    for i, v in enumerate(values):
        x = x0 + i * xstep
        y = yp(v)
        lines.append(f"  \\fill[blue] ({x:.1f},{y:.2f}) circle (0.06);")
        if i in role_map:
            yoff = 0.35 if i == max_idx else -0.3
            lines.append(f"  \\node[font=\\small] at ({x:.1f},{y+yoff:.2f}) {{{role_map[i]} ({v:+.0f})}};")
    # Ea / ΔH 标注框（右上角，两行）
    ea = emax - values[0]
    dh = values[-1] - values[0]
    lines.append(
        f"  \\node[draw, rounded corners, fill=yellow!10, font=\\small, align=left]"
        f" at ({x_last:.1f},4.2) {{Ea $\\approx$ {ea:.0f} kJ/mol\\\\$\\Delta$H $\\approx$ {dh:+.0f} kJ/mol}};"
    )
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] SN2 (3点):")
    print(render_energy("0,108,-20"))
    print("\n[2] 两步反应 (5点):")
    print(render_energy("0,80,-10,60,-30"))
    print("\n[3] 无效:")
    print(render_energy("abc"))
