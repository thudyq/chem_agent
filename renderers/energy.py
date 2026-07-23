# -*- coding: utf-8 -*-
"""renderers/energy.py — [ENERGY] 标记渲染器：反应能量剖面图（势能面）。

输入逗号分隔的相对能量值（kJ/mol，反应物=0），输出纯 TikZ 折线图：
点 + 连线 + 驻点标签 + Ea/ΔH 标注框。无 pgfplots 依赖，与其他渲染器统一。
驻点坐标由布局引擎（renderers/layout.py::energy_point_coords）计算。
"""

from .layout import energy_annotation_placement, energy_point_coords


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
    info = energy_point_coords(values)
    max_idx = info["max_idx"]
    x_last = info["x_last"]

    # 驻点标签占用区域 → 标注框与纵轴高度（避免遮挡）
    occupied = []
    for i, v, x, y in info["points"]:
        yoff = 0.35 if i == max_idx else -0.3
        occupied.append((x - 0.85, y + yoff - 0.22, x + 0.85, y + yoff + 0.22))
    box_x, box_y, box_anchor, axis_top = energy_annotation_placement(
        occupied, x_last)

    lines = ["\\begin{tikzpicture}"]
    # 坐标轴
    lines.append(f"  \\draw[->] (0,0) -- ({x_last + 0.8:.1f},0);")
    lines.append(f"  \\draw[->] (0,0) -- (0,{axis_top:.1f});")
    lines.append(f"  \\node[font=\\small] at ({(x_last + 0.8) / 2:.1f},-0.30) {{反应进程}};")
    lines.append(
        f"  \\node[font=\\small, rotate=90, anchor=south] at (-0.10,{axis_top - 0.5:.1f}) "
        "{能量 (kJ/mol)};"
    )
    # 反应物基线
    y0 = info["points"][0][3]
    lines.append(f"  \\draw[gray, dashed] (0,{y0:.2f}) -- ({x_last:.1f},{y0:.2f});")
    # 平滑曲线
    coords = " ".join(f"({x:.1f},{y:.2f})" for _, _, x, y in info["points"])
    lines.append(f"  \\draw[thick, blue, smooth] plot coordinates {{{coords}}};")
    # 点 + 标签：每个驻点一个 scope（局部坐标，便于后续叠加结构组件）
    role_map = {0: "反应物", n - 1: "产物"}
    if max_idx not in role_map:
        role_map[max_idx] = "过渡态"
    for i, v, x, y in info["points"]:
        lines.append(f"  \\begin{{scope}}[shift={{({x:.1f},{y:.2f})}}]")
        lines.append("    \\fill[blue] (0,0) circle (0.06);")
        if i in role_map:
            yoff = 0.35 if i == max_idx else -0.3
            lines.append(f"    \\node[font=\\small] at (0,{yoff:.2f}) {{{role_map[i]} ({v:+.0f})}};")
        lines.append("  \\end{scope}")
    # Ea / ΔH 标注框（净空带，anchor 定位）
    ea = max(values) - values[0]
    dh = values[-1] - values[0]
    node_text = f"Ea $\\approx$ {ea:.0f} kJ/mol\\\\$\\Delta$H $\\approx$ {dh:+.0f} kJ/mol"
    lines.append(
        "  \\node[draw, rounded corners, fill=yellow!10, font=\\small, align=left, "
        f"anchor={box_anchor}] at ({box_x:.2f},{box_y:.2f}) {{{node_text}}};"
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
