# -*- coding: utf-8 -*-
"""renderers/collide.py — 几何冲突检测与占据注册（渲染器空间感知基础）。

所有可绘元素声明占据形状：标签=矩形、电子点/电荷圈/H 节点=圆、键=线段。
注解类元素（电荷圈/孤对电子/显式 H）给出按教科书偏好排序的候选位置，
放置时取第一个零冲突者；全部冲突时取穿透最小者并告警（不再静默）。

设计要点：
- 候选序列首位 = 既有规则输出——零冲突时渲染结果与旧行为逐像素一致；
- 纯几何、确定性、无外部依赖；同时供渲染期微调与测试期断言使用
  （Drawbacks 一-8：检查机制 + 坐标微调）。
"""

import math

# 形状一律为 ("rect", x0, y0, x1, y1) / ("circle", cx, cy, r)
# / ("seg", x1, y1, x2, y2)

CHARGE_CIRCLE_R = 0.10   # 圆圈电荷节点占据半径（scriptsize scale=0.5 实测）
DOT_R = 0.028            # 电子点半径（与 lone_pair_tikz 一致）
_PAD = 0.02              # 元素间最小间隙


def _seg_point_dist(x1, y1, x2, y2, px, py) -> float:
    """点到线段的最短距离。"""
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _penetration(a: tuple, b: tuple, pad: float = _PAD) -> float:
    """两形状的重合深度（0 = 无冲突；越大冲突越严重）。"""
    if a[0] == "rect" and b[0] == "rect":
        # 轴对齐矩形：x/y 两个方向均重叠才算冲突
        ox = min(a[3], b[3]) - max(a[1], b[1])
        oy = min(a[4], b[4]) - max(a[2], b[2])
        return min(ox, oy) + pad if ox > 0 and oy > 0 else 0.0
    if a[0] == "rect":
        a, b = b, a
    if a[0] == "circle" and b[0] == "rect":
        _, cx, cy, r = a
        _, x0, y0, x1, y1 = b
        nx = max(x0, min(cx, x1))
        ny = max(y0, min(cy, y1))
        d = math.hypot(cx - nx, cy - ny)
        return max(0.0, r + pad - d)
    if a[0] == "circle" and b[0] == "circle":
        d = math.hypot(a[1] - b[1], a[2] - b[2])
        return max(0.0, a[3] + b[3] + pad - d)
    if a[0] == "circle" and b[0] == "seg":
        d = _seg_point_dist(b[1], b[2], b[3], b[4], a[1], a[2])
        return max(0.0, a[3] + pad - d)
    if a[0] == "seg" and b[0] == "circle":
        return _penetration(b, a, pad)
    # seg-seg / 其他组合暂不需要（注解元素不含线段-线段冲突）
    return 0.0


class Occupancy:
    """占据注册表：登记已放置元素的占据形状，查询候选位置冲突。

    place(candidates) 取第一个零冲突候选（全部冲突时取穿透最小者并返回
    其穿透值，调用方可告警）；登记后可继续避让后续元素。
    """

    def __init__(self, pad: float = _PAD):
        self.shapes = []
        self.pad = pad

    def add(self, shape: tuple) -> None:
        self.shapes.append(shape)

    def add_rect(self, x0, y0, x1, y1) -> None:
        self.add(("rect", x0, y0, x1, y1))

    def add_circle(self, cx, cy, r) -> None:
        self.add(("circle", cx, cy, r))

    def add_segment(self, x1, y1, x2, y2) -> None:
        self.add(("seg", x1, y1, x2, y2))

    def penetration(self, shape: tuple) -> float:
        """形状与已登记元素的最大穿透深度（0 = 无冲突）。"""
        return max((_penetration(shape, s, self.pad) for s in self.shapes),
                   default=0.0)

    def place(self, candidates: list, warn: str = "") -> tuple:
        """候选位置（形状列表）取第一个零冲突者并登记；全部冲突时取
        穿透最小者、登记并打印告警。返回 (形状, 穿透值)。candidates 为空
        返回 (None, 0.0)。"""
        if not candidates:
            return None, 0.0
        best, best_pen = candidates[0], float("inf")
        for cand in candidates:
            pen = self.penetration(cand)
            if pen <= 1e-9:
                self.add(cand)
                return cand, 0.0
            if pen < best_pen:
                best, best_pen = cand, pen
        self.add(best)
        if warn:
            print(f"[collide] {warn}：所有候选位均有冲突，"
                  f"取穿透最小者（深度 {best_pen:.2f}）")
        return best, best_pen
