# -*- coding: utf-8 -*-
"""renderers/collide.py 单元测试（几何求交 + Occupancy 候选放置）。

运行: python -m pytest tests/test_collide.py -v
"""

import re

import pytest

from renderers.collide import Occupancy, _penetration


class TestPrimitives:
    def test_rect_rect_overlap(self):
        a = ("rect", 0, 0, 1, 1)
        assert _penetration(a, ("rect", 0.5, 0.5, 1.5, 1.5), 0.0) > 0
        assert _penetration(a, ("rect", 2, 0, 3, 1), 0.0) == 0.0

    def test_circle_circle(self):
        a = ("circle", 0, 0, 0.1)
        assert _penetration(a, ("circle", 0.15, 0, 0.1), 0.0) > 0
        assert _penetration(a, ("circle", 0.5, 0, 0.1), 0.0) == 0.0

    def test_circle_rect(self):
        r = ("rect", 0, 0, 1, 1)
        assert _penetration(("circle", 0.5, 0.5, 0.1), r, 0.0) > 0
        assert _penetration(("circle", 2.0, 0.5, 0.1), r, 0.0) == 0.0

    def test_circle_segment(self):
        seg = ("seg", 0, 0, 1, 0)
        assert _penetration(("circle", 0.5, 0.05, 0.1), seg, 0.0) > 0
        assert _penetration(("circle", 0.5, 0.5, 0.1), seg, 0.0) == 0.0
        # 端点外：最近点是端点
        assert _penetration(("circle", 1.05, 0.0, 0.1), seg, 0.0) > 0


class TestOccupancy:
    def test_first_free_candidate_wins(self):
        occ = Occupancy(pad=0.0)
        occ.add_circle(0, 0, 0.1)
        cands = [("circle", 0, 0, 0.1), ("circle", 1, 0, 0.1)]
        chosen, pen = occ.place(cands)
        assert chosen == cands[1] and pen == 0.0

    def test_all_conflict_picks_min_and_warns(self, capsys):
        occ = Occupancy(pad=0.0)
        occ.add_circle(0, 0, 0.1)
        cands = [("circle", 0, 0, 0.1), ("circle", 0.15, 0, 0.1)]
        chosen, pen = occ.place(cands, warn="测试")
        assert chosen == cands[1] and pen > 0
        assert "冲突" in capsys.readouterr().out

    def test_registered_shape_blocks_later_candidates(self):
        occ = Occupancy(pad=0.0)
        occ.place([("circle", 0, 0, 0.1)])
        assert occ.penetration(("circle", 0.05, 0, 0.1)) > 0
        assert occ.penetration(("circle", 1, 0, 0.1)) == 0.0

    def test_empty_candidates(self):
        occ = Occupancy()
        assert occ.place([]) == (None, 0.0)


class TestRenderedChargeClearance:
    """TikZ 回放检查（R-8 检查机制固化）：带电物种电池组中，圆圈电荷
    与任何键线段的距离必须大于圈半径（Drawbacks 一-8 类重叠回归）。"""

    CASES = ["O=[N+]([O-])c1ccccc1", "[OH-]", "[Cl-]", "CC(=O)[O-]",
             "C[C@H]([NH3+])C(=O)[O-]", "[O-]S(=O)(=O)[O-]", "[NH4+]"]

    def test_charge_circles_clear_of_bonds(self):
        pytest.importorskip("rdkit")
        from core.tag_parser import parse_tags
        from renderers.collide import CHARGE_CIRCLE_R, _seg_point_dist
        from renderers.composite import render_composite

        for smi in self.CASES:
            text = f"[COMPOSITE:row][STRUCT:{smi}][/COMPOSITE]"
            tag = next(t for t in parse_tags(text) if t.type == "COMPOSITE")
            out = render_composite(tag.args[0], tag.args[1])
            segs, circles = [], []
            for m in re.finditer(
                    r"\\begin\{scope\}\[shift=\{\(([-\d.]+),([-\d.]+)\)\}\]"
                    r"(.*?)\\end\{scope\}", out, re.DOTALL):
                sx, sy, body = float(m.group(1)), float(m.group(2)), m.group(3)
                for x1, y1, x2, y2 in re.findall(
                        r"\\draw \(([-\d.]+),([-\d.]+)\) -- "
                        r"\(([-\d.]+),([-\d.]+)\);", body):
                    segs.append((sx + float(x1), sy + float(y1),
                                 sx + float(x2), sy + float(y2)))
                for cx, cy in re.findall(
                        r"\\node\[draw, circle[^\]]*\] at "
                        r"\(([-\d.]+),([-\d.]+)\)", body):
                    circles.append((sx + float(cx), sy + float(cy)))
            for cx, cy in circles:
                d = min((_seg_point_dist(x1, y1, x2, y2, cx, cy)
                         for x1, y1, x2, y2 in segs), default=99.0)
                assert d > CHARGE_CIRCLE_R - 0.01, \
                    f"{smi}：电荷圈 ({cx:.2f},{cy:.2f}) 距键线 {d:.3f} 过近"
