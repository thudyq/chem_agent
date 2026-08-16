# -*- coding: utf-8 -*-
"""renderers/chair.py（[CHAIR] 椅式构象）单元测试。

几何断言依据 Klein 五步构造法（instructions/cyclohexane.pdf）：
三对平行骨架键、axial 严格竖直交替、equatorial 平行浅斜键且指向环外。

运行: python -m pytest tests/test_chair.py -v
"""

import math
import re

import pytest

pytest.importorskip("rdkit", reason="rdkit 未安装，跳过椅式构象测试")

from core.tag_parser import parse_tags
from core.tag_validator import validate_tags
from renderers.chair import render_chair


def _validate(text):
    return validate_tags(parse_tags(text))


def _bond_angles(tikz: str) -> list:
    r"""提取骨架 \draw 线段的方向角（0~180，线无向）。"""
    angles = []
    for x1, y1, x2, y2 in re.findall(
            r"\\draw \(([-\d.]+),([-\d.]+)\) -- \(([-\d.]+),([-\d.]+)\);",
            tikz):
        ang = math.degrees(math.atan2(float(y2) - float(y1),
                                      float(x2) - float(x1)))
        angles.append(ang % 180.0)
    return angles


class TestParse:
    def test_parse_plain(self):
        tags = parse_tags("[CHAIR:C1CCCCC1]")
        assert tags[0].type == "CHAIR"
        assert tags[0].args == ["C1CCCCC1", ""]

    def test_parse_with_subs(self):
        tags = parse_tags("[CHAIR:BrC1CCCCC1,1:ax]")
        assert tags[0].args == ["BrC1CCCCC1", "1:ax"]


class TestValidate:
    def test_valid(self, fake_rdkit):
        _, invalid = _validate("[CHAIR:BrC1CCCCC1,1:ax]")
        assert len(invalid) == 0

    def test_no_ring_rejected(self):
        _, invalid = _validate("[CHAIR:CCO,1:ax]")
        assert len(invalid) == 1
        assert "六元环" in invalid[0].reason

    def test_bad_kind_rejected(self):
        _, invalid = _validate("[CHAIR:BrC1CCCCC1,1:xx]")
        assert len(invalid) == 1
        assert "格式错误" in invalid[0].reason

    def test_pos_out_of_range_rejected(self):
        _, invalid = _validate("[CHAIR:BrC1CCCCC1,7:ax]")
        assert len(invalid) == 1
        assert "超出范围" in invalid[0].reason

    def test_pos_without_substituent_rejected(self):
        _, invalid = _validate("[CHAIR:C1CCCCC1,3:ax]")
        assert len(invalid) == 1
        assert "无取代基" in invalid[0].reason


class TestRender:
    def test_plain_skeleton_six_bonds(self):
        out = render_chair("C1CCCCC1")
        assert out.startswith("\\begin{tikzpicture}")
        assert len(re.findall(r"\\draw \(", out)) == 6  # 6 条骨架键

    def test_substituent_label(self):
        out = render_chair("BrC1CCCCC1", "1:ax")
        assert "{Br}" in out

    def test_invalid_smiles(self):
        assert "渲染失败" in render_chair("XYZ")

    def test_not_cyclohexane(self):
        assert "渲染失败" in render_chair("CCO")


class TestGeometry:
    def test_three_parallel_pairs(self):
        """骨架 6 键构成三对平行线（Klein 自检规则）。"""
        angles = _bond_angles(render_chair("C1CCCCC1"))
        assert len(angles) == 6
        # 按角度聚类：应恰好 3 组，每组 2 条（容差 1°）
        groups = {}
        for a in angles:
            key = None
            for k in groups:
                d = abs(a - k) % 180.0
                if min(d, 180.0 - d) < 1.0:
                    key = k
                    break
            groups.setdefault(key if key is not None else a, []).append(a)
        sizes = sorted(len(v) for v in groups.values())
        assert sizes == [2, 2, 2], f"平行对数异常: {groups}"

    def test_axial_bonds_vertical(self):
        """axial 取代基键严格竖直（90°）。"""
        out = render_chair("BrC1CCCCC1", "1:ax")
        bonds = re.findall(
            r"\\draw \(([-\d.]+),([-\d.]+)\) -- \(([-\d.]+),([-\d.]+)\);", out)
        assert len(bonds) == 7          # 6 骨架 + 1 取代基
        sub = bonds[-1]
        dx = abs(float(sub[2]) - float(sub[0]))
        dy = abs(float(sub[3]) - float(sub[1]))
        assert dx < 0.01 and dy > 0.5, f"axial 键不竖直: {sub}"

    def test_equatorial_outward_and_parallel(self):
        """equatorial 键：与浅斜骨架键平行（±σ），水平分量指向环外。"""
        out = render_chair("CC1CCCCC1", "1:eq")
        bonds = re.findall(
            r"\\draw \(([-\d.]+),([-\d.]+)\) -- \(([-\d.]+),([-\d.]+)\);", out)
        sub = bonds[-1]
        x1, y1, x2, y2 = map(float, sub)
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360.0
        # 与 15° 线平行（15° 或 195° 或 165° 或 345°，容差 1°）
        line = ang % 180.0
        assert min(abs(line - 15.0), abs(line - 165.0)) < 1.0, \
            f"equatorial 键不平行浅斜骨架: {ang:.1f}°"