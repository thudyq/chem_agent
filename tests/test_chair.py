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
from renderers.chair import render_chair, _chair_vertices


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

    def test_substituent_bond_length(self):
        """取代基键线终点距环碳 _SUB_LEN=1.1（20260818 微调）。

        Br 1 位 ax：键线终点 (0,1.10)，标签中心 (0,1.40)
        （再外移 label_bond_margin(Br)=0.30）。
        """
        from renderers.chair import _SUB_LEN
        out = render_chair("BrC1CCCCC1", "1:ax")
        bonds = re.findall(
            r"\\draw \(([-\d.]+),([-\d.]+)\) -- \(([-\d.]+),([-\d.]+)\);", out)
        sub = bonds[-1]
        x1, y1, x2, y2 = map(float, sub)
        length = math.hypot(x2 - x1, y2 - y1)
        assert abs(length - _SUB_LEN) < 1e-6, \
            f"取代基键长 {length:.3f} ≠ {_SUB_LEN}"
        assert abs(y2 - y1 - _SUB_LEN) < 1e-6, f"1 位 ax 键应竖直朝上: {sub}"
        assert "{Br}" in out

    def _sub_angle(self, smi, spec):
        """渲染后最后一条键（取代基键）的方向角（0~360）。"""
        out = render_chair(smi, spec)
        bonds = re.findall(
            r"\\draw \(([-\d.]+),([-\d.]+)\) -- \(([-\d.]+),([-\d.]+)\);", out)
        x1, y1, x2, y2 = map(float, bonds[-1])
        return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360.0

    def test_equatorial_pos2_and_pos5_60deg(self):
        """2/5 号位平伏键与水平呈 60°（fast_latex_test.tex 参考，20260818）。

        原实现与浅斜骨架键平行（±15°）→ 2 号 165°、5 号 345°（视觉太平）；
        正确：2 号 120°（左上）、5 号 300°（右下）。
        """
        assert abs(self._sub_angle("C1C(C)CCCC1", "2:eq") - 120.0) < 1.0
        assert abs(self._sub_angle("C1CCCC(C)C1", "5:eq") - 300.0) < 1.0

    def test_equatorial_pos2_pos5_flip_y_reflection(self):
        """flip 画法 2/5 号位平伏键 = 正常画法的 y 反射（240°/60°）。"""
        assert abs(self._sub_angle("C1C(C)CCCC1", "flip,2:eq") - 240.0) < 1.0
        assert abs(self._sub_angle("C1CCCC(C)C1", "flip,5:eq") - 60.0) < 1.0


class TestFlip:
    def test_parse_flip(self):
        tags = parse_tags("[CHAIR:C1CCCCC1,flip]")
        assert tags[0].args == ["C1CCCCC1", "flip"]

    def test_flip_token_position_free(self):
        tags = parse_tags("[CHAIR:CC1CCCCC1,1:eq,flip]")
        assert tags[0].args == ["CC1CCCCC1", "1:eq,flip"]

    def test_flip_valid(self, fake_rdkit):
        _, invalid = _validate("[CHAIR:BrC1CCCCC1,flip,1:ax]")
        assert len(invalid) == 0

    def test_flip_still_rejects_bad_kind(self):
        _, invalid = _validate("[CHAIR:BrC1CCCCC1,flip,1:xx]")
        assert len(invalid) == 1
        assert "格式错误" in invalid[0].reason

    def test_mirror_is_y_reflection(self):
        """flip 骨架 = 正常骨架关于水平轴反射（x 不变、y 取反），且闭环。"""
        normal = _chair_vertices(False)
        flipped = _chair_vertices(True)
        assert len(normal) == len(flipped) == 6
        for (x1, y1), (x2, y2) in zip(normal, flipped):
            assert abs(x1 - x2) < 1e-9, (x1, x2)
            assert abs(y1 + y2) < 1e-9, (y1, y2)
        # 镜像画法第 6 边（闭环）为 +60° 线（240°）
        ang = math.degrees(math.atan2(flipped[0][1] - flipped[-1][1],
                                      flipped[0][0] - flipped[-1][0])) % 360.0
        assert abs(ang - 240.0) < 1e-9, f"镜像闭环边方向异常: {ang:.2f}°"

    def test_flip_three_parallel_pairs(self):
        """镜像骨架 6 键仍落在 15°/60°/165° 三条线族上（三对平行）。"""
        angles = _bond_angles(render_chair("C1CCCCC1", "flip"))
        assert len(angles) == 6
        for a in angles:
            assert any(abs(a - t) < 1.0 for t in (15.0, 60.0, 165.0)), \
                f"镜像骨架意外角度: {a:.1f}°"

    def test_flip_axial_down_at_pos1(self):
        """镜像画法 1 位竖直键朝下（正常画法朝上，交替翻转）。"""
        out = render_chair("BrC1CCCCC1", "flip,1:ax")
        bonds = re.findall(
            r"\\draw \(([-\d.]+),([-\d.]+)\) -- \(([-\d.]+),([-\d.]+)\);", out)
        sub = bonds[-1]
        dx = abs(float(sub[2]) - float(sub[0]))
        dy = float(sub[3]) - float(sub[1])
        assert dx < 0.01 and dy < -0.5, f"flip axial 键不朝下: {sub}"

    def test_flip_equatorial_outward_and_parallel(self):
        """镜像画法 eq 键仍外指（水平分量离环心）且与浅斜骨架平行。"""
        out = render_chair("CC1CCCCC1", "flip,1:eq")
        bonds = re.findall(
            r"\\draw \(([-\d.]+),([-\d.]+)\) -- \(([-\d.]+),([-\d.]+)\);", out)
        sub = bonds[-1]
        x1, y1, x2, y2 = map(float, sub)
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360.0
        line = ang % 180.0
        assert min(abs(line - 15.0), abs(line - 165.0)) < 1.0, \
            f"flip equatorial 不平行浅斜骨架: {ang:.1f}°"
        assert x2 < x1, f"flip equatorial 未指向环外: {sub}"