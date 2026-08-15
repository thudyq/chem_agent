# -*- coding: utf-8 -*-
"""core/tag_validator.py 标记契约校验层单元测试（P1）。

用假 rdkit（conftest.py 的 fake_rdkit fixture）解耦真实 RDKit，
保证测试在任何环境行为一致。
"""

import pytest

import core.tag_validator as tv
from app import process_question
from core.tag_parser import parse_tags


def _validate(text):
    return tv.validate_tags(parse_tags(text))


def test_valid_smiles_pass(fake_rdkit):
    valid, invalid = _validate("[STRUCT:c1ccccc1] [ENERGY:0,108,-20]")
    assert len(invalid) == 0
    assert len(valid) == 2


def test_invalid_smiles_rejected(fake_rdkit):
    _, invalid = _validate("[STRUCT:XYZABC]")
    assert len(invalid) == 1
    assert "无效 SMILES" in invalid[0].reason


def test_h_prefix_smiles_normalized():
    """[H3O+] 化学式习惯写法：规范化后通过校验（20260815）。

    [H3O+] 的 H3 前缀在 SMILES 语法中非法（RDKit 解析失败），经
    normalize_h_prefix_smiles 重写为 [OH3+] 后放行；[H+]/[2H]/
    [OH3+] 等合法写法原样不受影响。需真实 RDKit（不用 fake_rdkit）。"""
    pytest.importorskip("rdkit")
    from utils.rdkit_utils import normalize_h_prefix_smiles
    assert normalize_h_prefix_smiles("[H3O+]") == "[OH3+]"
    assert normalize_h_prefix_smiles("[H2O]") == "[OH2]"
    assert normalize_h_prefix_smiles("[H3N+]") == "[NH3+]"
    assert normalize_h_prefix_smiles("[H+]") == "[H+]"     # 质子原样
    assert normalize_h_prefix_smiles("[2H]") == "[2H]"     # 同位素原样
    assert normalize_h_prefix_smiles("[OH3+]") == "[OH3+]"
    assert normalize_h_prefix_smiles("[NH4+]") == "[NH4+]"
    _, invalid = _validate("[STRUCT:[H3O+],label=H3O+]")
    assert len(invalid) == 0
    _, invalid = _validate("[STRUCT:[H3O+],label=OH3+]")
    assert len(invalid) == 0


def test_group_abbrev_smiles_allowed():
    """通用基团缩写（R/X/Ph/Ac 等）作为 SMILES 原子放行（20260815）。

    R/X/Ph/Ac 不是合法 SMILES 元素，经 expand_group_abbrevs 替换为
    dummy 原子（[*:n]）后校验通过；化学式后缀（-OH/COOH 等）一并
    规范化；真实元素（Cl/Br/CCO）与方括号原子（[OH-]）不受影响。
    需真实 RDKit。"""
    pytest.importorskip("rdkit")
    from utils.rdkit_utils import expand_group_abbrevs
    out, m = expand_group_abbrevs("R-Br")
    assert out == "[*:1]-Br" and m == {1: "R"}
    out, m = expand_group_abbrevs("Ph-OH")
    assert out == "[*:1]-O" and m == {1: "Ph"}
    out, m = expand_group_abbrevs("AcOH")
    assert out == "[*:1]O" and m == {1: "Ac"}
    out, m = expand_group_abbrevs("BuLi")
    assert out == "[*:1][Li]" and m == {1: "Bu"}      # 金属加方括号
    out, m = expand_group_abbrevs("RC(=O)OEt")
    assert out == "[*:1]C(=O)O[*:2]" and m == {1: "R", 2: "Et"}  # 多占位
    out, m = expand_group_abbrevs("RC(=O)O[Et]")
    assert out == "[*:2]C(=O)O[*:1]" and m == {1: "Et", 2: "R"}  # 方括号无残留
    out, m = expand_group_abbrevs("R1-Br")
    assert out == "[*:1]-Br" and m == {1: "R1"}      # R/X 编号
    out, m = expand_group_abbrevs("R'-Br")
    assert out == "[*:1]-Br" and m == {1: "R'"}      # R 撇号
    assert expand_group_abbrevs("[OH-]")[0] == "[OH-]"  # 方括号原子不误伤
    assert expand_group_abbrevs("CCO")[1] == {}
    assert expand_group_abbrevs("Cl")[1] == {}          # Cl 的 C 不误伤
    for text in ("[STRUCT:R-Br]", "[STRUCT:Ph-OH]",
                 "[STRUCT:AcOH]", "[STRUCT:MeOH]",
                 "[STRUCT:EtBr,label=溴乙烷]", "[STRUCT:BuLi]",
                 "[STRUCT:PhCOOH]", "[STRUCT:RC(=O)OEt]",
                 "[STRUCT:RC(=O)O[Et]]", "[STRUCT:OEt]",
                 "[STRUCT:R1-Br]", "[STRUCT:R2COOH]",
                 "[STRUCT:R'-Br]", "[STRUCT:X2]"):
        _, invalid = _validate(text)
        assert len(invalid) == 0, f"{text} → {[r.reason for r in invalid]}"


def test_reaction_with_group_abbrev_balances():
    """REACTION 含 R 的物种：dummy 占位不计入元素守恒（未知组成），
    骨架元素与净电荷照常比对（20260815）。"""
    pytest.importorskip("rdkit")
    _, invalid = _validate("[REACTION:R-Br;[OH-]|R-OH;[Br-]|]")
    assert len(invalid) == 0
    # 骨架不守恒仍拦截（Br 消失、O 凭空出现）
    _, invalid = _validate("[REACTION:R-Br|[OH-]|R-O|]")
    assert len(invalid) == 1
    assert "不守恒" in invalid[0].reason


def test_empty_smiles_rejected():
    _, invalid = _validate("[STRUCT:]")
    assert len(invalid) == 1
    assert "SMILES 为空" in invalid[0].reason


def test_long_label_rejected():
    _, invalid = _validate("[STRUCT:CO,label=" + "长" * 25 + "]")
    assert len(invalid) == 1
    assert "label 过长" in invalid[0].reason


def test_newman_bad_angle_rejected():
    _, invalid = _validate("[NEWMAN:CC,xyz]")
    assert len(invalid) == 1
    assert "角度" in invalid[0].reason


def test_newman_angle_out_of_range():
    _, invalid = _validate("[NEWMAN:CC,450]")
    assert len(invalid) == 1


def test_energy_bad_value_rejected():
    _, invalid = _validate("[ENERGY:0,abc]")
    assert len(invalid) == 1
    assert "不是数字" in invalid[0].reason


def test_energy_too_few_points_rejected():
    _, invalid = _validate("[ENERGY:5]")
    assert len(invalid) == 1
    assert "至少需要 2" in invalid[0].reason


def test_composite_unknown_layout_rejected():
    _, invalid = _validate("[COMPOSITE:foo][STRUCT:CCl][/COMPOSITE]")
    assert len(invalid) == 1


def test_composite_no_struct_rejected():
    _, invalid = _validate("[COMPOSITE:row][PLUS][/COMPOSITE]")
    assert len(invalid) == 1


def test_composite_energy_at_out_of_range(fake_rdkit):
    text = ("[COMPOSITE:energy][ENERGY:0,108,-20]"
            "[STRUCT:CCl,at=0][STRUCT:CO,at=9][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "at=" in invalid[0].reason


def test_composite_mecharrow_unknown_id(fake_rdkit):
    text = ("[COMPOSITE:reaction_mech][STRUCT:CCl,id=r0][RXNARROW]"
            "[STRUCT:CO,id=p0][MECHARROW:ghost:0>r0:0][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "未知组件" in invalid[0].reason


def test_composite_mecharrow_atom_out_of_range(fake_rdkit):
    text = ("[COMPOSITE:reaction_mech][STRUCT:CCl,id=r0][RXNARROW]"
            "[STRUCT:CO,id=p0][MECHARROW:r0:5>p0:0][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "超出范围" in invalid[0].reason


def test_composite_mecharrow_nonexistent_bond_rejected():
    """MECHARROW 的 a-b 端点必须真实成键——引用不存在的键（乙醛 0-2）拦截。

    回归锚点：LLM 在羟醛缩合中写 `ald:0-2`（乙醛 CC=O 的 0 与 2 无键），
    旧校验只查序号范围不查键存在性，漏洞放行导致渲染到错误位置。
    """
    pytest.importorskip("rdkit")
    text = ("[COMPOSITE:reaction_mech][STRUCT:CC=O,id=ald][RXNARROW]"
            "[STRUCT:CCO,id=p0][MECHARROW:ald:0>ald:0-2][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "没有成键" in invalid[0].reason


def test_bond_ref_gives_neighbor_hint():
    """键端点引用无键时，原因含原子实际连接——可操作化（帮助重数索引）。"""
    pytest.importorskip("rdkit")
    text = ("[COMPOSITE:reaction_mech]"
            "[STRUCT:O=S([O-])(=O)C1C=CC=C[CH+]1,id=sigma]"
            "[MECHARROW:sigma:3-8>sigma:8][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "没有成键" in invalid[0].reason
    assert "实际连接" in invalid[0].reason


def test_composite_mecharrow_existing_bond_passes():
    """a-b 端点引用真实存在的键（乙醛 0-1）放行。"""
    pytest.importorskip("rdkit")
    text = ("[COMPOSITE:reaction_mech][STRUCT:CC=O,id=ald][RXNARROW]"
            "[STRUCT:CCO,id=p0][MECHARROW:ald:0>ald:0-1][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 0


class TestChemicalChecks:
    """化学校验（T2-3 原子守恒）：需要真实 RDKit，不使用 fake_rdkit
    （元素计数依赖真实 Mol）。

    注：T2-2 label 与 SMILES 化学式一致性校验已于 2026-08-14 按用户裁定
    删除（误报多于收益，如 H3O+ 配 [OH2+] 属可容忍表述差异）。
    """

    def test_label_formula_consistent_passes(self):
        pytest.importorskip("rdkit")
        _, invalid = _validate("[STRUCT:CCl,label=CH3Cl]")
        assert len(invalid) == 0

    def test_label_formula_mismatch_allowed(self):
        """label 与 SMILES 化学式不一致 → 放行（T2-2 已删除）。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[STRUCT:CCl,label=CH4Cl]")
        assert len(invalid) == 0

    def test_label_non_formula_skipped(self):
        pytest.importorskip("rdkit")
        _, invalid = _validate("[STRUCT:CCl,label=氯甲烷]")
        assert len(invalid) == 0

    def test_label_placeholder_letter_skipped(self):
        """占位字母（非真实元素）不视为化学式，跳过校验。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[STRUCT:CCl,label=A]")
        assert len(invalid) == 0

    def test_label_generic_group_skipped(self):
        """通用基团缩写（Ar=芳基等，prompt 允许）不按化学式校验。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[STRUCT:CC(=O)c1ccccc1,label=Ar]")
        assert len(invalid) == 0
        _, invalid = _validate("[STRUCT:CC(=O)c1ccccc1,label=Ph]")
        assert len(invalid) == 0

    def test_label_charge_formula_passes(self):
        pytest.importorskip("rdkit")
        _, invalid = _validate("[STRUCT:[OH-],label=OH-]")
        assert len(invalid) == 0

    def test_label_nitronium_passes(self):
        """NO2+：尾数字归元素（N1O2 带 +1）——回归：曾误解析为 NO + 2 价。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[STRUCT:[N+](=O)=O,label=NO2+]")
        assert len(invalid) == 0

    def test_label_charge_digit_belongs_to_charge(self):
        """Ca2+ / SO42-：尾数字归电荷（Ca 带 +2、SO4 带 -2）。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[STRUCT:[Ca+2],label=Ca2+]")
        assert len(invalid) == 0
        _, invalid = _validate("[STRUCT:[O-]S(=O)(=O)[O-],label=SO42-]")
        assert len(invalid) == 0

    def test_reaction_balanced_passes(self):
        pytest.importorskip("rdkit")
        _, invalid = _validate("[REACTION:C=C;O|CCO|H2SO4]")
        assert len(invalid) == 0

    def test_reaction_unbalanced_rejected(self):
        pytest.importorskip("rdkit")
        _, invalid = _validate("[REACTION:C=C|CCO|H2SO4]")
        assert len(invalid) == 1
        assert "化学校验" in invalid[0].reason
        assert "不守恒" in invalid[0].reason

    def test_ethanol_oxidation_unbalanced_rejected(self):
        """用户报告（20260811）：乙醇氧化成乙醛只写骨架变化（CCO|CC=O）
        漏氧化剂/脱氢产物——校验拦截。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[REACTION:CCO|CC=O|Cu, Δ]")
        assert len(invalid) == 1
        assert "化学校验" in invalid[0].reason
        assert "C2H6O" in invalid[0].reason and "C2H4O" in invalid[0].reason

    def test_ethanol_oxidation_balanced_passes(self):
        """配平的乙醇氧化（催化氧化 + 脱氢两写法）均放行。"""
        pytest.importorskip("rdkit")
        for text in (
            "[REACTION:CCO;CCO;O=O|CC=O;CC=O;O;O|Cu, Δ]",  # 2EtOH+O2→2CH3CHO+2H2O
            "[REACTION:CCO|CC=O;[H][H]|Cu, Δ]",            # EtOH→CH3CHO+H2
        ):
            _, invalid = _validate(text)
            assert len(invalid) == 0, f"配平写法被误拦: {text}"

    def test_reaction_second_attempt_still_unbalanced_rejected(self):
        """用户报告二次修正：CCO;O|CC=O;O 两侧都加水仍不守恒——继续拦截。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[REACTION:CCO;O|CC=O;O|Cu, Δ]")
        assert len(invalid) == 1
        assert "C2H8O2" in invalid[0].reason and "C2H6O2" in invalid[0].reason

    def test_degrade_text_chem_clean_for_user(self):
        """用户可见降级消息：去「化学校验：」前缀与括号详情/修正指导，
        只留主因；完整原因仍保留在 reason 中供 P2 修正与 metrics 使用。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[REACTION:CCO|CC=O|Cu, Δ]")
        tag, reason = invalid[0].tag, invalid[0].reason
        shown = tv.degrade_text(tag, reason)
        assert shown == "（反应方程式图示无法渲染：方程式两侧原子不守恒，已省略）"
        assert "化学校验：" not in shown
        assert "辅助试剂" not in shown          # 修正指导不再面向用户
        assert "C2H6O" in reason                # 完整原因仍保留

    def test_degrade_text_non_chem_unchanged(self):
        """非化学校验原因（无效 SMILES）降级文本保持原样。"""
        _, invalid = _validate("[STRUCT:XYZABC]")
        tag, reason = invalid[0].tag, invalid[0].reason
        shown = tv.degrade_text(tag, reason)
        assert "无效 SMILES" in shown
        assert shown.startswith("（结构式图示无法渲染：")

    def test_degrade_text_friendly_omits_details(self):
        """用户可见友好降级：不含校验技术细节（无效 SMILES/守恒等）。

        前后端分开：注入回答的降级文本只告知"图示未生成"，技术原因仍在
        reason（P2 修正 / diagnostics / metrics）里。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[STRUCT:XYZABC]")
        shown = tv.degrade_text_friendly(invalid[0].tag)
        assert shown == "（结构式图示无法渲染，已省略）"
        assert "无效 SMILES" not in shown
        _, invalid2 = _validate("[REACTION:CCO|CC=O|Cu, Δ]")
        shown2 = tv.degrade_text_friendly(invalid2[0].tag)
        assert shown2 == "（反应方程式图示无法渲染，已省略）"
        assert "不守恒" not in shown2
        assert "化学校验" not in shown2

    def test_reaction_protonation_balanced_passes(self):
        pytest.importorskip("rdkit")
        _, invalid = _validate("[REACTION:CCO;[H+]|CC[OH2+]]")
        assert len(invalid) == 0

    def test_composite_mech_balanced_passes(self):
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[COMPOSITE:reaction_mech][STRUCT:CCl][PLUS][STRUCT:[OH-]]"
            "[RXNARROW][STRUCT:CO][PLUS][STRUCT:[Cl-]][/COMPOSITE]")
        assert len(invalid) == 0

    def test_composite_mech_unbalanced_rejected(self):
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[COMPOSITE:reaction_mech][STRUCT:CCO][PLUS][STRUCT:C[OH2+]]"
            "[RXNARROW][STRUCT:CCOCC][PLUS][STRUCT:O][/COMPOSITE]")
        assert len(invalid) == 1
        assert "化学校验" in invalid[0].reason

    def test_composite_mech_deprotonation_tolerated(self):
        """EAS 去质子：产物不画 H+ 副产（H 差容忍，非 H 元素守恒）。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[COMPOSITE:reaction_mech][STRUCT:c1ccccc1][PLUS][STRUCT:[N+](=O)=O]"
            "[RXNARROW][STRUCT:O=[N+]([O-])c1ccccc1][/COMPOSITE]")
        assert len(invalid) == 0

    def test_composite_row_layout_skipped(self):
        """row 多步合成序列（辅助试剂写箭头条件）不做守恒检查。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[COMPOSITE:row][STRUCT:C=C][RXNARROW:H2O / H+]"
            "[STRUCT:CCO][RXNARROW:CuO, Δ][STRUCT:CC=O][/COMPOSITE]")
        assert len(invalid) == 0

    def test_bond_nonexistent_toplevel_rejected(self):
        """A1：BOND 的 a-b 必须真实成键——非相邻原子拦截（顶层形式）。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[BOND:CCC=O|0-2]")
        assert len(invalid) == 1
        assert "没有化学键" in invalid[0].reason

    def test_bond_nonexistent_composite_child_rejected(self):
        """A1：容器内子标记形式同样拦截非相邻键。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[COMPOSITE:row][STRUCT:CCC=O,id=pr][BOND:pr|0-2][/COMPOSITE]")
        assert len(invalid) == 1
        assert "没有化学键" in invalid[0].reason

    def test_xh_no_available_h_rejected(self):
        """A2：无隐含 H 的原子（[Cl-]）写 XH 拦截——幽灵 H。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[XH:[Cl-]|0]")
        assert len(invalid) == 1
        assert "可用隐含 H 为 0" in invalid[0].reason

    def test_xh_overstack_rejected(self):
        """A2：叠加次数超过可用 H 数拦截（醛基碳仅 1 H 叠 2 次）。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[XH:CCC=O|2,2]")
        assert len(invalid) == 1
        assert "可用隐含 H 为 1" in invalid[0].reason

    def test_xh_stacking_within_available_passes(self):
        """A2：CH2 叠 2 次（恰可用 2 个 H）放行。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[XH:CCC=O|1,1]")
        assert len(invalid) == 0

    def test_xh_composite_child_overstack_rejected(self):
        """A2：容器内多个 XH 子标记累计超限同样拦截。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[COMPOSITE:row][STRUCT:CCC=O,id=pr][XH:pr|2][XH:pr|2][/COMPOSITE]")
        assert len(invalid) == 1
        assert "可用隐含 H 为 1" in invalid[0].reason


class TestCoeffAndBalanceRules:
    """20260811：系数解析 / ARROW 当量检验 / REACTION 2b 箭头补足 / 电荷守恒。

    规则要点：
    - 系数：整数或 n/2（n 奇数），如 2CCO、1/2O2；其他分数拒绝；
    - ARROW：单→单骨架，只查 C 原子数守恒（O/H 增减允许）；
    - REACTION 2a：全元素 + 净电荷守恒；2b：条件中具体物质补足差额
      （无符号=反应物侧、-X=产物侧）；[O]/[H] 占位符禁止配平；
    - COMPOSITE reaction_mech：每步局部差额补足，跨步不求和。
    """

    def test_coeff_parse(self):
        """系数解析：整数、1/2、3/2、负系数；非法（0、1/3、2/3）拒绝。"""
        from core.tag_validator import _parse_coeff, _split_multi_coeff
        assert _parse_coeff("2CCO") == (2, "CCO")
        assert _parse_coeff("1/2O2") == (0.5, "O2")
        assert _parse_coeff("3/2O2") == (1.5, "O2")
        assert _parse_coeff("CCO") == (1, "CCO")
        assert _parse_coeff("-H2O") is not None  # 负系数仅箭头补足用
        assert _parse_coeff("0CCO") is None
        assert _parse_coeff("1/3O2") is None
        assert _parse_coeff("2/3O2") is None
        assert _split_multi_coeff("2CCO;1/2O2") == [(2, "CCO"), (0.5, "O2")]

    def test_balance_reason_gives_element_diff(self):
        """守恒失败原因含两侧元素差——可操作化（H -2 提示脱氢漏 H2）。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[REACTION:CCO|CC=O|Cu, Δ]")
        assert len(invalid) == 1
        assert "右侧相对左侧" in invalid[0].reason
        assert "H -2" in invalid[0].reason

    def test_arrow_c_conservation(self):
        """ARROW 当量检验：C 守恒通过（O/H 增减允许），C 不守恒拦截。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[ARROW:CCO,CC=O,Cu, Δ]")       # C2=C2 氧化
        assert len(invalid) == 0
        _, invalid = _validate("[ARROW:2CCO,CCOCC,浓H2SO4,140℃]")  # C4=C4 系数
        assert len(invalid) == 0
        _, invalid = _validate("[ARROW:CCO,CCC,Cu]")            # C2≠C3
        assert len(invalid) == 1
        assert "碳原子数不守恒" in invalid[0].reason

    def test_arrow_fractional_coeff_oh_allowed(self):
        """ARROW 分数系数：O/H 分数原子放行（只比 C，1/2CCOCC 的 0.5O 忽略）。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[ARROW:CCO,1/2CCOCC,浓H2SO4,140℃]")  # C2=C2
        assert len(invalid) == 0

    def test_arrow_fractional_c_rejected(self):
        """ARROW 分数系数：C 为分数（1/2×奇数 C）拦截。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[ARROW:1/2CCC,CCO,Cu]")         # 1.5C≠2C 分数 C
        assert len(invalid) == 1
        assert "物种无法计数" in invalid[0].reason

    def test_arrow_formula_species(self):
        """ARROW 物种可为化学式（O2/H2O）而非 SMILES（公式回退计数）。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[ARROW:1/2O2,H2O,燃烧]")        # C0=C0
        assert len(invalid) == 0

    def test_arrow_illegal_coeff_rejected(self):
        """ARROW 非法系数（1/3）拦截。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[ARROW:1/3O2,H2O,x]")
        assert len(invalid) == 1

    def test_reaction_2b_esterification(self):
        """2b 酯化：-H2O 补产物侧，差额抵消。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[REACTION:CC(=O)O;CCO|CC(=O)OCC|浓H2SO4, Δ, -H2O]")
        assert len(invalid) == 0

    def test_reaction_2b_hydrolysis(self):
        """2b 水解：无符号 H2O 补反应物侧（催化剂 NaOH 不误判）。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[REACTION:CC(=O)OCC|CC(=O)O;CCO|NaOH, H2O, Δ]")
        assert len(invalid) == 0

    def test_reaction_2b_ethanol_to_acetic_acid(self):
        """2b 乙醇→乙酸：O2 补反应物侧 + -H2O 补产物侧（双 token）。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[REACTION:CCO|CC(=O)O|O2, -H2O]")
        assert len(invalid) == 0

    def test_reaction_2b_ethylene_to_glycol(self):
        """2b 乙烯→乙二醇：1/2O2 分数系数补反应物侧。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[REACTION:C=C;O|OCCO|1/2O2]")
        assert len(invalid) == 0

    def test_reaction_2b_dehydration_single_to_single(self):
        """单→单也允许 2b：乙醇→乙烯 -H2O（与 ARROW 并存不冲突）。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[REACTION:CCO|C=C|-H2O]")
        assert len(invalid) == 0

    def test_reaction_2b_aromatization(self):
        """芳构化真实放氢：-3H2 补产物侧（环己烷→苯 + 3H2）。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[REACTION:C1CCCCC1|c1ccccc1|-3H2, Δ]")
        assert len(invalid) == 0

    def test_reaction_2b_placeholder_forbidden(self):
        """[O]/[H] 占位符禁止作为配平物质。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[REACTION:CCO|CC=O|[O], Cu]")
        assert len(invalid) == 1
        assert "不守恒" in invalid[0].reason

    def test_reaction_2b_no_declaration_rejected(self):
        """无箭头声明的不守恒方程式拦截（乙醇→乙醛骨架缺 O2/H2）。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[REACTION:CCO|CC=O|Cu, Δ]")
        assert len(invalid) == 1

    def test_reaction_2b_catalyst_not_mistaken(self):
        """催化剂（NaOH）出现在条件但不匹配差额 → 不误判为补足。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[REACTION:CC(=O)O;CCO|CC(=O)OCC|NaOH, Δ]")
        assert len(invalid) == 1

    def test_reaction_charge_conservation(self):
        """2a 净电荷守恒：H+ 参与的质子化放行；电荷不平衡拦截。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate("[REACTION:CCO;[H+]|CC[OH2+]]")
        assert len(invalid) == 0
        # 原子相同但电荷不等（[H] 中性 vs [H+] 带 +1）→ 纯电荷不守恒
        _, invalid = _validate("[REACTION:[H]|[H+]]")
        assert len(invalid) == 1
        assert "电荷不守恒" in invalid[0].reason

    def test_reaction_2b_charge_supplement(self):
        """2b 补足物质可含电荷（[H+] 补反应物侧平衡电荷差）。"""
        pytest.importorskip("rdkit")
        # 乙醚 + H+ → 质子化乙醚（H+ 补反应物侧，电荷 +1 一并平衡）
        _, invalid = _validate("[REACTION:CCOCC|CC[OH+]CC|[H+], Δ]")
        assert len(invalid) == 0

    def test_composite_step_local_supplement(self):
        """COMPOSITE reaction_mech 每步局部：RXNARROW 条件可 2b 补足。"""
        pytest.importorskip("rdkit")
        # 第一步：乙醇 → 乙烯（-H2O 补本步产物侧）；第二步跨步不求和
        _, invalid = _validate(
            "[COMPOSITE:reaction_mech]"
            "[STRUCT:CCO][RXNARROW:-H2O][STRUCT:C=C]"
            "[/COMPOSITE]")
        assert len(invalid) == 0

    def test_composite_step_charge_tolerated(self):
        """COMPOSITE 分步保持旁观离子省略惯例：电荷不比对。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[COMPOSITE:reaction_mech]"
            "[STRUCT:CCl][PLUS][STRUCT:[OH-]]"
            "[RXNARROW][STRUCT:CO][PLUS][STRUCT:[Cl-]]"
            "[/COMPOSITE]")
        assert len(invalid) == 0


def test_composite_mecharrow_bond_form_midpoint_passes(fake_rdkit):
    text = ("[COMPOSITE:reaction_mech][STRUCT:CCl,id=r0][RXNARROW]"
            "[STRUCT:CO,id=p0][MECHARROW:r0:0>>r0:0+p0:0][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 0
    text = ("[COMPOSITE:reaction_mech][STRUCT:CCl,id=r0][RXNARROW]"
            "[STRUCT:CO,id=p0][MECHARROW:r0:0>>r0:0+p0:0][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 0


def test_composite_mecharrow_midpoint_unknown_id(fake_rdkit):
    text = ("[COMPOSITE:reaction_mech][STRUCT:CCl,id=r0][RXNARROW]"
            "[STRUCT:CO,id=p0][MECHARROW:r0:0>r0:0+ghost:0][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "未知组件" in invalid[0].reason


def test_composite_mecharrow_midpoint_atom_out_of_range(fake_rdkit):
    text = ("[COMPOSITE:reaction_mech][STRUCT:CCl,id=r0][RXNARROW]"
            "[STRUCT:CO,id=p0][MECHARROW:r0:0>r0:0+p0:9][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "超出范围" in invalid[0].reason


def test_composite_mecharrow_midpoint_bond_mixed_rejected(fake_rdkit):
    text = ("[COMPOSITE:reaction_mech][STRUCT:CCl,id=r0][RXNARROW]"
            "[STRUCT:CO,id=p0][MECHARROW:r0:0>r0:0-1+p0:0][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "成键空白位端点格式错误" in invalid[0].reason


class TestMechArrowExplicitH:
    """B1（20260812）：MECHARROW 端点 a#k（显式 H 引用）校验。

    自由基夺氢机理：单碳组分 C（CH4）加 [XH] 显式画 H，用 ch4:0#1 引用。
    真实 RDKit（fake_rdkit 白名单不含 C/[Cl]/[CH3]）。
    """

    def test_explicit_h_endpoint_passes(self):
        """[XH:ch4|0] + ch4:0#1 引用 → 放行。"""
        pytest.importorskip("rdkit")
        text = ("[COMPOSITE:reaction_mech]"
                "[STRUCT:[Cl],id=cl,label=Cl·][PLUS][STRUCT:C,id=ch4,label=CH4][XH:ch4|0]"
                "[RXNARROW]"
                "[STRUCT:Cl,id=hcl,label=HCl][PLUS][STRUCT:[CH3],id=me,label=·CH3]"
                "[MECHARROW:cl:0>>cl:0+ch4:0]"
                "[MECHARROW:ch4:0#1>>cl:0+ch4:0]"
                "[MECHARROW:ch4:0#1>>me:0]"
                "[/COMPOSITE]")
        _, invalid = _validate(text)
        assert len(invalid) == 0

    def test_explicit_h_without_xh_rejected(self):
        """a#k 但组件未声明 [XH] → 拦截（提示需先画显式 H）。"""
        pytest.importorskip("rdkit")
        text = ("[COMPOSITE:reaction_mech]"
                "[STRUCT:[Cl],id=cl,label=Cl·][PLUS][STRUCT:C,id=ch4,label=CH4]"
                "[RXNARROW]"
                "[STRUCT:Cl,id=hcl,label=HCl][PLUS][STRUCT:[CH3],id=me,label=·CH3]"
                "[MECHARROW:cl:0>>cl:0+ch4:0]"
                "[MECHARROW:ch4:0#1>>me:0]"
                "[/COMPOSITE]")
        _, invalid = _validate(text)
        assert len(invalid) == 1
        assert "未先写 [XH" in invalid[0].reason

    def test_explicit_h_index_out_of_range_rejected(self):
        """k 超出该原子显式 H 数（[XH:ch4|0] 只画 1 个，写 0#2）→ 拦截。"""
        pytest.importorskip("rdkit")
        text = ("[COMPOSITE:reaction_mech]"
                "[STRUCT:[Cl],id=cl,label=Cl·][PLUS][STRUCT:C,id=ch4,label=CH4][XH:ch4|0]"
                "[RXNARROW]"
                "[STRUCT:Cl,id=hcl,label=HCl][PLUS][STRUCT:[CH3],id=me,label=·CH3]"
                "[MECHARROW:ch4:0#2>>me:0]"
                "[/COMPOSITE]")
        _, invalid = _validate(text)
        assert len(invalid) == 1
        assert "只画了" in invalid[0].reason

    def test_legacy_implicit_h_bond_rejected(self):
        """旧写法 ch4:0-1（隐含 H 键）仍拦截——提示改用 a#k（B1 回归）。"""
        pytest.importorskip("rdkit")
        text = ("[COMPOSITE:reaction_mech]"
                "[STRUCT:[Cl],id=cl,label=Cl·][PLUS][STRUCT:C,id=ch4,label=CH4][XH:ch4|0]"
                "[RXNARROW]"
                "[STRUCT:Cl,id=hcl,label=HCl][PLUS][STRUCT:[CH3],id=me,label=·CH3]"
                "[MECHARROW:ch4:0-1>>me:0]"
                "[/COMPOSITE]")
        _, invalid = _validate(text)
        assert len(invalid) == 1
        assert "键端点" in invalid[0].reason


def test_xh_toplevel_valid(fake_rdkit):
    _, invalid = _validate("[XH:CC=O|1]")
    assert len(invalid) == 0


def test_xh_toplevel_out_of_range(fake_rdkit):
    _, invalid = _validate("[XH:CC=O|9]")
    assert len(invalid) == 1
    assert "超出范围" in invalid[0].reason


def test_xh_toplevel_missing_spec(fake_rdkit):
    _, invalid = _validate("[XH:CC=O]")
    assert len(invalid) == 1
    assert "缺少标注参数" in invalid[0].reason


def test_bond_toplevel_valid(fake_rdkit):
    _, invalid = _validate("[BOND:CC=O|1-2]")
    assert len(invalid) == 0


def test_bond_toplevel_bad_format(fake_rdkit):
    _, invalid = _validate("[BOND:CC=O|1x2]")
    assert len(invalid) == 1
    assert "格式错误" in invalid[0].reason


def test_bond_toplevel_out_of_range(fake_rdkit):
    _, invalid = _validate("[BOND:CC=O|1-9]")
    assert len(invalid) == 1
    assert "超出范围" in invalid[0].reason


def test_composite_xh_child_unknown_id(fake_rdkit):
    text = ("[COMPOSITE:row][STRUCT:CC=O,id=pr][XH:ghost|1][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "未知组件" in invalid[0].reason


def test_composite_xh_child_bad_format(fake_rdkit):
    text = ("[COMPOSITE:row][STRUCT:CC=O,id=pr][XH:pr|x][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "不是数字" in invalid[0].reason


def test_composite_xh_child_out_of_range(fake_rdkit):
    text = ("[COMPOSITE:row][STRUCT:CC=O,id=pr][XH:pr|9][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "超出" in invalid[0].reason


def test_composite_bond_child_unknown_id(fake_rdkit):
    text = ("[COMPOSITE:row][STRUCT:CC=O,id=pr][BOND:ghost|1-2][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "未知组件" in invalid[0].reason


def test_composite_bond_child_bad_format(fake_rdkit):
    text = ("[COMPOSITE:row][STRUCT:CC=O,id=pr][BOND:pr|1x2][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "格式错误" in invalid[0].reason


def test_composite_bond_child_out_of_range(fake_rdkit):
    text = ("[COMPOSITE:row][STRUCT:CC=O,id=pr][BOND:pr|1-9][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "超出" in invalid[0].reason


def test_composite_real_mechanism_passes(fake_rdkit):
    text = ("[COMPOSITE:reaction_mech][STRUCT:CCO,label=乙醇,id=nu][PLUS]"
            "[STRUCT:CC[OH2+],label=质子化的乙醇,id=pe][RXNARROW]"
            "[STRUCT:CC[OH+]CC,label=质子化的乙醚][PLUS][STRUCT:O,label=水]"
            "[MECHARROW:nu:2>pe:1][MECHARROW:pe:1-2>pe:2][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 0


def test_composite_charge_unknown_id(fake_rdkit):
    text = "[COMPOSITE:row][STRUCT:CCl,id=a][CHARGE:ghost|0:δ-][/COMPOSITE]"
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "未知组件" in invalid[0].reason


def test_pipeline_degrades_invalid_tags(fake_rdkit, fake_renderers, monkeypatch):
    llm_text = ("苯是 [STRUCT:c1ccccc1]，无效 [STRUCT:XYZABC]，"
                "超长 [STRUCT:CO,label=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]。")
    monkeypatch.setattr("app.ask_llm", lambda *a, **k: llm_text)
    diag = []
    result = process_question("测试", diagnostics=diag)
    # 部分修正：非法/超长标记被修正输出中的合法标记替换修复（无降级）
    assert "RENDERED:c1ccccc1" in result
    assert "图示无法渲染" not in result
    # 后端：diagnostics 记录每轮失败（最终被修复 → resolved=True）
    assert len(diag) >= 2
    assert any("无效 SMILES" in d["reason"] for d in diag)
    assert any("label 过长" in d["reason"] for d in diag)
    assert all(d["resolved"] is True for d in diag)


def test_benzene_style_consistency_warning():
    """软提示：同一机理内苯环写法不一致（圆圈式/凯库勒A/B）→ 警告。

    不拦截（标记仍通过校验），通过 get_last_warnings() 暴露。
    """
    pytest.importorskip("rdkit")
    from core.tag_validator import get_last_warnings

    # 场景1：苯用凯库勒A，硝基苯用凯库勒B → 警告
    text = (
        "[COMPOSITE:reaction_mech][STRUCT:C1C=CC=CC=1,label=苯,id=ar]"
        "[PLUS][STRUCT:[N+](=O)=O,label=NO2+,id=nu]"
        "[RXNARROW][STRUCT:O=[N+]([O-])[CH]1C=CC=C[CH+]1,label=σ 络合物,id=sigma]"
        "[MECHARROW:ar:0-5>nu:0][/COMPOSITE]"
        "\n"
        "[COMPOSITE:reaction_mech][STRUCT:O=[N+]([O-])[CH]1C=CC=C[CH+]1,label=σ 络合物,id=sigma]"
        "[RXNARROW][STRUCT:O=[N+]([O-])C1=CC=CC=C1,label=硝基苯][PLUS][STRUCT:[H+],label=H+]"
        "[XH:sigma|3][MECHARROW:sigma:3#1>sigma:3-8][/COMPOSITE]"
    )
    _, invalid = _validate(text)
    assert len(invalid) == 0  # 不拦截
    assert get_last_warnings(), "应产生苯环写法不一致警告"

    # 场景2：都用凯库勒A → 无警告
    text2 = text.replace("O=[N+]([O-])C1=CC=CC=C1", "O=[N+]([O-])C1C=CC=CC=1")
    _, invalid2 = _validate(text2)
    assert len(invalid2) == 0
    assert not get_last_warnings(), "写法一致不应警告"

    # 场景3：圆圈式 + 凯库勒混用 → 警告
    text3 = (
        "[COMPOSITE:row][STRUCT:c1ccccc1,label=苯,id=ar][/COMPOSITE]"
        "\n"
        "[COMPOSITE:row][STRUCT:C1=CC=CC=C1,label=苯,id=ar][/COMPOSITE]"
    )
    _, invalid3 = _validate(text3)
    assert len(invalid3) == 0
    assert get_last_warnings(), "圆圈+凯库勒混用应警告"


# ---------------------------------------------------------------------------
# 双轨制（20260814）：REACTION 物种 = 合法 SMILES 或教科书化学式（KMnO4 等）。
# 这些测试用真实 RDKit（不用 fake_rdkit fixture），保证守恒校验真实生效。
# ---------------------------------------------------------------------------


def test_reaction_kmno4_oxidation_balances():
    """双轨制核心场景：乙醇被 KMnO4 氧化（酸性）完整方程式通过守恒校验。

    5CCO + 4KMnO4 + 6H2SO4 → 5CH3COOH + 4MnSO4 + 2K2SO4 + 11H2O
    有机物（CCO）走 SMILES 轨、无机物（KMnO4/H2SO4/MnSO4/K2SO4/H2O）与
    CH3COOH 走化学式轨——两侧原子与电荷守恒应通过。
    """
    pytest.importorskip("rdkit")
    text = ("[REACTION:5CCO;4KMnO4;6H2SO4"
            "|5CH3COOH;4MnSO4;2K2SO4;11H2O|Δ]")
    _, invalid = _validate(text)
    assert len(invalid) == 0, f"KMnO4 氧化乙醇应通过守恒校验: {invalid}"


def test_reaction_formula_species_only():
    """纯化学式轨：所有物种都用教科书化学式（无 SMILES）也应通过。"""
    pytest.importorskip("rdkit")
    text = ("[REACTION:5CH3CH2OH;4KMnO4;6H2SO4"
            "|5CH3COOH;4MnSO4;2K2SO4;11H2O|Δ]")
    _, invalid = _validate(text)
    assert len(invalid) == 0, f"纯化学式轨应通过: {invalid}"


def test_reaction_mixed_tracks():
    """混合轨：同一方程式中 SMILES 与化学式物种共存（乙醇氧化成乙醛）。"""
    pytest.importorskip("rdkit")
    text = ("[REACTION:CCO;[O-][Mn](=O)(=O)=O|CC=O|]")
    _, invalid = _validate(text)
    # 物种均可解析（SMILES + 化学式），但反应式不守恒（KMnO4 还原产物缺失）
    assert len(invalid) == 1
    assert "化学校验" in invalid[0].reason


def test_reaction_formula_inorganic_salt():
    """化学式轨无机盐可解析（非 SMILES 但化学式合法）：高锰酸钾单物种。"""
    pytest.importorskip("rdkit")
    from core.tag_validator import _parse_plain_formula
    cands = _parse_plain_formula("KMnO4")
    assert cands and cands[0][0] == {"K": 1, "Mn": 1, "O": 4}
    cands2 = _parse_plain_formula("K2Cr2O7")
    assert cands2 and cands2[0][0] == {"K": 2, "Cr": 2, "O": 7}
    cands3 = _parse_plain_formula("CH3COOH")
    assert cands3 and cands3[0][0] == {"C": 2, "H": 4, "O": 2}


def test_reaction_unresolvable_species_rejected():
    """既非 SMILES 也非化学式（占位符/乱码）仍被拦截，且保留「无效 SMILES」关键词。"""
    pytest.importorskip("rdkit")
    _, invalid = _validate("[REACTION:CCO;XYZABC|CC=O|]")
    assert len(invalid) == 1
    assert "无效 SMILES" in invalid[0].reason


def test_formula_tail_digit_disambiguation():
    """化学式尾数字歧义消解（2026-08-14 修复）：

    - 多位尾数字：最后一位归电荷（SO42- → SO4 带 -2、Cr2O72- → Cr2O7 带 -2）；
    - 单数字 + 多元素：归元素（FeBr4- → FeBr4 带 -1、NO2+ → NO2 带 +1）；
    - 单数字 + 单元素：归电荷（Ca2+ → Ca 带 +2、Al3+ → Al 带 +3）。
    回归锚点：FeBr4- 曾误解析为 FeBr 带 -4（1 个 Br、4 个负电荷）。
    """
    pytest.importorskip("rdkit")
    from core.tag_validator import _formula_or_smiles_counts as f
    assert f("FeBr4-") == ({"Fe": 1, "Br": 4}, -1)
    assert f("NO2+") == ({"N": 1, "O": 2}, 1)
    assert f("SO42-") == ({"S": 1, "O": 4}, -2)
    assert f("Cr2O72-") == ({"Cr": 2, "O": 7}, -2)
    assert f("NH4+") == ({"N": 1, "H": 4}, 1)
    assert f("HSO4-") == ({"H": 1, "S": 1, "O": 4}, -1)
    assert f("CO32-") == ({"C": 1, "O": 3}, -2)
    assert f("Ca2+") == ({"Ca": 1}, 2)
    assert f("Al3+") == ({"Al": 1}, 3)
    assert f("Fe3+") == ({"Fe": 1}, 3)
    assert f("OH-") == ({"O": 1, "H": 1}, -1)


def test_reaction_formula_unbalanced_rejected():
    """化学式轨同样受守恒约束：乙醇+氧气不守恒（缺产物水）被拦截。"""
    pytest.importorskip("rdkit")
    text = "[REACTION:CCO;O2|CH3COOH|]"
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "化学校验" in invalid[0].reason

