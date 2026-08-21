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



def test_empty_smiles_rejected():
    _, invalid = _validate("[STRUCT:]")
    assert len(invalid) == 1
    assert "SMILES 为空" in invalid[0].reason


def test_long_label_rejected():
    _, invalid = _validate("[STRUCT:CO,label=" + "长" * 25 + "]")
    assert len(invalid) == 1
    assert "label 过长" in invalid[0].reason


def test_newman_bad_angle_rejected():
    _, invalid = _validate("[STRUCT:CC,mode=newman,angle=xyz]")
    assert len(invalid) == 1
    assert "角度" in invalid[0].reason


def test_newman_angle_out_of_range():
    _, invalid = _validate("[STRUCT:CC,mode=newman,angle=450]")
    assert len(invalid) == 1


def test_newman_with_bond_valid():
    """mode=newman + bond=a-b：合法键放行。"""
    _, invalid = _validate("[STRUCT:CC,mode=newman,bond=0-1,angle=60]")
    assert len(invalid) == 0


def test_newman_bond_not_exist_rejected():
    """键 a-b 不存在（索引在范围内但不成键）→ 拦截。"""
    _, invalid = _validate("[STRUCT:CCC,mode=newman,bond=0-2,angle=60]")
    assert len(invalid) == 1
    assert "键" in invalid[0].reason


def test_newman_bond_atom_out_of_range():
    """原子序号越界（5 超过原子数 2）→ 拦截。"""
    _, invalid = _validate("[STRUCT:CC,mode=newman,bond=5-1,angle=60]")
    assert len(invalid) == 1
    assert "越界" in invalid[0].reason


def test_newman_bond_missing_angle():
    """新格式缺角度 → 拦截。"""
    _, invalid = _validate("[STRUCT:CC,mode=newman,bond=0-1]")
    assert len(invalid) == 1
    assert "角度" in invalid[0].reason


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
    """非 row 布局无 STRUCT 仍拦截（reaction_mech 需要组件供机理引用）。"""
    _, invalid = _validate("[COMPOSITE:reaction][ARROW:type=single][/COMPOSITE]")
    assert len(invalid) == 1
    assert "缺少 [STRUCT]" in invalid[0].reason


def test_struct_mode_invalid_rejected():
    """STRUCT mode 非法枚举拦截。"""
    _, invalid = _validate("[STRUCT:O, mode=xyz]")
    assert len(invalid) == 1
    assert "未知 STRUCT 模式" in invalid[0].reason


def test_struct_mode_lewis_stereo_passes():
    """STRUCT mode=lewis/stereo 放行（分子家族统一写法）。"""
    _, invalid = _validate("[STRUCT:O, mode=lewis, label=水]")
    assert len(invalid) == 0
    _, invalid2 = _validate(
        "[STRUCT:C[C@H](O)C(=O)O, mode=stereo, label=(R)-乳酸]")
    assert len(invalid2) == 0


def test_struct_mode_newman_passes():
    """STRUCT mode=newman：bond 存在性 + angle 范围校验（与 [NEWMAN] 等价）。"""
    _, invalid = _validate("[STRUCT:CC, mode=newman, bond=0-1, angle=60]")
    assert len(invalid) == 0
    _, invalid2 = _validate("[STRUCT:CC, mode=newman, bond=0-1, angle=xyz]")
    assert len(invalid2) == 1
    assert "角度" in invalid2[0].reason
    _, invalid3 = _validate("[STRUCT:CC, mode=newman, bond=0-5, angle=60]")
    assert len(invalid3) == 1


def test_struct_mode_chair_passes():
    """STRUCT mode=chair：subs 校验（与 [CHAIR] 等价）。"""
    _, invalid = _validate("[STRUCT:BrC1CCCCC1, mode=chair, subs=1:ax]")
    assert len(invalid) == 0
    _, invalid2 = _validate("[STRUCT:BrC1CCCCC1, mode=chair, subs=1:xx]")
    assert len(invalid2) == 1
    assert "格式错误" in invalid2[0].reason


def test_struct_bond_charge_params():
    """20260821：STRUCT bond=/charge= 参数化标注校验（单分子标注新写法）。"""
    # 合法：bond=a-b 真实成键（CCC=O 的 1-2 是 C=O 键）+ charge=idx:+/- 列表
    _, invalid = _validate("[STRUCT:CCC=O, bond=1-2, charge=0:+,3:-]")
    assert len(invalid) == 0
    _, invalid = _validate("[STRUCT:CCC=O, charge=0:+]")
    assert len(invalid) == 0
    # bond 越界 / 不存在的键
    _, invalid = _validate("[STRUCT:CCC=O, bond=0-5]")
    assert len(invalid) == 1 and "超出原子范围" in invalid[0].reason
    _, invalid = _validate("[STRUCT:CC=O, bond=0-2]")   # CC=O 的 0-2 无键
    assert len(invalid) == 1 and "不存在" in invalid[0].reason
    # charge 格式（只收裸 +/-）与越界
    _, invalid = _validate("[STRUCT:CCC=O, charge=0:x]")
    assert len(invalid) == 1 and "charge 标注格式错误" in invalid[0].reason
    _, invalid = _validate("[STRUCT:CCC=O, charge=9:+]")
    assert len(invalid) == 1 and "超出原子范围" in invalid[0].reason
    # 容器内同样生效（与 [BOND]/[CHARGE] 子标记等价）
    _, invalid = _validate(
        "[COMPOSITE:row][STRUCT:CCC=O,id=pr,bond=1-2,charge=0:+][/COMPOSITE]")
    assert len(invalid) == 0
    _, invalid = _validate(
        "[COMPOSITE:row][STRUCT:CCC=O,id=pr,bond=0-5][/COMPOSITE]")
    assert len(invalid) == 1


def test_composite_mode_restriction():
    """容器内 mode 按布局放开（20260821）：reaction 禁 newman，row/energy 不限。"""
    _, invalid = _validate(
        "[COMPOSITE:row][STRUCT:O, mode=lewis, id=w][/COMPOSITE]")
    assert len(invalid) == 0
    # row 布局：stereo/chair/newman 均放行
    _, invalid2 = _validate(
        "[COMPOSITE:row][STRUCT:C[C@H](O)C(=O)O, mode=stereo, id=w]"
        "[/COMPOSITE]")
    assert len(invalid2) == 0
    _, invalid3 = _validate(
        "[COMPOSITE:row][STRUCT:CC, mode=newman, bond=0-1, angle=60, id=w]"
        "[/COMPOSITE]")
    assert len(invalid3) == 0
    # reaction 布局：stereo/chair 放行、newman 拦截
    _, invalid4 = _validate(
        "[COMPOSITE:reaction][STRUCT:C[C@H](O)C(=O)O, mode=stereo, id=a]"
        "[ARROW:type=single][STRUCT:C[C@H](O)C(=O)O, mode=stereo, id=b]"
        "[/COMPOSITE]")
    assert len(invalid4) == 0
    _, invalid5 = _validate(
        "[COMPOSITE:reaction][STRUCT:CC, mode=newman, bond=0-1, angle=60, id=w]"
        "[/COMPOSITE]")
    assert len(invalid5) == 1
    assert "mode=newman" in invalid5[0].reason
    # energy 布局：newman 放行
    _, invalid6 = _validate(
        "[COMPOSITE:energy][ENERGY:0,10]"
        "[STRUCT:CC, mode=newman, bond=0-1, angle=60, id=w, at=0][/COMPOSITE]")
    assert len(invalid6) == 0


def test_opaque_comp_reference_rejected():
    """立体画法组件（stereo/chair/newman）仅展示：机理/标注/附件引用拦截。"""
    # MECHARROW 端点引用 stereo 组件
    _, invalid = _validate(
        "[COMPOSITE:reaction][STRUCT:CCl,id=r0][PLUS][STRUCT:[OH-],id=nu]"
        "[ARROW:type=single][STRUCT:CO][PLUS][STRUCT:[Cl-]]"
        "[MECHARROW:nu:0>r0:0]"
        "[STRUCT:C[C@H](O)C(=O)O, mode=stereo, id=st]"
        "[MECHARROW:nu:0>st:1][/COMPOSITE]")
    assert any("立体画法组件" in r.reason for r in invalid)
    # CHARGE 标注引用 chair 组件
    _, invalid2 = _validate(
        "[COMPOSITE:row][STRUCT:BrC1CCCCC1, mode=chair, subs=1:ax, id=c]"
        "[CHARGE:c|0:δ+][/COMPOSITE]")
    assert any("立体画法组件" in r.reason for r in invalid2)
    # 箭头附件（arrow 令牌）不允许立体画法组件
    _, invalid3 = _validate(
        "[COMPOSITE:reaction][STRUCT:CCO,id=a]"
        "[STRUCT:BrC1CCCCC1, mode=chair, subs=1:ax, id=w, arrow]"
        "[ARROW:type=single,sup=+w][STRUCT:CC=O,id=b][/COMPOSITE]")
    assert any("arrow 令牌" in r.reason for r in invalid3)


def test_composite_coefficient_prefix():
    """容器内 STRUCT 支持化学计量系数前缀（2CCO）：剥离后校验 SMILES。"""
    _, invalid = _validate(
        "[COMPOSITE:reaction][STRUCT:2CCO,id=a][ARROW:type=single,O2]"
        "[STRUCT:2CC=O,id=b][/COMPOSITE]")
    assert len(invalid) == 0
    _, invalid2 = _validate(
        "[COMPOSITE:reaction][STRUCT:0CCO,id=a][ARROW:type=single]"
        "[STRUCT:CC=O,id=b][/COMPOSITE]")
    assert len(invalid2) == 1
    assert "系数" in invalid2[0].reason


# ---------- 大一统架构：reaction 布局（20260819） ----------


def test_reaction_single_to_single_equivalent():
    """reaction 布局单→单：C 当量（原 ARROW 逻辑）；不等拦截。"""
    _, invalid = _validate(
        "[COMPOSITE:reaction][STRUCT:CCO,id=A][ARROW:type=single]"
        "[STRUCT:CC=O,id=B][/COMPOSITE]")
    assert len(invalid) == 0
    _, invalid2 = _validate(
        "[COMPOSITE:reaction][STRUCT:CCO,id=A][ARROW:type=single]"
        "[STRUCT:CCC,id=B][/COMPOSITE]")
    assert len(invalid2) == 1
    assert "C 原子数不等" in invalid2[0].reason


def test_reaction_multi_full_balance():
    """reaction 布局任一边 ≥2：完整原子+电荷守恒（原 REACTION 2a）。"""
    _, invalid = _validate(
        "[COMPOSITE:reaction][STRUCT:CC(=O)O,id=A][PLUS][STRUCT:CCO,id=B]"
        "[ARROW:type=single,浓H2SO4]"
        "[STRUCT:CC(=O)OCC,id=C][PLUS][STRUCT:O,id=D][/COMPOSITE]")
    assert len(invalid) == 0
    _, invalid2 = _validate(
        "[COMPOSITE:reaction][STRUCT:CC(=O)O,id=A][PLUS][STRUCT:CCO,id=B]"
        "[ARROW:type=single]"
        "[STRUCT:CC(=O)OCC,id=C][/COMPOSITE]")
    assert len(invalid2) == 1
    assert "化学校验" in invalid2[0].reason


def test_reaction_sup_attachment_balance():
    """sup 附件参与补足：副反应物(+E)计左侧、副产物(-F)计右侧。"""
    _, invalid = _validate(
        "[COMPOSITE:reaction][STRUCT:CC(=O)O,id=A][PLUS][STRUCT:CCO,id=B]"
        "[STRUCT:O,id=w,arrow]"
        "[ARROW:type=single,sup=-w,浓H2SO4]"
        "[STRUCT:CC(=O)OCC,id=C][/COMPOSITE]")
    assert len(invalid) == 0
    # 附件未声明 arrow 令牌 → 拦截
    _, invalid2 = _validate(
        "[COMPOSITE:reaction][STRUCT:CC(=O)O,id=A][PLUS][STRUCT:CCO,id=B]"
        "[STRUCT:O,id=w]"
        "[ARROW:type=single,sup=-w]"
        "[STRUCT:CC(=O)OCC,id=C][/COMPOSITE]")
    assert len(invalid2) == 1
    assert "arrow 令牌" in invalid2[0].reason


def test_reaction_arrow_type_invalid():
    """ARROW type 非法枚举拦截。"""
    _, invalid = _validate(
        "[COMPOSITE:reaction][STRUCT:CCl,id=A][ARROW:type=foo]"
        "[STRUCT:CO,id=B][/COMPOSITE]")
    assert len(invalid) == 1
    assert "ARROW 类型" in invalid[0].reason


def test_reaction_block_balance():
    """BLOCK 共振块作为单一结构参与每步守恒（块分子式 = 首个 STRUCT）。"""
    _, invalid = _validate(
        "[COMPOSITE:reaction][STRUCT:c1ccccc1,id=A]"
        "[ARROW:type=single,条件]"
        "[BLOCK][STRUCT:C1=CC=CC=C1,id=B1][ARROW:type=resonance]"
        "[STRUCT:C1C=CC=CC=1,id=B2][/BLOCK]"
        "[ARROW:type=single,条件]"
        "[STRUCT:O=[N+]([O-])c1ccccc1,id=C][/COMPOSITE]")
    assert len(invalid) == 0
    # 块内箭头非 resonance → 拦截
    _, invalid2 = _validate(
        "[COMPOSITE:reaction][BLOCK][STRUCT:C1=CC=CC=C1,id=B1]"
        "[ARROW:type=single][STRUCT:C1C=CC=CC=1,id=B2][/BLOCK][/COMPOSITE]")
    assert len(invalid2) == 1
    assert "resonance" in invalid2[0].reason


def test_reaction_retro_balance():
    """逆合成箭头：宽松当量（前体 C 数 ≤ 目标；断键不增碳）。"""
    _, invalid = _validate(
        "[COMPOSITE:reaction][STRUCT:O=Cc1ccccc1,id=T]"
        "[ARROW:type=retro,formylation][STRUCT:c1ccccc1,id=P][/COMPOSITE]")
    assert len(invalid) == 0
    _, invalid2 = _validate(
        "[COMPOSITE:reaction][STRUCT:c1ccccc1,id=T]"
        "[ARROW:type=retro][STRUCT:CCc1ccccc1,id=P][/COMPOSITE]")
    assert len(invalid2) == 1
    assert "前体 C" in invalid2[0].reason


def test_block_mecharrow_supported():
    """BLOCK 内 MECHARROW（共振式间转化）放行；块内/跨块混合引用支持。"""
    # 块内↔块内
    _, invalid = _validate(
        "[COMPOSITE:reaction]"
        "[BLOCK][STRUCT:C1=CC=CC=C1,id=b1][ARROW:type=resonance]"
        "[STRUCT:C1C=CC=CC=1,id=b2]"
        "[MECHARROW:b1:0>b2:1][/BLOCK][/COMPOSITE]")
    assert len(invalid) == 0
    # 跨块（块内 → 块外）
    _, invalid2 = _validate(
        "[COMPOSITE:reaction]"
        "[BLOCK][STRUCT:C1=CC=CC=C1,id=b1][ARROW:type=resonance]"
        "[STRUCT:C1C=CC=CC=1,id=b2][/BLOCK]"
        "[ARROW:type=single]"
        "[STRUCT:c1ccccc1,id=C]"
        "[MECHARROW:b2:0>C:0][/COMPOSITE]")
    assert len(invalid2) == 0
    # 块内引用未知组件 → 拦截
    _, invalid3 = _validate(
        "[COMPOSITE:reaction]"
        "[BLOCK][STRUCT:C1=CC=CC=C1,id=b1]"
        "[MECHARROW:ghost:0>b1:1][/BLOCK][/COMPOSITE]")
    assert len(invalid3) == 1
    assert "未知组件" in invalid3[0].reason


def test_block_id_global_unique():
    """块内/块外组件 id 全局查重。"""
    _, invalid = _validate(
        "[COMPOSITE:reaction][STRUCT:C1=CC=CC=C1,id=x]"
        "[BLOCK][STRUCT:C1C=CC=CC=1,id=x][/BLOCK][/COMPOSITE]")
    assert len(invalid) == 1
    assert "id 重复" in invalid[0].reason


def test_composite_row_without_struct_passes():
    """row 布局允许无 [STRUCT]（纯箭头/条件/连接符序列合法）——要求已删除。"""
    _, invalid = _validate("[COMPOSITE:row][PLUS][ARROW:type=single,条件][/COMPOSITE]")
    assert len(invalid) == 0
    # row 无组件时 MECHARROW 引用仍拦截（无组件可引用）
    _, invalid2 = _validate("[COMPOSITE:row][MECHARROW:r0:0>r0:1][/COMPOSITE]")
    assert len(invalid2) == 1
    assert "未知组件" in invalid2[0].reason


def test_composite_energy_at_out_of_range(fake_rdkit):
    text = ("[COMPOSITE:energy][ENERGY:0,108,-20]"
            "[STRUCT:CCl,at=0][STRUCT:CO,at=9][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "at=" in invalid[0].reason


def test_composite_mecharrow_unknown_id(fake_rdkit):
    text = ("[COMPOSITE:reaction][STRUCT:CCl,id=r0][ARROW:type=single]"
            "[STRUCT:CO,id=p0][MECHARROW:ghost:0>r0:0][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "未知组件" in invalid[0].reason


def test_composite_mecharrow_atom_out_of_range(fake_rdkit):
    text = ("[COMPOSITE:reaction][STRUCT:CCl,id=r0][ARROW:type=single]"
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
    text = ("[COMPOSITE:reaction][STRUCT:CC=O,id=ald][ARROW:type=single]"
            "[STRUCT:CCO,id=p0][MECHARROW:ald:0>ald:0-2][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "没有成键" in invalid[0].reason


def test_bond_ref_gives_neighbor_hint():
    """键端点引用无键时，原因含原子实际连接——可操作化（帮助重数索引）。"""
    pytest.importorskip("rdkit")
    text = ("[COMPOSITE:reaction]"
            "[STRUCT:O=S([O-])(=O)C1C=CC=C[CH+]1,id=sigma]"
            "[MECHARROW:sigma:3-8>sigma:8][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "没有成键" in invalid[0].reason
    assert "实际连接" in invalid[0].reason


def test_composite_mecharrow_existing_bond_passes():
    """a-b 端点引用真实存在的键（乙醛 0-1）放行。"""
    pytest.importorskip("rdkit")
    text = ("[COMPOSITE:reaction][STRUCT:CC=O,id=ald][ARROW:type=single]"
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






    def test_degrade_text_chem_clean_for_user(self):
        """用户可见降级消息：去「化学校验：」前缀与括号详情/修正指导，
        只留主因；完整原因仍保留在 reason 中供 P2 修正与 metrics 使用。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[COMPOSITE:reaction][STRUCT:CCO,id=a][PLUS][STRUCT:O,id=w]"
            "[ARROW:type=single,Cu, Δ][STRUCT:CC=O,id=b][/COMPOSITE]")
        tag, reason = invalid[0].tag, invalid[0].reason
        shown = tv.degrade_text(tag, reason)
        assert shown == "（复合图图示无法渲染：第 1 步两侧原子不守恒，已省略）"
        assert "化学校验：" not in shown
        assert "辅助试剂" not in shown          # 修正指导不再面向用户
        assert "C2H8O2" in reason               # 完整原因仍保留

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
        _, invalid2 = _validate(
            "[COMPOSITE:reaction][STRUCT:CCO,id=a][PLUS][STRUCT:O,id=w]"
            "[ARROW:type=single][STRUCT:CC=O,id=b][/COMPOSITE]")
        shown2 = tv.degrade_text_friendly(invalid2[0].tag)
        assert shown2 == "（复合图图示无法渲染，已省略）"
        assert "不守恒" not in shown2
        assert "化学校验" not in shown2


    def test_composite_mech_balanced_passes(self):
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[COMPOSITE:reaction][STRUCT:CCl][PLUS][STRUCT:[OH-]]"
            "[ARROW:type=single][STRUCT:CO][PLUS][STRUCT:[Cl-]][/COMPOSITE]")
        assert len(invalid) == 0

    def test_composite_mech_unbalanced_rejected(self):
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[COMPOSITE:reaction][STRUCT:CCO][PLUS][STRUCT:C[OH2+]]"
            "[ARROW:type=single][STRUCT:CCOCC][PLUS][STRUCT:O][/COMPOSITE]")
        assert len(invalid) == 1
        assert "化学校验" in invalid[0].reason

    def test_composite_mech_deprotonation_strict_h(self):
        """EAS 去质子（reaction 布局 H 严格）：产物写 [H+] 放行；
        省略 H+ 副产 → H 不守恒拦截（旧 reaction_mech 的 H 容忍已随
        B2 清理移除）。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[COMPOSITE:reaction][STRUCT:c1ccccc1][PLUS][STRUCT:[N+](=O)=O]"
            "[ARROW:type=single][STRUCT:O=[N+]([O-])c1ccccc1]"
            "[PLUS][STRUCT:[H+]][/COMPOSITE]")
        assert len(invalid) == 0
        _, invalid2 = _validate(
            "[COMPOSITE:reaction][STRUCT:c1ccccc1][PLUS][STRUCT:[N+](=O)=O]"
            "[ARROW:type=single][STRUCT:O=[N+]([O-])c1ccccc1][/COMPOSITE]")
        assert len(invalid2) == 1
        assert "不守恒" in invalid2[0].reason

    def test_composite_row_layout_skipped(self):
        """row 多步合成序列（辅助试剂写箭头条件）不做守恒检查。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[COMPOSITE:row][STRUCT:C=C][ARROW:type=single,H2O / H+]"
            "[STRUCT:CCO][ARROW:type=single,CuO, Δ][STRUCT:CC=O][/COMPOSITE]")
        assert len(invalid) == 0


    def test_bond_nonexistent_composite_child_rejected(self):
        """A1：容器内子标记形式同样拦截非相邻键。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[COMPOSITE:row][STRUCT:CCC=O,id=pr][BOND:pr|0-2][/COMPOSITE]")
        assert len(invalid) == 1
        assert "没有化学键" in invalid[0].reason



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
        _, invalid = _validate(
            "[COMPOSITE:reaction][STRUCT:CCO,id=a][PLUS][STRUCT:O,id=w]"
            "[ARROW:type=single][STRUCT:CC=O,id=b][/COMPOSITE]")
        assert len(invalid) == 1
        assert "右侧相对左侧" in invalid[0].reason
        assert "H -4" in invalid[0].reason








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
            "[COMPOSITE:reaction]"
            "[STRUCT:CCO][ARROW:type=single,-H2O][STRUCT:C=C]"
            "[/COMPOSITE]")
        assert len(invalid) == 0

    def test_composite_step_charge_tolerated(self):
        """COMPOSITE 分步保持旁观离子省略惯例：电荷不比对。"""
        pytest.importorskip("rdkit")
        _, invalid = _validate(
            "[COMPOSITE:reaction]"
            "[STRUCT:CCl][PLUS][STRUCT:[OH-]]"
            "[ARROW:type=single][STRUCT:CO][PLUS][STRUCT:[Cl-]]"
            "[/COMPOSITE]")
        assert len(invalid) == 0


def test_composite_mecharrow_bond_form_midpoint_passes(fake_rdkit):
    text = ("[COMPOSITE:reaction][STRUCT:CCl,id=r0][ARROW:type=single]"
            "[STRUCT:CO,id=p0][MECHARROW:r0:0>>r0:0+p0:0][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 0
    text = ("[COMPOSITE:reaction][STRUCT:CCl,id=r0][ARROW:type=single]"
            "[STRUCT:CO,id=p0][MECHARROW:r0:0>>r0:0+p0:0][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 0


def test_composite_mecharrow_midpoint_unknown_id(fake_rdkit):
    text = ("[COMPOSITE:reaction][STRUCT:CCl,id=r0][ARROW:type=single]"
            "[STRUCT:CO,id=p0][MECHARROW:r0:0>r0:0+ghost:0][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "未知组件" in invalid[0].reason


def test_composite_mecharrow_midpoint_atom_out_of_range(fake_rdkit):
    text = ("[COMPOSITE:reaction][STRUCT:CCl,id=r0][ARROW:type=single]"
            "[STRUCT:CO,id=p0][MECHARROW:r0:0>r0:0+p0:9][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "超出范围" in invalid[0].reason


def test_composite_mecharrow_midpoint_bond_mixed_rejected(fake_rdkit):
    text = ("[COMPOSITE:reaction][STRUCT:CCl,id=r0][ARROW:type=single]"
            "[STRUCT:CO,id=p0][MECHARROW:r0:0>r0:0-1+p0:0][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "成键空白位端点格式错误" in invalid[0].reason


class TestMechArrowExplicitH:
    """20260821：显式 H 是真实原子参与编号，MECHARROW 直接写 H 原子序号
    （a#k 语法废弃）。自由基夺氢机理：单碳组分 C([H])([H])([H])[H]
    （0 号 C、1~4 号 H），夺 H 用 ch4:1 引用。
    真实 RDKit（fake_rdkit 白名单不含 C/[Cl]/[CH3]）。
    """

    def test_explicit_h_endpoint_passes(self):
        """ch4:1（显式 H 原子序号）引用 → 放行。"""
        pytest.importorskip("rdkit")
        text = ("[COMPOSITE:reaction]"
                "[STRUCT:[Cl],id=cl,label=Cl·][PLUS]"
                "[STRUCT:C([H])([H])([H])[H],id=ch4,label=CH4]"
                "[ARROW:type=single]"
                "[STRUCT:Cl,id=hcl,label=HCl][PLUS][STRUCT:[CH3],id=me,label=·CH3]"
                "[MECHARROW:cl:0>>cl:0+ch4:0]"
                "[MECHARROW:ch4:1>>cl:0+ch4:0]"
                "[MECHARROW:ch4:1>>me:0]"
                "[/COMPOSITE]")
        _, invalid = _validate(text)
        assert len(invalid) == 0

    def test_explicit_h_bond_break_passes(self):
        """C–H 键断键（ch4:0-1 键中点）→ 放行（键真实存在）。"""
        pytest.importorskip("rdkit")
        text = ("[COMPOSITE:reaction]"
                "[STRUCT:[Cl],id=cl,label=Cl·][PLUS]"
                "[STRUCT:C([H])([H])([H])[H],id=ch4,label=CH4]"
                "[ARROW:type=single]"
                "[STRUCT:Cl,id=hcl,label=HCl][PLUS][STRUCT:[CH3],id=me,label=·CH3]"
                "[MECHARROW:ch4:0-1>>me:0]"
                "[/COMPOSITE]")
        _, invalid = _validate(text)
        assert len(invalid) == 0

    def test_explicit_h_out_of_range_rejected(self):
        """H 原子序号超出范围 → 拦截。"""
        pytest.importorskip("rdkit")
        text = ("[COMPOSITE:reaction]"
                "[STRUCT:[Cl],id=cl,label=Cl·][PLUS]"
                "[STRUCT:C([H])([H])([H])[H],id=ch4,label=CH4]"
                "[ARROW:type=single]"
                "[STRUCT:Cl,id=hcl,label=HCl][PLUS][STRUCT:[CH3],id=me,label=·CH3]"
                "[MECHARROW:ch4:5>>me:0]"
                "[/COMPOSITE]")
        _, invalid = _validate(text)
        assert len(invalid) == 1
        assert "超出范围" in invalid[0].reason

    def test_legacy_a_k_syntax_rejected(self):
        """旧写法 ch4:0#1（a#k 语法，20260821 废弃）→ 格式错误拦截。"""
        pytest.importorskip("rdkit")
        text = ("[COMPOSITE:reaction]"
                "[STRUCT:[Cl],id=cl,label=Cl·][PLUS]"
                "[STRUCT:C([H])([H])([H])[H],id=ch4,label=CH4]"
                "[ARROW:type=single]"
                "[STRUCT:Cl,id=hcl,label=HCl][PLUS][STRUCT:[CH3],id=me,label=·CH3]"
                "[MECHARROW:ch4:0#1>>me:0]"
                "[/COMPOSITE]")
        _, invalid = _validate(text)
        assert len(invalid) == 1
        assert "格式错误" in invalid[0].reason








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
    text = ("[COMPOSITE:reaction][STRUCT:CCO,label=乙醇,id=nu][PLUS]"
            "[STRUCT:CC[OH2+],label=质子化的乙醇,id=pe][ARROW:type=single]"
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
        "[COMPOSITE:reaction][STRUCT:C1C=CC=CC=1,label=苯,id=ar]"
        "[PLUS][STRUCT:[N+](=O)=O,label=NO2+,id=nu]"
        "[ARROW:type=single][STRUCT:O=[N+]([O-])C([H])1C=CC=C[CH+]1,label=σ 络合物,id=sigma]"
        "[MECHARROW:ar:0-5>nu:0][/COMPOSITE]"
        "\n"
        "[COMPOSITE:reaction][STRUCT:O=[N+]([O-])C([H])1C=CC=C[CH+]1,label=σ 络合物,id=sigma]"
        "[ARROW:type=single][STRUCT:O=[N+]([O-])C1=CC=CC=C1,label=硝基苯][PLUS][STRUCT:[H+],label=H+]"
        "[MECHARROW:sigma:4>sigma:3-9][/COMPOSITE]"
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



def test_composite_formula_comp_validation():
    """COMPOSITE 双轨制（20260821）：化学式组件放行；原子级引用/标注拦截。"""
    # 化学式组件放行（守恒通过）
    _, invalid = _validate(
        "[COMPOSITE:reaction][STRUCT:CaO,id=a][PLUS][STRUCT:CO2,id=b]"
        "[ARROW:type=single][STRUCT:CaCO3,id=c][/COMPOSITE]")
    assert len(invalid) == 0
    # 化学式组件参与守恒（不配平拦截）
    _, invalid2 = _validate(
        "[COMPOSITE:reaction][STRUCT:CCO,id=a][STRUCT:KMnO4,id=k]"
        "[ARROW:type=single][STRUCT:CC=O,id=b][/COMPOSITE]")
    assert any("原子不守恒" in r.reason for r in invalid2)
    # MECHARROW 引用化学式组件 → 拦截（无原子可索引）
    _, invalid3 = _validate(
        "[COMPOSITE:reaction][STRUCT:CCl,id=r0][PLUS][STRUCT:[OH-],id=nu]"
        "[ARROW:type=single][STRUCT:CO][PLUS][STRUCT:[Cl-]]"
        "[STRUCT:NaCl,id=s][MECHARROW:nu:0>s:0][/COMPOSITE]")
    assert any("化学式组件" in r.reason for r in invalid3)
    # 化学式组件 mode/bond/charge 标注 → 拦截
    _, invalid4 = _validate(
        "[COMPOSITE:row][STRUCT:KMnO4,id=k,mode=lewis][/COMPOSITE]")
    assert any("化学式组件" in r.reason for r in invalid4)
    _, invalid5 = _validate(
        "[COMPOSITE:row][STRUCT:KMnO4,id=k,charge=0:+][/COMPOSITE]")
    assert any("化学式组件" in r.reason for r in invalid5)


def test_arrow_token_unique_reference():
    """arrow 令牌组件必须被唯一一个 ARROW 的 sup 引用（20260821）。"""
    # 未被引用 → 拦截
    _, invalid = _validate(
        "[COMPOSITE:reaction][STRUCT:CCO,id=a]"
        "[STRUCT:O,id=w,arrow,label=水]"
        "[ARROW:type=single][STRUCT:CC=O,id=b][/COMPOSITE]")
    assert any("未被任何 ARROW" in r.reason for r in invalid)
    # 被两个 ARROW 引用 → 拦截
    _, invalid2 = _validate(
        "[COMPOSITE:reaction][STRUCT:CCO,id=a]"
        "[STRUCT:O,id=w,arrow,label=水]"
        "[ARROW:type=single,sup=+w][STRUCT:CC=O,id=b]"
        "[ARROW:type=single,sup=-w][STRUCT:CC(=O)O,id=c][/COMPOSITE]")
    assert any("被 2 个 ARROW" in r.reason for r in invalid2)
    # 恰好一个引用 → 放行
    _, invalid3 = _validate(
        "[COMPOSITE:reaction][STRUCT:CC(=O)OC,id=e]"
        "[STRUCT:O,id=w,arrow,label=水]"
        "[ARROW:type=single,sup=+w,H+]"
        "[STRUCT:CC(=O)O,id=a][PLUS][STRUCT:CO,id=m][/COMPOSITE]")
    assert len(invalid3) == 0


def test_radical_charge_conflict_rejected():
    """同一原子电荷+自由基拦截（20260820 基线图 32：Br⊖ 还带单电子点）；
    合法情形放行——自由基（[CH3]）、离子（[OH-]）、自由基离子分写
    不同原子（超氧根 [O-][O]）、配合物电荷（FeBr4- 的 [Fe-] 无单电子）。"""
    pytest.importorskip("rdkit")
    _, bad = _validate("[STRUCT:[O-]]")
    assert len(bad) == 1
    assert "自由基单电子" in bad[0].reason
    _, bad2 = _validate("[STRUCT:[NH3+]]")
    assert len(bad2) == 1
    for ok_text in ("[STRUCT:[CH3]]", "[STRUCT:[OH-]]", "[STRUCT:[O-][O]]",
                    "[STRUCT:Br[Fe-](Br)(Br)Br]",
                    "[COMPOSITE:reaction][STRUCT:[CH3],id=a][PLUS][STRUCT:[Cl],id=c]"
                    "[ARROW:type=single][STRUCT:CCl,id=b][/COMPOSITE]"):
        _, invalid = _validate(ok_text)
        assert len(invalid) == 0, f"{ok_text} → {[r.reason for r in invalid]}"
    # 容器内同样拦截
    _, bad3 = _validate(
        "[COMPOSITE:reaction][STRUCT:CCl,id=a][PLUS][STRUCT:[O-],id=b]"
        "[ARROW:type=single][STRUCT:CO,id=c][/COMPOSITE]")
    assert any("自由基单电子" in r.reason for r in bad3)


def test_proton_transfer_pairing():
    """质子转移配对校验（4b，20260820）：
    A. 碱孤对→显式 H 但缺 X—H 键电子回落 → 拦截；
    B. 脱质子（X—H 键电子落回同组件 X）但缺碱夺 H 箭头 → 拦截；
    C. 自由脱质子（容器内有 [H+]）豁免；
    D. 双箭头配对完整 → 放行（示例 9 模式）；
    E. 鱼钩夺氢（自由基）豁免（示例 8 模式）。"""
    pytest.importorskip("rdkit")
    # A：缺配对 → 拦截
    _, bad = _validate(
        "[COMPOSITE:reaction][STRUCT:C([H])C=O,id=ald][PLUS][STRUCT:[OH-],id=base]"
        "[ARROW:type=single][STRUCT:[CH2-]C=O][PLUS][STRUCT:O]"
        "[MECHARROW:base:0>ald:1][/COMPOSITE]")
    assert any("缺配对箭头" in r.reason for r in bad)
    # D：配对完整 → 放行
    _, ok = _validate(
        "[COMPOSITE:reaction][STRUCT:C([H])C=O,id=ald][PLUS][STRUCT:[OH-],id=base]"
        "[ARROW:type=single][STRUCT:[CH2-]C=O][PLUS][STRUCT:O]"
        "[MECHARROW:base:0>ald:1][MECHARROW:ald:0-1>ald:0][/COMPOSITE]")
    assert not ok, [r.reason for r in ok]
    # B：脱质子缺碱箭头 → 拦截
    _, bad2 = _validate(
        "[COMPOSITE:reaction][STRUCT:C([H])C=O,id=ald][PLUS][STRUCT:[OH-],id=base]"
        "[ARROW:type=single][STRUCT:[CH2-]C=O][PLUS][STRUCT:O]"
        "[MECHARROW:ald:0-1>ald:0][/COMPOSITE]")
    assert any("缺配对箭头" in r.reason for r in bad2)
    # C：自由脱质子（写出 [H+]）→ 豁免
    _, ok2 = _validate(
        "[COMPOSITE:reaction][STRUCT:C([H])C=O,id=ald]"
        "[ARROW:type=single][STRUCT:[CH2-]C=O,id=en][PLUS][STRUCT:[H+],id=hp]"
        "[MECHARROW:ald:0-1>ald:0][/COMPOSITE]")
    assert not ok2, [r.reason for r in ok2]
    # E：鱼钩（>>）夺氢豁免 → 放行（示例 8 完整写法）
    _, ok3 = _validate(
        "[COMPOSITE:reaction][STRUCT:[Cl],id=cl,label=Cl·][PLUS]"
        "[STRUCT:C([H]),id=ch4,label=CH4][ARROW:type=single]"
        "[STRUCT:Cl,id=hcl,label=HCl][PLUS][STRUCT:[CH3],id=me,label=·CH3]"
        "[MECHARROW:cl:0>>cl:0+ch4:1]"
        "[MECHARROW:ch4:0-1>>cl:0+ch4:1][MECHARROW:ch4:0-1>>ch4:0][/COMPOSITE]")
    assert not ok3, [r.reason for r in ok3]


def test_sn2_attack_site():
    """SN2 进攻位点校验（4c，20260820）：进攻终点须为连离去基团的 α-C。
    A. 攻 β-C（基线 Q1/Q4 病例：CC[OH2+] 被攻 0 号甲基碳）→ 拦截并建议 α-C；
    B. 攻 α-C → 放行；C. 含多重键组件不查（豁免）；D. 无离去基团不查。"""
    pytest.importorskip("rdkit")
    # A：攻到未连 OH2+ 的 0 号碳 → 拦截，建议 1 号
    _, bad = _validate(
        "[COMPOSITE:reaction][STRUCT:CC[OH2+],id=sub][PLUS][STRUCT:[OH-],id=nu]"
        "[ARROW:type=single][STRUCT:CCO,id=p]"
        "[MECHARROW:nu:0>sub:0][/COMPOSITE]")
    assert any("进攻位点" in r.reason and "sub:1" in r.reason for r in bad)
    # B：攻 α-C（1 号，连 OH2+）→ 放行（质子化乙醇 + 氢氧根 → 乙醇 + 水）
    _, ok = _validate(
        "[COMPOSITE:reaction][STRUCT:CC[OH2+],id=sub][PLUS][STRUCT:[OH-],id=nu]"
        "[ARROW:type=single][STRUCT:CCO,id=p][PLUS][STRUCT:O,id=w]"
        "[MECHARROW:nu:0>sub:1][/COMPOSITE]")
    assert not ok, [r.reason for r in ok]
    # B2：卤素底物攻 α-C（示例 3 SN2）→ 放行
    _, ok2 = _validate(
        "[COMPOSITE:reaction][STRUCT:CCl,id=r0][PLUS][STRUCT:[OH-],id=nu]"
        "[ARROW:type=single][STRUCT:CO][PLUS][STRUCT:[Cl-]]"
        "[MECHARROW:nu:0>r0:0][MECHARROW:r0:0-1>r0:1][/COMPOSITE]")
    assert not ok2, [r.reason for r in ok2]
    # C：终点组件含双键（如 BrC=C）→ 不查（共轭/羰基模式豁免）
    _, ok3 = _validate(
        "[COMPOSITE:reaction][STRUCT:BrC=C,id=v][PLUS][STRUCT:[OH-],id=nu]"
        "[ARROW:type=single][STRUCT:C=CO,id=p][PLUS][STRUCT:[Br-],id=br]"
        "[MECHARROW:nu:0>v:2][/COMPOSITE]")
    assert not ok3, [r.reason for r in ok3]
    # D：终点组件无离去基团（碳正离子被水进攻，SN1 第二步）→ 不查
    _, ok4 = _validate(
        "[COMPOSITE:reaction][STRUCT:C[C+](C)C,id=cat][PLUS][STRUCT:O,id=w]"
        "[ARROW:type=single][STRUCT:CC(C)(C)[OH2+],id=ox]"
        "[MECHARROW:w:0>cat:1][/COMPOSITE]")
    assert not ok4, [r.reason for r in ok4]
