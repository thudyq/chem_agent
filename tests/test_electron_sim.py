# -*- coding: utf-8 -*-
"""tests/test_electron_sim.py — 电子推动模拟器（core/electron_sim.py，P1）测试。

验证机理图的自洽性：弯箭头能否从反应物推出声明产物（Drawbacks §16.2/16.3）。

运行: python -m pytest tests/test_electron_sim.py -v
"""

import pytest

from core.electron_sim import verify_step
from core.tag_parser import parse_tags
from core.tag_validator import _parse_mol, validate_tags


def _run(left, arrows, right):
    """left/right: [(cid, smiles)]；arrows: [spec]。返回原因串。"""
    mech = parse_tags("[COMPOSITE:reaction][STRUCT:C,id=x]"
                      + "".join(f"[MECHARROW:{a}]" for a in arrows)
                      + "[/COMPOSITE]")[0].args[1]
    mech = [c for c in mech if c.type == "MECHARROW"]
    lmols = [(cid, _parse_mol(smi)) for cid, smi in left]
    rmols = [(cid, _parse_mol(smi)) for cid, smi in right]
    reason, _trace = verify_step(lmols, mech, rmols)
    return reason


def _validate_text(text):
    return validate_tags(parse_tags(text))


# ---------- canonical 机理放行（回归锚点：模拟结果正确性） ----------


def test_sn2_passes():
    """示例 3 同款：OH- 进攻 CH3Cl + C—Cl 断裂 → CH3OH + Cl-。"""
    pytest.importorskip("rdkit")
    assert _run([("r0", "CCl"), ("nu", "[OH-]")],
                ["nu:0>r0:0", "r0:0-1>r0:1"],
                [("p0", "CO"), ("p1", "[Cl-]")]) == ""


def test_eas_attack_and_deprotonation_pass():
    """硝化：π 进攻（端位歧义变体）+ 超价补偿；脱质子落回闭环键。"""
    pytest.importorskip("rdkit")
    assert _run([("ar", "C1=CC=CC=C1"), ("nu", "[N+](=O)=O")],
                ["ar:0-1>nu:0", "nu:0-1>nu:1"],
                [("sg", "O=[N+]([O-])C([H])1C=CC=C[CH+]1")]) == ""
    assert _run([("sg", "O=[N+]([O-])C([H])1C=CC=C[CH+]1")],
                ["sg:3-4>sg:3-9"],
                [("nb", "O=[N+]([O-])C1=CC=CC=C1"),
                 ("hp", "[H+]")]) == ""


def test_aldol_deprotonation_passes():
    """示例 9 同款：碱夺 α-H（显式 H 转移）→ 烯醇负离子 + 水。"""
    pytest.importorskip("rdkit")
    assert _run([("ald", "C([H])C=O"), ("base", "[OH-]")],
                ["base:0>ald:1", "ald:0-1>ald:0"],
                [("en", "[CH2-]C=O"), ("w", "O")]) == ""


def test_radical_abstraction_passes():
    """示例 8 同款：Cl· 夺 CH4 的 H（三鱼钩）→ HCl + ·CH3。"""
    pytest.importorskip("rdkit")
    assert _run([("cl", "[Cl]"), ("ch4", "C([H])")],
                ["cl:0>>cl:0+ch4:1", "ch4:0-1>>cl:0+ch4:1",
                 "ch4:0-1>>ch4:0"],
                [("hcl", "Cl"), ("me", "[CH3]")]) == ""


def test_sn1_full_chain_passes():
    """SN1 三步：离去 / 水进攻 / 去质子（实验 C1 修正版写法）。"""
    pytest.importorskip("rdkit")
    assert _run([("r0", "CC(C)(C)Br")], ["r0:1-4>r0:4"],
                [("cat", "C[C+](C)C"), ("br", "[Br-]")]) == ""
    assert _run([("cat", "C[C+](C)C"), ("w", "O")], ["w:0>cat:1"],
                [("ox", "CC(C)(C)[OH2+]")]) == ""
    assert _run([("ox", "CC(C)(C)[O+]([H])[H]"), ("w2", "O")],
                ["w2:0>ox:5", "ox:4-5>ox:4"],
                [("alc", "CC(C)(C)O"), ("hyd", "[OH3+]")]) == ""


def test_hydride_shift_passes():
    """1,2-氢迁移（正丙基 → 异丙基正离子）：H 带电子对搬家。"""
    pytest.importorskip("rdkit")
    assert _run([("r0", "CC([H])[CH2+]")], ["r0:1-2>r0:3"],
                [("p", "C[CH+]C")]) == ""


# ---------- 拦截案例（错侧判定） ----------


def test_q17_wrong_arrows_rejected():
    """que_test_retry Q17 病例：文字/产物对、箭头乱写（落回单原子 +
    从闭环键断键）→ 拦截，且消息默认修箭头（产物已过独立检查）。"""
    pytest.importorskip("rdkit")
    reason = _run([("sigma", "BrC([H])1C=CC=C[CH+]1")],
                  ["sigma:1-2>sigma:1", "sigma:7-1>sigma:7"],
                  [("bp", "BrC1=CC=CC=C1"), ("hp", "[H+]")])
    assert "推不出声明的产物" in reason
    assert "重画机理箭头" in reason          # 错侧判定：信产物修箭头
    assert "改为模拟推得的结构" in reason      # 附模拟预期值（P1.5 翻转提示）


def test_missing_compensation_arrow_impossible():
    """π 进攻缺补偿箭头 → N 五键超价 → 箭头必错（确定的错侧判定：
    结构不合法）。"""
    pytest.importorskip("rdkit")
    reason = _run([("ar", "C1=CC=CC=C1"), ("nu", "[N+](=O)=O")],
                  ["ar:0-1>nu:0"],
                  [("sg", "O=[N+]([O-])C([H])1C=CC=C[CH+]1")])
    assert "不成立" in reason and "重画机理箭头" in reason


def test_h_atom_as_fishhook_source_rejected():
    """鱼钩起点是 H 原子（无单电子可给）→ 箭头必错。"""
    pytest.importorskip("rdkit")
    reason = _run([("cl", "[Cl]"), ("ch4", "C([H])")],
                  ["ch4:1>>cl:0+ch4:1"],
                  [("hcl", "Cl"), ("me", "[CH3]")])
    assert "不成立" in reason


# ---------- 校验器集成与跳过面 ----------


def test_validator_integration_rejects_q17():
    """完整校验路径：Q17 病例被拦（EAS 方向规则先开枪；模拟器是深层
    兜底，两规则给出同一正确建议 sigma:1-2>sigma:1-7）。"""
    pytest.importorskip("rdkit")
    _, bad = _validate_text(
        "[COMPOSITE:reaction]"
        "[STRUCT:BrC([H])1C=CC=C[CH+]1,label=σ 络合物,id=sigma]"
        "[ARROW:type=single]"
        "[STRUCT:BrC1=CC=CC=C1,label=溴苯][PLUS][STRUCT:[H+],label=H+]"
        "[MECHARROW:sigma:1-2>sigma:1][MECHARROW:sigma:7-1>sigma:7]"
        "[/COMPOSITE]")
    assert len(bad) == 1 and "sigma:1-7" in bad[0].reason


def test_validator_integration_simulator_backstop():
    """纯模拟器拦截案例：均裂两个电子都归同一原子（各模式规则均不查，
    模拟推不出两个 Cl·）——验证模拟器作为兜底生效。"""
    pytest.importorskip("rdkit")
    _, bad = _validate_text(
        "[COMPOSITE:reaction]"
        "[STRUCT:ClCl,label=Cl2,id=cl2]"
        "[ARROW:type=single,hν]"
        "[STRUCT:[Cl],label=Cl·,id=cl1][PLUS][STRUCT:[Cl],label=Cl·,id=cl2b]"
        "[MECHARROW:cl2:0-1>>cl2:0][MECHARROW:cl2:0-1>>cl2:0]"
        "[/COMPOSITE]")
    assert len(bad) == 1 and "电子流模拟" in bad[0].reason


def test_validator_integration_passes_canonical_nitration():
    """完整校验路径：硝化三步（示例 11）全部放行（含 π 进攻歧义变体）。"""
    pytest.importorskip("rdkit")
    _, bad = _validate_text(
        "[COMPOSITE:reaction]"
        "[STRUCT:C1=CC=CC=C1,label=苯,id=ar][PLUS]"
        "[STRUCT:[N+](=O)=O,label=NO2+,id=nu]"
        "[ARROW:type=single]"
        "[STRUCT:O=[N+]([O-])C([H])1C=CC=C[CH+]1,label=σ 络合物,id=sigma]"
        "[MECHARROW:ar:0-1>nu:0][MECHARROW:nu:0-1>nu:1]"
        "[/COMPOSITE]")
    assert not bad, [r.reason for r in bad]


def test_skips_no_arrow_and_formula_components():
    """跳过面：无箭头步不模拟；化学式文本组件（KMnO4）参与的步跳过。"""
    pytest.importorskip("rdkit")
    _, bad = _validate_text(
        "[COMPOSITE:reaction]"
        "[STRUCT:CCO,label=乙醇][PLUS][STRUCT:KMnO4,label=高锰酸钾]"
        "[ARROW:type=single][STRUCT:CC(=O)O,label=乙酸][/COMPOSITE]")
    # 无机理箭头——守恒可能拦（本例不守恒会被拦，但绝不会因模拟器拦）
    assert not any("电子流模拟" in r.reason for r in bad)


def test_impossible_message_uses_local_refs():
    """超价等"不可能"报错用 组件:局部序号（nu:0），LLM 可直接定位——
    不用大图全局序号或 RDKit 碎片内序号（P2 修正 prompt 依赖可读定位）。"""
    pytest.importorskip("rdkit")
    reason = _run([("ar", "C1=CC=CC=C1"), ("nu", "[N+](=O)=O")],
                  ["ar:0-1>nu:0"],
                  [("sg", "O=[N+]([O-])C([H])1C=CC=C[CH+]1")])
    assert "不成立" in reason
    assert "nu:0" in reason            # 组件:局部序号定位（N 五键超价）
    assert "atom #" not in reason      # 不裸用 RDKit 碎片内序号
