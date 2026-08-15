# -*- coding: utf-8 -*-
"""utils/rdkit_utils.py — RDKit 验证与分子式工具（转型后保留）。"""

import contextlib
import re

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

# 游离氢组分（[H+]/[H]/[H-] 孤立 H，前后是 . 或边界）：合法组分
# （质子/氢自由基/氢负离子），但 RDKit 解析（AdjustHs）与坐标计算
# （RemoveHs）会对孤立 H 打 "not removing hydrogen atom without neighbors"
# 警告（无害，C4）。
FREE_H_COMPONENT_RE = re.compile(r"(?:^|(?<=\.))\[H[+-]?\](?=\.|$)")

# H 数字前缀的化学式习惯写法（非法 SMILES）：[H3O+] / [H2O] / [H3N+]。
# 方括号内以 H 开头、后跟数字 n 与元素符号 X 及可选的电荷（+/-/数字）——
# LLM 把化学式 H₃O⁺ 误写成 SMILES 时，H3 前缀在 SMILES 语法中非法
# （H 原子不能带隐式 H），RDKit 解析失败；规范化重写为 [XHn...]（[OH3+]）。
# 不匹配 H 原子/离子写法（[H]/[H+]/[H-]，无数字）、同位素（[2H]，
# 数字在 H 前）与元素开头写法（[OH3+]/[CH3]/[NH4+]）→ 原样返回。
_H_PREFIX_SMILES_RE = re.compile(r"\[H(\d+)([A-Z][a-z]?)([+-]?\d*)\]")


def normalize_h_prefix_smiles(smiles: str) -> str:
    """规范化"H 数字前缀"SMILES 写法：[H3O+] → [OH3+]、[H2O] → [OH2]、
    [H3N+] → [NH3+]。校验层与渲染层共用（tag_validator._parse_mol /
    renderers.mol_primitives.prepare_mol 解析前调用）；未匹配原样返回。
    """
    if not smiles or not isinstance(smiles, str):
        return smiles
    return _H_PREFIX_SMILES_RE.sub(r"[\2H\1\3]", smiles)


@contextlib.contextmanager
def mute_rdkit_warnings(include_error: bool = False):
    """局部屏蔽 RDKit 日志（进程级开关，仅影响日志输出）。

    include_error=False（默认）：仅屏蔽 rdApp.warning（无害警告，如孤立氢）。
    include_error=True：同时屏蔽 rdApp.error——用于"探测性"解析（如条件 token
    试解析 SMILES，失败是常态），此时解析错误属噪音。
    并发调用时最坏情况是漏掉一条无关日志，可接受。
    """
    RDLogger.DisableLog("rdApp.warning")
    if include_error:
        RDLogger.DisableLog("rdApp.error")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.warning")
        if include_error:
            RDLogger.EnableLog("rdApp.error")


def parse_smiles(smiles: str):
    """Chem.MolFromSmiles 包装：含游离氢组分（合法）时局部屏蔽其无害警告。
    返回 Mol；非法/空返回 None。
    """
    if not smiles or not isinstance(smiles, str):
        return None
    cm = (mute_rdkit_warnings() if FREE_H_COMPONENT_RE.search(smiles)
          else contextlib.nullcontext())
    with cm:
        return Chem.MolFromSmiles(smiles)


def validate_smiles(smiles: str) -> bool:
    """校验 SMILES 是否合法。合法返回 True，否则 False。"""
    return parse_smiles(smiles) is not None


def mol_to_formula(smiles: str) -> str:
    """SMILES -> 分子式（如 'CH4'）；非法或空返回空串。"""
    mol = parse_smiles(smiles)
    if mol is None:
        return ""
    return CalcMolFormula(mol)


if __name__ == "__main__":
    # Day 1 完成标志验证：validate_smiles('C') 返回 True
    print("validate_smiles('C') =", validate_smiles("C"))
    print("validate_smiles('XYZ') =", validate_smiles("XYZ"))
    print("mol_to_formula('c1ccccc1') =", mol_to_formula("c1ccccc1"))
