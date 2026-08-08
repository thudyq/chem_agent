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


@contextlib.contextmanager
def mute_rdkit_warnings():
    """局部屏蔽 RDKit warning 日志（进程级开关，仅影响日志输出）。

    并发调用时最坏情况是漏掉一条无关警告，可接受。
    """
    RDLogger.DisableLog("rdApp.warning")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.warning")


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
