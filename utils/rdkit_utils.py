# -*- coding: utf-8 -*-
"""utils/rdkit_utils.py — RDKit 验证与分子式工具（转型后保留）。"""

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


def validate_smiles(smiles: str) -> bool:
    """校验 SMILES 是否合法。合法返回 True，否则 False。"""
    if not smiles or not isinstance(smiles, str):
        return False
    return Chem.MolFromSmiles(smiles) is not None


def mol_to_formula(smiles: str) -> str:
    """SMILES -> 分子式（如 'CH4'）；非法或空返回空串。"""
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return CalcMolFormula(mol)


if __name__ == "__main__":
    # Day 1 完成标志验证：validate_smiles('C') 返回 True
    print("validate_smiles('C') =", validate_smiles("C"))
    print("validate_smiles('XYZ') =", validate_smiles("XYZ"))
    print("mol_to_formula('c1ccccc1') =", mol_to_formula("c1ccccc1"))
