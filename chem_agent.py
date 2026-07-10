# -*- coding: utf-8 -*-
"""
chem_agent.py
=============
有机化学知识智能体 - 主入口脚本。

Day 1 任务：环境验证与 RDKit 基础
    - 实现 validate_mol(smiles)：使用 rdkit.Chem.MolFromSmiles 校验 SMILES 合法性。
    - 提供 get_molecular_formula(mol)：基于 RDKit 计算分子式。
    - 在 __main__ 中对硬编码 'C1CCC1'（环丁烷）进行验证并打印分子式。
"""

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


def validate_mol(smiles: str):
    """校验 SMILES 字符串是否合法。

    参数:
        smiles (str): 待校验的 SMILES 字符串，例如 'C1CCC1'。

    返回:
        rdkit.Chem.rdchem.Mol | None:
            - 合法时返回 RDKit 的 Mol 对象（便于下游继续使用）。
            - 非法或解析失败时返回 None。
    """
    print(f"[validate_mol] 正在校验 SMILES: {smiles!r}")
    if not smiles or not isinstance(smiles, str):
        print("[validate_mol] 输入为空或类型错误，校验失败。")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # MolFromSmiles 对无法解析的输入会返回 None
        print(f"[validate_mol] 解析失败：'{smiles}' 不是合法的 SMILES。")
        return None

    print(f"[validate_mol] 校验通过，共解析到 {mol.GetNumAtoms()} 个重原子。")
    return mol


def get_molecular_formula(mol) -> str:
    """根据 RDKit Mol 对象计算分子式（如 'C4H8'）。

    参数:
        mol: rdkit.Chem.rdchem.Mol 对象（由 validate_mol 返回）。

    返回:
        str: 分子式字符串；mol 为 None 时返回空串。
    """
    if mol is None:
        print("[get_molecular_formula] 输入的 Mol 对象为空，无法计算分子式。")
        return ""
    formula = CalcMolFormula(mol)
    mol_weight = Descriptors.MolWt(mol)
    print(f"[get_molecular_formula] 分子式={formula}，分子量≈{mol_weight:.2f}")
    return formula


def main():
    """Day 1 测试入口：验证硬编码 SMILES 'C1CCC1' 并打印分子式。"""
    print("=" * 60)
    print("Day 1 - 环境验证与 RDKit 基础测试")
    print("=" * 60)

    # 硬编码测试用例：环丁烷 C1CCC1（应解析为 C4H8）
    test_smiles = "C1CCC1"
    mol = validate_mol(test_smiles)

    if mol is not None:
        formula = get_molecular_formula(mol)
        print(f"\n[结果] SMILES '{test_smiles}' 校验通过 -> 分子式: {formula}")
    else:
        print(f"\n[结果] SMILES '{test_smiles}' 校验失败，请检查输入。")

    print("=" * 60)
    print("Day 1 测试结束。")
    print("=" * 60)


if __name__ == "__main__":
    # 直接运行本文件即可执行 Day 1 的硬编码验证测试。
    # 用法: python chem_agent.py
    main()
