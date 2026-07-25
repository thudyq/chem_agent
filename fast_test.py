from legacy.name_resolver import name_to_smiles

smiles = name_to_smiles("aspirin")      # 先 PubChem REST，失败回退 pyopsin
print(smiles)