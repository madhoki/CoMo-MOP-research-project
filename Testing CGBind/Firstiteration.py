from rdkit.Chem import MolFromXYZFile


raw_mol = Chem.MolFromXYZFile('acetate.xyz')
print(raw_mol)
mol = Chem.Mol(raw_mol)



