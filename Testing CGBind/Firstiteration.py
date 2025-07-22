from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from rdkit.Chem import MolFromXYZFile


raw_mol = Chem.MolFromXYZFile('acetate.xyz')
mol = Chem.Mol(raw_mol)
rdDetermineBonds.DetermineBonds(mol,charge=-1)



