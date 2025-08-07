# import pandas as pd
# import cgbind
# import os
# from rdkit import Chem

# # Load the CSV database
# csv_path = os.path.join('', 'FinalData', 'known_mop_cbu_charges.csv')
# df = pd.read_csv(csv_path)

# output_dir = 'assembled_mops_xyz'
# os.makedirs(output_dir, exist_ok=True)


# # Assemble only the first molecule in the database

# # Assemble only the first molecule in the database
# row = df.iloc[0]
# # The 5th column is Organic_CBU (by name or index)
# organic_cbu_id = row.iloc[4]
# #arch_name = row.get('Assembly_Model_Label')
# arch_name = 'm6l8'
# #['m12l24', 'm2l4', 'm4l6', 'm4l6n', 'm4l6t', 'm6l8']

# metal = row.get('Metal_CBU')
# metal_cbu_id = row.iloc[6]

# print(metal)
# metal_charge = row.get('Metal_Charge') if 'Metal_Charge' in row else row.iloc[7] if len(row) > 7 else None
# ccdc = row.get('CCDC_Number')
# mop_name = row.get('MOP')


# # Find any MOL file in organic_cbu_geo_files/mol_files that contains the CBU ID
# mol_dir = os.path.join('', 'organic_cbu_geo_files', 'mol_files')
# mol_file = None
# for fname in os.listdir(mol_dir):
#     if organic_cbu_id in fname and fname.endswith('.mol'):
#         mol_file = os.path.join(mol_dir, fname)
#         break
# if not mol_file:
#     print(f"Skipping {mop_name} (missing MOL file for {organic_cbu_id})")
# else:
#     try:
#         mol = Chem.MolFromMolFile(mol_file)
#         if mol is None:
#             print(f"Failed to read MOL file for {organic_cbu_id}")
#         else:
#             smiles = Chem.MolToSmiles(mol)
#             print(smiles)
#             if not smiles or not arch_name or not metal:
#                 print(f"Skipping {mop_name} (missing info)")
#             else:
#                 linker = cgbind.Linker(smiles=smiles, arch_name=arch_name)
#                 cage = cgbind.Cage(linker=linker, metal=metal, metal_charge=metal_charge)
#                 assembled_filename = f"{ccdc}_{mop_name}.xyz"
#                 # Create a folder for this structure
#                 struct_dir = os.path.join(output_dir, str(ccdc))
#                 os.makedirs(struct_dir, exist_ok=True)
#                 # Save assembled .xyz
#                 assembled_filepath = os.path.join(struct_dir, assembled_filename)
#                 cage.print_xyz_file(filename=assembled_filepath)
#                 print(f"Assembled {assembled_filename}")
#                 # Find and copy the twa structure .xyz file
#                 twa_src = os.path.join('twa_mop_cavity_data', f'{ccdc}.xyz')
#                 if os.path.isfile(twa_src):
#                     twa_dst = os.path.join(struct_dir, f'{ccdc}.xyz')
#                     with open(twa_src, 'rb') as fin, open(twa_dst, 'wb') as fout:
#                         fout.write(fin.read())
#                     print(f"Copied TWA structure {ccdc}.xyz")
#                 else:
#                     print(f"TWA structure {ccdc}.xyz not found.")
#     except Exception as e:
#         print(f"Failed to assemble {mop_name}: {e}")

# metal_dir = os.path.join('', 'metal_cbu_geo_files', '')
# for fname in os.listdir(metal_dir):
#     if fname.endswith('.xyz') and metal_cbu_id in fname:
#         molecule = Chem.MolFromXYZFile(os.path.join(metal_dir, fname))
#         smiles = Chem.MolToSmiles(mol)

#         print(smiles)
#         if molecule is None:
#             print(f"Failed to read XYZ file {fname}")


import os
import pandas as pd
from rdkit import Chem
import cgbind

csv_path = os.path.join('FinalData', 'known_mop_cbu_charges.csv')
mol_dir = os.path.join('organic_cbu_geo_files', 'mol_files')
metal_dir = os.path.join('metal_cbu_geo_files')
twa_dir = 'FinalData/twa_mop_cavity_data'
output_dir = 'assembled_mops_xyz'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
row = df.iloc[0]


organic_cbu_id = row.iloc[4]
arch_name = 'm6l8'  # could also be row.get('Assembly_Model_Label')
metal_name = row.get('Metal_CBU')
metal_cbu_id = row.iloc[6]
metal_charge = row.get('Metal_Charge') or (row.iloc[7] if len(row) > 7 else None)
ccdc = row.get('CCDC_Number')
mop_name = row.get('MOP')

print(f"Processing MOP: {mop_name} | Metal: {metal_name} | Linker ID: {organic_cbu_id}")

mol_file = next(
    (os.path.join(mol_dir, f) for f in os.listdir(mol_dir) if organic_cbu_id in f and f.endswith('.mol')),
    None
)
if not mol_file:
    print(f"Skipping {mop_name} (no MOL file found for {organic_cbu_id})")
    exit()

mol = Chem.MolFromMolFile(mol_file)
if mol is None:
    print(f"Failed to read MOL file: {mol_file}")
    exit()

linker_smiles = Chem.MolToSmiles(mol)
print(f"Linker SMILES: {linker_smiles}")

core_xyz_path = next(
    (os.path.join(metal_dir, f) for f in os.listdir(metal_dir) if metal_cbu_id in f and f.endswith('.xyz')),
    None
)
if not core_xyz_path:
    print(f"Skipping {mop_name} (no XYZ file found for metal core {metal_cbu_id})")
    exit()

print(f"Using metal core file: {core_xyz_path}")

core_mol = Chem.MolFromXYZFile(core_xyz_path)
if core_mol:
    core_smiles = Chem.MolToSmiles(core_mol)
    print(f"Core SMILES (optional): {core_smiles}")
else:
    print(f"Warning: Failed to parse core XYZ into RDKit Mol object (proceeding anyway)")

try:
    linker = cgbind.Linker(smiles=linker_smiles, arch_name=arch_name)
    cage = cgbind.Cage(core=core_xyz_path, linker=linker, metal=metal_name, metal_charge=metal_charge)

    struct_dir = os.path.join(output_dir, str(ccdc))
    os.makedirs(struct_dir, exist_ok=True)
    xyz_filename = f"{ccdc}_{mop_name}.xyz"
    xyz_path = os.path.join(struct_dir, xyz_filename)
    cage.print_xyz_file(filename=xyz_path)
    print(f"Saved assembled MOP: {xyz_filename}")

    twa_src = os.path.join(twa_dir, f"{ccdc}.xyz")
    twa_dst = os.path.join(struct_dir, f"{ccdc}.xyz")
    if os.path.isfile(twa_src):
        with open(twa_src, 'rb') as fin, open(twa_dst, 'wb') as fout:
            fout.write(fin.read())
        print(f"Copied TWA structure to: {twa_dst}")
    else:
        print(f"TWA structure not found for {ccdc}")

except Exception as e:
    print(f"Failed to assemble MOP {mop_name}: {e}")
