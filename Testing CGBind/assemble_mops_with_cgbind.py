import pandas as pd
import cgbind
import os
from rdkit import Chem

# Load the CSV database
csv_path = os.path.join('', 'FinalData', 'known_mop_cbu_charges.csv')
df = pd.read_csv(csv_path)

output_dir = 'assembled_mops_xyz'
os.makedirs(output_dir, exist_ok=True)


# Assemble only the first molecule in the database

# Assemble only the first molecule in the database
row = df.iloc[0]
# The 5th column is Organic_CBU (by name or index)
organic_cbu_id = row.iloc[4] if len(row) > 4 else row.get('Organic_CBU')
#arch_name = row.get('Assembly_Model_Label')
arch_name = 'm6l8'
#['m12l24', 'm2l4', 'm4l6', 'm4l6n', 'm4l6t', 'm6l8']

metal = row.get('Metal_CBU')
print(metal)
metal_charge = row.get('Metal_Charge') if 'Metal_Charge' in row else row.iloc[7] if len(row) > 7 else None
ccdc = row.get('CCDC_Number')
mop_name = row.get('MOP')


# Find any MOL file in organic_cbu_geo_files/mol_files that contains the CBU ID
mol_dir = os.path.join('', 'organic_cbu_geo_files', 'mol_files')
mol_file = None
for fname in os.listdir(mol_dir):
    if organic_cbu_id in fname and fname.endswith('.mol'):
        mol_file = os.path.join(mol_dir, fname)
        break
if not mol_file:
    print(f"Skipping {mop_name} (missing MOL file for {organic_cbu_id})")
else:
    try:
        mol = Chem.MolFromMolFile(mol_file)
        if mol is None:
            print(f"Failed to read MOL file for {organic_cbu_id}")
        else:
            smiles = Chem.MolToSmiles(mol)
            print(smiles)
            if not smiles or not arch_name or not metal:
                print(f"Skipping {mop_name} (missing info)")
            else:
                linker = cgbind.Linker(smiles=smiles, arch_name=arch_name)
                cage = cgbind.Cage(linker=linker, metal=metal, metal_charge=metal_charge)
                assembled_filename = f"{ccdc}_{mop_name}.xyz"
                # Create a folder for this structure
                struct_dir = os.path.join(output_dir, str(ccdc))
                os.makedirs(struct_dir, exist_ok=True)
                # Save assembled .xyz
                assembled_filepath = os.path.join(struct_dir, assembled_filename)
                cage.print_xyz_file(filename=assembled_filepath)
                print(f"Assembled {assembled_filename}")
                # Find and copy the twa structure .xyz file
                twa_src = os.path.join('twa_mop_cavity_data', f'{ccdc}.xyz')
                if os.path.isfile(twa_src):
                    twa_dst = os.path.join(struct_dir, f'{ccdc}.xyz')
                    with open(twa_src, 'rb') as fin, open(twa_dst, 'wb') as fout:
                        fout.write(fin.read())
                    print(f"Copied TWA structure {ccdc}.xyz")
                else:
                    print(f"TWA structure {ccdc}.xyz not found.")
    except Exception as e:
        print(f"Failed to assemble {mop_name}: {e}")
