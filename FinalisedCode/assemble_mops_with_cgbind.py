import pandas as pd
import cgbind
import os

# Load the CSV database
csv_path = os.path.join('..', 'FinalData', 'known_mop_cbu_charges.csv')
df = pd.read_csv(csv_path)

output_dir = 'assembled_mops_xyz'
os.makedirs(output_dir, exist_ok=True)

for idx, row in df.iterrows():
    smiles = row.get('Organic_CBU')  # You may need to map this to actual SMILES
    arch_name = row.get('Assembly_Model_Label')  # e.g., 'm2l4', may need mapping
    metal = row.get('Metal_CBU')  # You may need to map this to element symbol
    ccdc = row.get('CCDC_Number')
    mop_name = row.get('MOP')

    # Example: If you have a SMILES string for the linker, use it here
    # If not, you need to provide a mapping from CBU to SMILES
    if not smiles or not arch_name or not metal:
        print(f"Skipping {mop_name} (missing info)")
        continue

    try:
        linker = cgbind.Linker(smiles=smiles, arch_name=arch_name)
        cage = cgbind.Cage(linker=linker, metal=metal)
        filename = f"{ccdc}_{mop_name}.xyz"
        filepath = os.path.join(output_dir, filename)
        cage.print_xyz_file(filename=filepath)
        print(f"Assembled {filename}")
    except Exception as e:
        print(f"Failed to assemble {mop_name}: {e}")
