import os
import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors, AllChem

CBU_DIR = 'zr_tet_mops_xtb_results/cbus'
MERGED_INPUT = 'ML_model_larger_datasets/data/merged_mop_data.csv'
MERGED_OUTPUT = 'ML_model_larger_datasets/data/enriched_mop_data.csv'

merged_df = pd.read_csv(MERGED_INPUT)

records = []
mol_files = [f for f in os.listdir(CBU_DIR) if f.endswith('.mol')]
total_files = len(mol_files)
success_count = 0
fail_count = 0





for idx, fname in enumerate(mol_files, 1):
    print(f"[{idx}/{total_files}] Processing: {fname}")
    full_path = os.path.join(CBU_DIR, fname)
    mol = Chem.MolFromMolFile(full_path)
    if mol is None:
        fail_count += 1
        continue
    try:
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol) != 0:
            fail_count += 1
            continue
        AllChem.UFFOptimizeMolecule(mol)
        record = {
            'uuid': re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', fname).group(1),
            'mol_wt': rdMolDescriptors.CalcExactMolWt(mol),
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'num_rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            'num_h_donors': Lipinski.NumHDonors(mol),
            'num_h_acceptors': Lipinski.NumHAcceptors(mol),
            'tpsa': rdMolDescriptors.CalcTPSA(mol),
            'logp': Crippen.MolLogP(mol),
            'heavy_atom_count': Descriptors.HeavyAtomCount(mol),
            'fraction_sp3_carbons': rdMolDescriptors.CalcFractionCSP3(mol)
        }
        records.append(record)
        success_count += 1
    except Exception as e:
        fail_count += 1

print(f"\nCompleted. {success_count} structures processed successfully, {fail_count} failed.")

descriptor_df = pd.DataFrame(records)

def extract_uuid(text):
    match = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', text)
    return match.group(1) if match else None

merged_df['uuid'] = merged_df['filename'].apply(extract_uuid)
enriched_df = pd.merge(merged_df, descriptor_df, on='uuid', how='left')
enriched_df = enriched_df.drop(columns=['uuid'])
enriched_df.to_csv(MERGED_OUTPUT, index=False)
print(f"Enriched dataset saved to: {MERGED_OUTPUT}")
