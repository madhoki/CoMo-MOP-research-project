import pandas as pd
import os

print(os.getcwd())

gaps_df = pd.read_csv(
    'zr_tet_mops_xtb_results/homo_lumo_gaps.csv',
    names=['mop_id', 'homo_lumo_gap_eV']
)

geometry_df = pd.read_csv('FinalData/200_asph_all_results_for_zrTET.csv')

geometry_df['mop_id'] = geometry_df['filename'].str.replace('_opt.xyz', '', regex=False)

merged_df = geometry_df.merge(gaps_df, on='mop_id', how='left')

merged_df = merged_df.drop(columns=['mop_id'])

merged_df.to_csv('ML_model_larger_datasets/data/merged_mop_data.csv', index=False)




