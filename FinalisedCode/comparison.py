import pandas as pd
import matplotlib.pyplot as plt
import os

calc_csv = 'all_results_for_twa.csv'
#calc_csv = 'all_results_for_ONTOMOPKG.csv'
cambridge_csv = 'twa_mop_cavity_data/data.csv'

calc_df = pd.read_csv(calc_csv)
calc_df['id'] = calc_df['filename'].str.replace('.xyz','', regex=False)

cambridge_df = pd.read_csv(cambridge_csv)
cambridge_df['id'] = cambridge_df['csd_number'].astype(str)

merged = pd.merge(calc_df, cambridge_df, on='id', suffixes=('_calc','_cambridge'))

pairs = [
    ('inner_diameter', 'inner_sphere_diameter_Angstrom'),
    ('max_pore_size_diameter', 'pore_size_diameter_Angstrom')
]

for calc_col, cambridge_col in pairs:
    plt.figure(figsize=(8,5))
    plt.scatter(merged[cambridge_col], merged[calc_col], alpha=0.7)
    for i, row in merged.iterrows():
        plt.text(row[cambridge_col], row[calc_col], row['filename'], fontsize=6, alpha=0.6)
    plt.plot([merged[cambridge_col].min(), merged[cambridge_col].max()], [merged[cambridge_col].min(), merged[cambridge_col].max()], 'r--')
    plt.xlabel(f'Cambridge: {cambridge_col}')
    plt.ylabel(f'Calculated: {calc_col}')
    plt.title(f'Parity plot: {calc_col} vs {cambridge_col}')
    plt.tight_layout()
    plt.show()
