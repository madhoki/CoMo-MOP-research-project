import os

folder_path = 'expt_structures_from_OntoMOPs_KG'
# folder_path = 'zr_tet_mops_xtb_results' 
xyz_files = [f for f in os.listdir(folder_path) if f.endswith('.xyz')]

print(f"📁 Total .xyz files: {len(xyz_files)}")