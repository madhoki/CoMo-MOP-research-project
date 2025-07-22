import os

folder_path = 'expt_structures_from_OntoMOPs_KG'
xyz_files = [f for f in os.listdir(folder_path) if f.endswith('.xyz')]

print(f"📁 Total .xyz files: {len(xyz_files)}")