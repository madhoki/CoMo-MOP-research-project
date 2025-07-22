import os
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix

# === Set your folder path ===
folder_path = 'expt_structures_from_OntoMOPs_KG'
data = []

# === Helper: Read .xyz file ===
def read_xyz(filepath):
    atoms = []
    coords = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:  # skip header lines
            parts = line.strip().split()
            if len(parts) >= 4:
                atom = parts[0]
                x, y, z = map(float, parts[1:4])
                atoms.append(atom)
                coords.append([x, y, z])
    return np.array(atoms), np.array(coords)

# === Main loop ===
for filename in os.listdir(folder_path):
    if filename.endswith('.xyz'):
        filepath = os.path.join(folder_path, filename)
        try:
            atoms, coords = read_xyz(filepath)

            # Identify Pd atoms
            pd_indices = np.where(atoms == 'Pd')[0]
            pd_coords = coords[pd_indices]

            # Compute Pd–Pd distances
            pd_dists = []
            for i in range(len(pd_coords)):
                for j in range(i + 1, len(pd_coords)):
                    dist = np.linalg.norm(pd_coords[i] - pd_coords[j])
                    pd_dists.append(dist)

            # Compute escape sphere radius: max distance from centroid to any atom
            centroid = np.mean(coords, axis=0)
            escape_radii = np.linalg.norm(coords - centroid, axis=1)
            escape_radius = np.max(escape_radii)
            escape_volume = (4/3) * np.pi * escape_radius**3

            # Cavity volume estimate using convex hull
            try:
                hull = ConvexHull(coords)
                cavity_volume = hull.volume
            except:
                cavity_volume = np.nan

            data.append({
                'filename': filename,
                'num_Pd': len(pd_coords),
                'min_Pd–Pd_dist (Å)': round(min(pd_dists), 2) if pd_dists else None,
                'max_Pd–Pd_dist (Å)': round(max(pd_dists), 2) if pd_dists else None,
                'cavity_volume (Å³)': round(cavity_volume, 2),
                'escape_radius (Å)': round(escape_radius, 2),
                'escape_volume (Å³)': round(escape_volume, 2)
            })

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

# === Create and show DataFrame ===
df = pd.DataFrame(data)
print(df)

df.to_excel("mop_properties_full.xlsx", index=False)

