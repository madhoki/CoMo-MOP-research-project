import os
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem import GetPeriodicTable
import plotly.graph_objects as go

PERIODIC_TABLE = GetPeriodicTable()
folder = 'expt_structures_from_OntoMOPs_KG'  # or 'twa_mop_cavity_data'

# Helper functions
def read_xyz(filepath):
    atoms, coords = [], []
    with open(filepath) as f:
        for line in f.readlines()[2:]:
            p = line.strip().split()
            if len(p) >= 4:
                atoms.append(p[0])
                coords.append([float(x) for x in p[1:4]])
    return np.array(atoms), np.array(coords)

def is_metal(symbol):
    Z = PERIODIC_TABLE.GetAtomicNumber(symbol)
    if Z in {3, 11, 19, 37, 55, 87, 4, 12, 20, 38, 56, 88, 13, 31, 49, 50, 81, 82, 83, 113, 114, 115, 116}: return True
    if 21 <= Z <= 30 or 39 <= Z <= 48 or 72 <= Z <= 80 or 104 <= Z <= 112: return True
    if 57 <= Z <= 71 or 89 <= Z <= 103: return True
    return False

def get_centroid_diffs(folder):
    metal_centroids, all_centroids, filenames, diffs = [], [], [], []
    for filename in os.listdir(folder):
        if filename.endswith('.xyz'):
            filepath = os.path.join(folder, filename)
            atoms, coords = read_xyz(filepath)
            metal_idx = [i for i, a in enumerate(atoms) if is_metal(a)]
            if metal_idx:
                metal_centroid = np.mean(coords[metal_idx], axis=0)
            else:
                metal_centroid = np.mean(coords, axis=0)
            all_centroid = np.mean(coords, axis=0)
            diff = np.linalg.norm(metal_centroid - all_centroid)
            metal_centroids.append(metal_centroid)
            all_centroids.append(all_centroid)
            filenames.append(filename)
            diffs.append(diff)
    return filenames, diffs

folder1 = 'expt_structures_from_OntoMOPs_KG'
folder2 = 'twa_mop_cavity_data'
files1, diffs1 = get_centroid_diffs(folder1)
files2, diffs2 = get_centroid_diffs(folder2)

avg_diff1 = np.mean(diffs1)
avg_diff2 = np.mean(diffs2)
print(f"Average centroid difference (expt_structures): {avg_diff1:.3f} Å")
print(f"Average centroid difference (twa_mop_cavity): {avg_diff2:.3f} Å")

max_diff1 = np.max(diffs1)
max_file1 = files1[np.argmax(diffs1)] if diffs1 else None
max_diff2 = np.max(diffs2)
max_file2 = files2[np.argmax(diffs2)] if diffs2 else None
print(f"Maximum centroid difference (expt_structures): {max_diff1:.3f} Å in {max_file1}")
print(f"Maximum centroid difference (twa_mop_cavity): {max_diff2:.3f} Å in {max_file2}")

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
axes[0].bar(range(len(diffs1)), diffs1)

axes[0].set_ylabel('Centroid Difference (Angstrom)')
axes[0].set_title('expt_structures_from_OntoMOPs_KG')
axes[0].set_xticks(range(len(files1)))
axes[0].set_xticklabels(files1, rotation=90, fontsize=7)
axes[1].bar(range(len(diffs2)), diffs2)

axes[1].set_ylabel('Centroid Difference (Angstrom)')
axes[1].set_title('twa_mop_cavity_data')
axes[1].set_xticks(range(len(files2)))
axes[1].set_xticklabels(files2, rotation=90, fontsize=7)
plt.subplots_adjust(hspace=0.4)
plt.tight_layout()
plt.show()

def plot_centroids_xyz(xyz_path, metal_centroid, all_centroid):
    atoms, coords = read_xyz(xyz_path)
    atom_colors = {'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red', 'F': 'green', 'Cl': 'green', 'Br': 'brown', 'I': 'purple', 'Cu': 'orange', 'Zn': 'lightblue', 'S': 'yellow', 'P': 'orange', 'Fe': 'darkred', 'Co': 'magenta', 'Ni': 'teal'}
    sizes = [15 for _ in atoms]
    colors = [atom_colors.get(a, 'lightgray') for a in atoms]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], mode='markers', marker=dict(size=sizes, color=colors, opacity=0.7, line=dict(width=0)), text=atoms, name='Atoms', hoverinfo='text'))
    fig.add_trace(go.Scatter3d(x=[metal_centroid[0]], y=[metal_centroid[1]], z=[metal_centroid[2]], mode='markers', marker=dict(size=18, color='red'), name='Metal Centroid'))
    fig.add_trace(go.Scatter3d(x=[all_centroid[0]], y=[all_centroid[1]], z=[all_centroid[2]], mode='markers', marker=dict(size=18, color='blue'), name='All-atom Centroid'))
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), title=f'Centroids for {os.path.basename(xyz_path)}', legend=dict(itemsizing='constant'))
    fig.show()

# Plot for expt_structures_from_OntoMOPs_KG
if max_file1:
    xyz_path1 = os.path.join(folder1, max_file1)
    atoms1, coords1 = read_xyz(xyz_path1)
    metal_idx1 = [i for i, a in enumerate(atoms1) if is_metal(a)]
    metal_centroid1 = np.mean(coords1[metal_idx1], axis=0) if metal_idx1 else np.mean(coords1, axis=0)
    all_centroid1 = np.mean(coords1, axis=0)
    plot_centroids_xyz(xyz_path1, metal_centroid1, all_centroid1)
# Plot for twa_mop_cavity_data
if max_file2:
    xyz_path2 = os.path.join(folder2, max_file2)
    atoms2, coords2 = read_xyz(xyz_path2)
    metal_idx2 = [i for i, a in enumerate(atoms2) if is_metal(a)]
    metal_centroid2 = np.mean(coords2[metal_idx2], axis=0) if metal_idx2 else np.mean(coords2, axis=0)
    all_centroid2 = np.mean(coords2, axis=0)
    plot_centroids_xyz(xyz_path2, metal_centroid2, all_centroid2)


