import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
from functions import read_xyz, is_metal, largest_internal_sphere

def get_centroid_and_volume_diffs(folder):
    metal_centroids, all_centroids, filenames, diffs = [], [], [], []
    metal_vols, all_vols, vol_diffs = [], [], []
    metal_radii, all_radii = [], []
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
            # Calculate volumes and radii using largest_internal_sphere
            _, metal_diam, metal_vol = largest_internal_sphere(atoms, coords)
            metal_radius = metal_diam / 2
            # For all-atom centroid, shift centroid and recalc volume and radius
            min_dist_all = np.inf
            for i, atom in enumerate(atoms):
                r = largest_internal_sphere.__globals__["get_radius"](atom)
                d = np.linalg.norm(coords[i]-all_centroid)-r
                if d < min_dist_all:
                    min_dist_all = d
            min_dist_all = max(0, min_dist_all)
            all_radius = min_dist_all
            all_vol = (4/3)*np.pi*all_radius**3
            metal_centroids.append(metal_centroid)
            all_centroids.append(all_centroid)
            filenames.append(filename)
            diffs.append(diff)
            metal_vols.append(metal_vol)
            all_vols.append(all_vol)
            vol_diffs.append(abs(metal_vol - all_vol))
            metal_radii.append(metal_radius)
            all_radii.append(all_radius)
    return filenames, diffs, metal_vols, all_vols, vol_diffs, metal_centroids, all_centroids, metal_radii, all_radii


folder1 = 'expt_structures_from_OntoMOPs_KG'
folder2 = 'twa_mop_cavity_data'
files1, diffs1, metal_vols1, all_vols1, vol_diffs1, metal_centroids1, all_centroids1, metal_radii1, all_radii1 = get_centroid_and_volume_diffs(folder1)
files2, diffs2, metal_vols2, all_vols2, vol_diffs2, metal_centroids2, all_centroids2, metal_radii2, all_radii2 = get_centroid_and_volume_diffs(folder2)


avg_diff1 = np.mean(diffs1)
avg_diff2 = np.mean(diffs2)
avg_vol_diff1 = np.mean(vol_diffs1)
avg_vol_diff2 = np.mean(vol_diffs2)
avg_metal_vol1 = np.mean(metal_vols1)
avg_all_vol1 = np.mean(all_vols1)
avg_metal_vol2 = np.mean(metal_vols2)
avg_all_vol2 = np.mean(all_vols2)
print(f"Average centroid difference (expt_structures): {avg_diff1:.3f} Å")
print(f"Average centroid difference (twa_mop_cavity): {avg_diff2:.3f} Å")
print(f"Average cavity volume difference (expt_structures): {avg_vol_diff1:.3f} Å³")
print(f"Average cavity volume difference (twa_mop_cavity): {avg_vol_diff2:.3f} Å³")
print(f"Average cavity volume (metal centroid, expt_structures): {avg_metal_vol1:.3f} Å³")
print(f"Average cavity volume (all-atom centroid, expt_structures): {avg_all_vol1:.3f} Å³")
print(f"Average cavity volume (metal centroid, twa_mop_cavity): {avg_metal_vol2:.3f} Å³")
print(f"Average cavity volume (all-atom centroid, twa_mop_cavity): {avg_all_vol2:.3f} Å³")

max_diff1 = np.max(diffs1)
max_file1 = files1[np.argmax(diffs1)] if diffs1 else None
max_diff2 = np.max(diffs2)
max_file2 = files2[np.argmax(diffs2)] if diffs2 else None
max_vol_diff1 = np.max(vol_diffs1)
max_vol_file1 = files1[np.argmax(vol_diffs1)] if vol_diffs1 else None
max_vol_diff2 = np.max(vol_diffs2)
max_vol_file2 = files2[np.argmax(vol_diffs2)] if vol_diffs2 else None
print(f"Maximum centroid difference (expt_structures): {max_diff1:.3f} Å in {max_file1}")
print(f"Maximum centroid difference (twa_mop_cavity): {max_diff2:.3f} Å in {max_file2}")
print(f"Maximum cavity volume difference (expt_structures): {max_vol_diff1:.3f} Å³ in {max_vol_file1}")
print(f"Maximum cavity volume difference (twa_mop_cavity): {max_vol_diff2:.3f} Å³ in {max_vol_file2}")

# Print absolute cavity volumes for each structure for context

# Print absolute cavity volumes only for the structures with maximum volume difference
if max_vol_file1:
    idx1 = files1.index(max_vol_file1)
    print(f"\nMax volume diff structure (expt_structures_from_OntoMOPs_KG): {max_vol_file1}")
    print(f"metal centroid = {metal_vols1[idx1]:.3f} Å³, all-atom centroid = {all_vols1[idx1]:.3f} Å³, diff = {vol_diffs1[idx1]:.3f} Å³")
if max_vol_file2:
    idx2 = files2.index(max_vol_file2)
    print(f"\nMax volume diff structure (twa_mop_cavity_data): {max_vol_file2}")
    print(f"metal centroid = {metal_vols2[idx2]:.3f} Å³, all-atom centroid = {all_vols2[idx2]:.3f} Å³, diff = {vol_diffs2[idx2]:.3f} Å³")

fig, axes = plt.subplots(2, 2, figsize=(20, 8), sharex=False)
axes[0,0].bar(range(len(diffs1)), diffs1)
axes[0,0].set_ylabel('Centroid Difference (Å)')
axes[0,0].set_title('expt_structures_from_OntoMOPs_KG - Centroid Diff')
axes[0,0].set_xticks(range(len(files1)))
axes[0,0].set_xticklabels(files1, rotation=90, fontsize=7)
axes[0,1].bar(range(len(vol_diffs1)), vol_diffs1)
axes[0,1].set_ylabel('Cavity Volume Difference (Å³)')
axes[0,1].set_title('expt_structures_from_OntoMOPs_KG - Volume Diff')
axes[0,1].set_xticks(range(len(files1)))
axes[0,1].set_xticklabels(files1, rotation=90, fontsize=7)
axes[1,0].bar(range(len(diffs2)), diffs2)
axes[1,0].set_ylabel('Centroid Difference (Å)')
axes[1,0].set_title('twa_mop_cavity_data - Centroid Diff')
axes[1,0].set_xticks(range(len(files2)))
axes[1,0].set_xticklabels(files2, rotation=90, fontsize=7)
axes[1,1].bar(range(len(vol_diffs2)), vol_diffs2)
axes[1,1].set_ylabel('Cavity Volume Difference (Å³)')
axes[1,1].set_title('twa_mop_cavity_data - Volume Diff')
axes[1,1].set_xticks(range(len(files2)))
axes[1,1].set_xticklabels(files2, rotation=90, fontsize=7)
plt.subplots_adjust(hspace=0.4)
plt.tight_layout()
plt.show()


def plot_centroids_and_spheres(xyz_path, metal_centroid, all_centroid, metal_radius, all_radius):
    atoms, coords = read_xyz(xyz_path)
    atom_colors = {'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red', 'F': 'green', 'Cl': 'green', 'Br': 'brown', 'I': 'purple', 'Cu': 'orange', 'Zn': 'lightblue', 'S': 'yellow', 'P': 'orange', 'Fe': 'darkred', 'Co': 'magenta', 'Ni': 'teal'}
    sizes = [15 for _ in atoms]
    colors = [atom_colors.get(a, 'lightgray') for a in atoms]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], mode='markers', marker=dict(size=sizes, color=colors, opacity=0.7, line=dict(width=0)), text=atoms, name='Atoms', hoverinfo='text'))
    fig.add_trace(go.Scatter3d(x=[metal_centroid[0]], y=[metal_centroid[1]], z=[metal_centroid[2]], mode='markers', marker=dict(size=18, color='red'), name='Metal Centroid'))
    fig.add_trace(go.Scatter3d(x=[all_centroid[0]], y=[all_centroid[1]], z=[all_centroid[2]], mode='markers', marker=dict(size=18, color='blue'), name='All-atom Centroid'))
    # Add spheres for cavity
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    # Metal centroid sphere
    xs = metal_centroid[0] + metal_radius * np.cos(u) * np.sin(v)
    ys = metal_centroid[1] + metal_radius * np.sin(u) * np.sin(v)
    zs = metal_centroid[2] + metal_radius * np.cos(v)
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.3, colorscale=[[0, 'red'], [1, 'red']], showscale=False, name='Metal Cavity'))
    # All-atom centroid sphere
    xs2 = all_centroid[0] + all_radius * np.cos(u) * np.sin(v)
    ys2 = all_centroid[1] + all_radius * np.sin(u) * np.sin(v)
    zs2 = all_centroid[2] + all_radius * np.cos(v)
    fig.add_trace(go.Surface(x=xs2, y=ys2, z=zs2, opacity=0.3, colorscale=[[0, 'blue'], [1, 'blue']], showscale=False, name='All-atom Cavity'))
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), title=f'Centroids & Cavity Spheres for {os.path.basename(xyz_path)}', legend=dict(itemsizing='constant'))
    fig.show()




# Plot cavity spheres for 1586600.xyz in twa_mop_cavity_data (files2)
target_file = '1586600.xyz'
idx = files2.index(target_file)
xyz_path = os.path.join(folder2, target_file)
metal_centroid = metal_centroids2[idx]
all_centroid = all_centroids2[idx]
metal_radius = metal_radii2[idx]
all_radius = all_radii2[idx]
print(4/3 * math.pi * metal_radius**3)
print(4/3 * math.pi * all_radius**3)
print(metal_vols2[idx])
print(all_vols2[idx])
plot_centroids_and_spheres(xyz_path, metal_centroid, all_centroid, metal_radius, all_radius)

# Find structures where all-atom centroid volume > metal centroid volume, and vice versa

# Find pairs of structures where the ranking of cavity volumes flips between methods
print("\nPairs of structures where ranking flips between all-atom and metal centroid methods:")
flip_pairs = []
for i, (fa, va_all, va_metal) in enumerate(zip(files2, all_vols2, metal_vols2)):
    for j, (fb, vb_all, vb_metal) in enumerate(zip(files2, all_vols2, metal_vols2)):
        if i >= j:
            continue  # Avoid duplicate and self-pairs
        # In method 1 (all-atom): fa > fb, in method 2 (metal): fb > fa
        if (va_all > vb_all) and (vb_metal > va_metal):
            flip_pairs.append(((fa, fb), (va_all, vb_all), (va_metal, vb_metal)))
        # Or vice versa
        elif (vb_all > va_all) and (va_metal > vb_metal):
            flip_pairs.append(((fb, fa), (vb_all, va_all), (vb_metal, va_metal)))
if flip_pairs:
    with open("flipping_pairs.txt", "w") as ftxt:
        for (fa, fb), (va_all, vb_all), (va_metal, vb_metal) in flip_pairs:
            out = (
                f"{fa} vs {fb}:\n"
                f"  All-atom centroid: {fa} = {va_all:.3f} Å³, {fb} = {vb_all:.3f} Å³ \n"
                f"  Metal centroid:    {fa} = {va_metal:.3f} Å³, {fb} = {vb_metal:.3f} Å³ \n"
                f"  Ranking flips\n\n"
            )
            print(out)
            ftxt.write(out)
    total_pairs = len(files2) * (len(files2) - 1) // 2
    print(f"Number of flipping pairs: {len(flip_pairs)} / {total_pairs} possible pairings.")
else:
    print("No ranking-flip pairs found.")


