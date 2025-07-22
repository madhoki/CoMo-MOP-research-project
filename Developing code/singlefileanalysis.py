import numpy as np
from scipy.spatial import ConvexHull
import math

# === Step 1: Read the XYZ file ===
def read_xyz(filename):
    atoms = []
    coords = []
    with open(filename, 'r') as f:
        lines = f.readlines()[2:]  # skip first two lines
        for line in lines:
            parts = line.strip().split()
            atoms.append(parts[0])
            coords.append(list(map(float, parts[1:4])))
    return np.array(atoms), np.array(coords)

atoms, coords = read_xyz("samplestructure.xyz")

# === Step 2: Identify metal atoms (Pd) ===
pd_coords = coords[atoms == 'Pd']

# === Step 3: Compute Pd–Pd distances ===
pd_dists = [
    np.linalg.norm(pd_coords[i] - pd_coords[j])
    for i in range(len(pd_coords))
    for j in range(i + 1, len(pd_coords))
]

min_pd = min(pd_dists)
max_pd = max(pd_dists)

# === Step 4: Estimate cavity volume using Convex Hull ===
try:
    hull = ConvexHull(coords)
    cavity_volume = hull.volume
except:
    cavity_volume = np.nan

# === Step 5: Escape sphere = max distance from centroid ===
centroid = np.mean(coords, axis=0)
escape_radius = np.max(np.linalg.norm(coords - centroid, axis=1))
escape_volume = (4 / 3) * math.pi * escape_radius**3

# === Step 6: Print results ===
print(f"✅ Loaded XYZ with {len(coords)} atoms")
print(f"Pd–Pd Distance (min–max): {min_pd:.2f} Å – {max_pd:.2f} Å")
print(f"Cavity Volume (Convex Hull): {cavity_volume:.2f} Å³")
print(f"Max Escape Sphere Radius: {escape_radius:.2f} Å")
print(f"Max Escape Sphere Volume: {escape_volume:.2f} Å³")
