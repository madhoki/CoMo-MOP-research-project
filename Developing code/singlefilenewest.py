import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.optimize import basinhopping

# === CONFIG ===
file_path = 'first.xyz'

# === VAN DER WAALS RADII (Å) ===
vDW_radii = {
    'H': 1.20, 'He': 1.40, 'Li': 1.82, 'C': 1.70, 'N': 1.55, 'O': 1.52,
    'F': 1.47, 'Ne': 1.54, 'Na': 2.27, 'Mg': 1.73, 'P': 1.80, 'S': 1.80,
    'Cl': 1.75, 'Ar': 1.88, 'K': 2.75, 'Ni': 1.63, 'Cu': 1.40, 'Zn': 1.39,
    'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02, 'Pd': 1.63, 'Ag': 1.72,
    'Cd': 1.58, 'In': 1.93, 'Sn': 2.17, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16,
    'Pt': 1.75, 'Au': 1.66, 'Hg': 1.55, 'Tl': 1.96, 'Pb': 2.02, 'U': 1.86
}

# === Functions ===

def read_xyz(filepath):
    atoms, coords = [], []
    with open(filepath, 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                atoms.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])
    return np.array(atoms), np.array(coords)

def get_pd_distances(atoms, coords):
    pd_mask = np.array(atoms) == 'Pd'
    pd_coords = coords[pd_mask]
    dists = [np.linalg.norm(a - b) for i, a in enumerate(pd_coords) for b in pd_coords[i+1:]]
    return dists

def get_max_enclosed_sphere(atoms, coords, vdw_radii, centroid=None, resolution=0.001):
    if centroid is None:
        metal_coords = coords[np.isin(atoms, ['Pd', 'Pt', 'Fe'])]
        centroid = metal_coords.mean(axis=0)

    for r in np.linspace(20, 0.1, int(20 / resolution)):
        intersects = any(
            np.linalg.norm(coord - centroid) < r + vdw_radii.get(atom, 1.7)
            for atom, coord in zip(atoms, coords)
        )
        if not intersects:
            volume = (4 / 3) * np.pi * r**3
            return r, volume
    return 0.0, 0.0

def escape_radius_at_direction(direction, centroid, tree, radii, max_distance=20.0, resolution=0.05):
    direction = direction / np.linalg.norm(direction)
    escape_radius = float('inf')
    for r in np.arange(0, max_distance, resolution):
        point = centroid + r * direction
        dists, idxs = tree.query(point, k=10, distance_upper_bound=8.0)
        for dist, idx in zip(dists, idxs):
            if idx >= len(radii): continue
            clearance = dist - radii[idx]
            if clearance < 0:
                return 0
            escape_radius = min(escape_radius, clearance)
    return escape_radius

def escape_objective(direction, centroid, tree, radii):
    return -escape_radius_at_direction(direction, centroid, tree, radii)

def get_escape_radius(coords, atoms, vdw_radii):
    metal_indices = [i for i, a in enumerate(atoms) if a in ['Pd', 'Pt', 'Fe']]
    centroid = np.mean(coords[metal_indices], axis=0)

    radii = np.array([vdw_radii.get(atom, 1.7) for atom in atoms])
    tree = KDTree(coords)

    initial_direction = np.random.randn(3)
    initial_direction /= np.linalg.norm(initial_direction)

    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "args": (centroid, tree, radii)
    }

    result = basinhopping(
        escape_objective,
        initial_direction,
        minimizer_kwargs=minimizer_kwargs,
        niter=20,
        stepsize=0.5,
        disp=False
    )

    best_escape_radius = -result.fun
    escape_volume = (4 / 3) * np.pi * best_escape_radius**3
    return best_escape_radius, escape_volume

# === Execution ===

data = []
try:
    atoms, coords = read_xyz(file_path)
    print(1)
    centroid = np.mean(coords, axis=0)

    pd_dists = get_pd_distances(atoms, coords)
    min_pd, max_pd = (min(pd_dists), max(pd_dists)) if pd_dists else (None, None)
    print(1)
    enc_radius, enc_volume = get_max_enclosed_sphere(atoms, coords, vDW_radii)
    print(1)

    esc_radius, esc_volume = get_escape_radius(coords, atoms, vDW_radii)
    print(1)

    data.append({
        'filename': file_path,
        'num_Pd': np.sum(atoms == 'Pd'),
        'min_Pd–Pd_dist (Å)': round(min_pd, 2) if min_pd else None,
        'max_Pd–Pd_dist (Å)': round(max_pd, 2) if max_pd else None,
        'enclosed_radius (Å)': round(enc_radius, 2),
        'enclosed_volume (Å³)': round(enc_volume, 2),
        'escape_radius (Å)': round(esc_radius, 2),
        'escape_volume (Å³)': round(esc_volume, 2)
    })

except Exception as e:
    print(f"Error processing {file_path}: {e}")

df = pd.DataFrame(data)
print(df)
