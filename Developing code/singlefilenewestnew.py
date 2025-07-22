import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import basinhopping

# Van der Waals radii dictionary (Å)
vdW_radii = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
    'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
    'Pd': 1.63, 'Pt': 1.75, 'Cu': 1.40, 'Zn': 1.39,
    'He': 1.40, 'Ne': 1.54, 'Ar': 1.88, 'Xe': 2.16,
    'U': 1.86, 'Au': 1.66, 'Hg': 1.55, 'Ag': 1.72
}

# ---------- File loading ----------
def read_xyz(file_path):
    atoms, coords = [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                atoms.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])
    return np.array(atoms), np.array(coords)


# ---------- Core functions ----------
def escape_radius_from_point_dir(origin, direction, coords, radii, resolution=0.1, max_dist=15.0):
    tree = KDTree(coords)
    direction = direction / np.linalg.norm(direction)
    best_radius = np.inf

    for d in np.arange(0, max_dist, resolution):
        point = origin + d * direction
        dists, idxs = tree.query(point, k=5, distance_upper_bound=4.0)
        for dist, idx in zip(dists, idxs):
            if idx == len(coords):  # no neighbor
                continue
            clearance = dist - radii[idx]
            if clearance < 0:
                return best_radius if best_radius != np.inf else 0.0
            best_radius = min(best_radius, clearance)

    return best_radius if best_radius != np.inf else 0.0


def escape_objective(x, coords, radii):
    origin = x[:3]
    direction = x[3:]
    if np.linalg.norm(direction) < 1e-6:
        return 1e6  # Bad direction
    return -escape_radius_from_point_dir(origin, direction, coords, radii)


def optimize_escape(coords, atoms, n_starting_points=10):
    # Build spatial index
    radii = np.array([vdW_radii.get(atom, 1.7) for atom in atoms])

    best_radius = 0.0
    best_volume = 0.0

    for _ in range(n_starting_points):
        origin = coords[np.random.randint(0, len(coords))]  # inside cage
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        x0 = np.concatenate([origin, direction])

        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "args": (coords, radii)
        }

        result = basinhopping(
            escape_objective,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=10,
            stepsize=1.0,
            disp=False
        )

        radius = -result.fun
        if radius > best_radius:
            best_radius = radius
            best_volume = (4/3) * np.pi * radius**3

    return best_radius, best_volume


# ---------- Main ----------
if __name__ == "__main__":
    file_path = "samplestructure.xyz"
    atoms, coords = read_xyz(file_path)

    best_r, best_v = optimize_escape(coords, atoms, n_starting_points=8)

    print(f"Best escape radius: {round(best_r, 2)} Å")
    print(f"Escape sphere volume: {round(best_v, 2)} Å³")
