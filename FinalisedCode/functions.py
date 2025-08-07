import numpy as np
import math
from rdkit.Chem import GetPeriodicTable
from rdkit import Chem
from rdkit.Chem import AllChem
import os
PERIODIC_TABLE = GetPeriodicTable()

VDW_RADII_DICT = {'H': 1.20, 'He': 1.40, 'Li': 1.82, 'Be': 1.53, 'B': 1.92, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54, 'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88, 'K': 2.75, 'Ca': 2.31, 'Sc': 2.30, 'Ti': 2.15, 'V': 2.05, 'Cr': 2.05, 'Mn': 2.05, 'Fe': 2.00, 'Co': 2.00, 'Ni': 1.97, 'Cu': 1.96, 'Zn': 2.01, 'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02, 'Rb': 3.03, 'Sr': 2.49, 'Y': 2.40, 'Zr': 2.30, 'Nb': 2.15, 'Mo': 2.10, 'Tc': 2.05, 'Ru': 2.05, 'Rh': 2.00, 'Pd': 2.05, 'Ag': 2.03, 'Cd': 2.18, 'In': 1.93, 'Sn': 2.17, 'Sb': 2.06, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16, 'Cs': 3.43, 'Ba': 2.68, 'La': 2.50, 'Ce': 2.48, 'Pr': 2.47, 'Nd': 2.45, 'Pm': 2.43, 'Sm': 2.42, 'Eu': 2.40, 'Gd': 2.38, 'Tb': 2.37, 'Dy': 2.35, 'Ho': 2.33, 'Er': 2.32, 'Tm': 2.30, 'Yb': 2.28, 'Lu': 2.27, 'Hf': 2.25, 'Ta': 2.20, 'W': 2.10, 'Re': 2.05, 'Os': 2.00, 'Ir': 2.00, 'Pt': 2.05, 'Au': 2.10, 'Hg': 2.05, 'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07, 'Po': 1.97, 'At': 2.02, 'Rn': 2.20}

USE_VDW_RADII = False



def get_radius(atom):
    if USE_VDW_RADII:
        r = VDW_RADII_DICT.get(atom, 0)
        return r if r > 0 else 1.70
    return PERIODIC_TABLE.GetRcovalent(atom)

def is_metal(symbol):
    Z = PERIODIC_TABLE.GetAtomicNumber(symbol)
    if Z in {3, 11, 19, 37, 55, 87, 4, 12, 20, 38, 56, 88, 13, 31, 49, 50, 81, 82, 83, 113, 114, 115, 116}: return True
    if 21 <= Z <= 30 or 39 <= Z <= 48 or 72 <= Z <= 80 or 104 <= Z <= 112: return True
    if 57 <= Z <= 71 or 89 <= Z <= 103: return True
    return False

def average_metal_distance(atoms, coords):
    idx = [i for i, a in enumerate(atoms) if is_metal(a)]
    if len(idx) < 2: return None
    dists = [np.linalg.norm(coords[i]-coords[j]) for i in range(len(idx)) for j in range(i+1, len(idx))]
    return np.mean(dists) if dists else None

def largest_internal_sphere(atoms, coords):
    idx = [i for i, a in enumerate(atoms) if is_metal(a)]
    centroid = np.mean(coords[idx], axis=0) if idx else np.mean(coords, axis=0)
    min_dist, closest = np.inf, None
    for i, atom in enumerate(atoms):
        r = get_radius(atom)
        d = np.linalg.norm(coords[i]-centroid)-r
        if d < min_dist:
            min_dist, closest = d, atom
    min_dist = max(0, min_dist)
    v = (4/3)*math.pi*min_dist**3
    return closest, min_dist*2, v

def pore_size_along_vector(atoms, coords, vector):
    centroid = np.mean(coords, axis=0)
    vector = vector/np.linalg.norm(vector)
    max_dist = 0
    for i, atom in enumerate(atoms):
        r = get_radius(atom)
        rel = coords[i]-centroid
        proj = np.dot(rel, vector)
        if proj > 0:
            perp = np.linalg.norm(rel-proj*vector)-r
            if perp < max_dist or max_dist == 0:
                max_dist = perp
    max_dist = max(0, max_dist)
    d = max_dist*2
    v = (4/3)*math.pi*max_dist**3
    return d, v

def generate_sphere_directions(n):
    phi = math.pi*(3.-math.sqrt(5.))
    return [np.array([math.cos(phi*i)*math.sqrt(1-(1-(i/(n-1))*2)**2), 1-(i/(n-1))*2, math.sin(phi*i)*math.sqrt(1-(1-(i/(n-1))*2)**2)]) for i in range(n)]

def max_pore_size_all_directions(atoms, coords, n_directions=1000):
    directions = generate_sphere_directions(n_directions)
    max_d, max_v, best_vec = 0, 0, None
    for vec in directions:
        d, v = pore_size_along_vector(atoms, coords, vec)
        if d > max_d:
            max_d, max_v, best_vec = d, v, vec
    return max_d, max_v, best_vec

def read_xyz(filepath):
    atoms, coords = [], []
    with open(filepath) as f:
        for line in f.readlines()[2:]:
            p = line.strip().split()
            if len(p) >= 4:
                atoms.append(p[0])
                coords.append([float(x) for x in p[1:4]])
    return np.array(atoms), np.array(coords)


def plot_sphere_directions(n=1000):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    d = generate_sphere_directions(n)
    xs, ys, zs = [v[0] for v in d], [v[1] for v in d], [v[2] for v in d]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, s=10, c='b', alpha=0.7)
    ax.set(xlabel='X', ylabel='Y', zlabel='Z', title=f'{n} directions on a sphere (Fibonacci sphere)')
    plt.tight_layout(); plt.show()




def get_xyz_coords_and_masses_from_file(xyz_path, unit_conversion=0.01):
    with open(xyz_path, 'r') as f:
        lines = f.readlines()[2:]
    symbols = []
    coords = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    coords = np.array(coords) * unit_conversion
    pt = Chem.GetPeriodicTable()
    masses = np.array([pt.GetAtomicWeight(s) for s in symbols])
    return coords, masses

def calculate_asphericity_from_xyz(xyz_path):
    coords, masses = get_xyz_coords_and_masses_from_file(xyz_path)
    total_mass = np.sum(masses)
    center_of_mass = np.average(coords, axis=0, weights=masses)
    centered_coords = coords - center_of_mass

    I = np.zeros((3, 3))
    for i in range(len(masses)):
        x, y, z = centered_coords[i]
        m = masses[i]
        I[0, 0] += m * (y**2 + z**2)
        I[1, 1] += m * (x**2 + z**2)
        I[2, 2] += m * (x**2 + y**2)
        I[0, 1] -= m * x * y
        I[0, 2] -= m * x * z
        I[1, 2] -= m * y * z
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]

    eigenvals = np.linalg.eigvalsh(I)
    λ1, λ2, λ3 = sorted(eigenvals)

    # Asphericity formula
    mean = (λ1 + λ2 + λ3) / 3
    asphericity = (λ1 - mean)**2 + (λ2 - mean)**2 + (λ3 - mean)**2

    return asphericity
