import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from rdkit import Chem

from functions import (
    get_radius,
    is_metal,
    average_metal_distance,
    largest_internal_sphere,
    pore_size_along_vector,
    generate_sphere_directions,
    max_pore_size_all_directions
)

# Restore variables needed for main code
# folder = 'expt_structures_from_OntoMOPs_KG'  # adjust if needed
# output_csv = 'all_results_for_ONTOMOPKG.csv'
folder = 'zr_tet_mops_xtb_results'  # adjust if needed
output_csv = 'all_results_for_zrTET.csv'
from functions import (
    get_radius,
    is_metal,
    average_metal_distance,
    largest_internal_sphere,
    pore_size_along_vector,
    generate_sphere_directions,
    max_pore_size_all_directions,
    read_xyz,
    plot_sphere_directions,
    calculate_asphericity_from_xyz,
)
n_directions = 200
selected_xyz_file = '1823162.xyz'
atoms_selected = coords_selected = best_vec_selected = None
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'avg_metal_dist', 'inner_atom', 'inner_diameter', 'inner_volume', 'max_pore_size_diameter', 'max_pore_size_volume','asphericity'])
    count = 0
    total = len([fn for fn in os.listdir(folder) if fn.endswith('.xyz')])
    for filename in os.listdir(folder):
        if filename.endswith('.xyz'):
            filepath = os.path.join(folder, filename)
            try:
                atoms, coords = read_xyz(filepath)
                avg_metal = average_metal_distance(atoms, coords)
                asphericity = calculate_asphericity_from_xyz(filepath)
                if asphericity is None:
                    asphericity = 0
                inner_atom, inner_diam, inner_vol = largest_internal_sphere(atoms, coords)
                if filename == selected_xyz_file:
                    max_pore_size, max_pore_vol, best_vec_selected = max_pore_size_all_directions(atoms, coords, n_directions)
                    coords_selected, atoms_selected = coords, atoms
                else:
                    max_pore_size, max_pore_vol, _ = max_pore_size_all_directions(atoms, coords, n_directions)
                writer.writerow([filename, avg_metal, inner_atom, round(inner_diam,2), round(inner_vol,2), round(max_pore_size,2), round(max_pore_vol,2), round(asphericity, 2)])
                count += 1
                print(f"Processed {count}/{total}: {filename}")
            except Exception as e:
                writer.writerow([filename, 'error', 'error', 'error', 'error', 'error', 'error'])
                print(f"Error processing {filename}: {e}")

# Uncomment to plot sphere directions

# atom_colors = {'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red', 'F': 'green', 'Cl': 'green', 'Br': 'brown', 'I': 'purple', 'Cu': 'orange', 'Zn': 'lightblue', 'S': 'yellow', 'P': 'orange', 'Fe': 'darkred', 'Co': 'magenta', 'Ni': 'teal'}
# sizes = [get_radius(a)*20 for a in atoms_selected]
# colors = [atom_colors.get(a, 'lightgray') for a in atoms_selected]
# directions = np.array(generate_sphere_directions(1000))
# centroid = np.mean(coords_selected, axis=0)
# line_length = 20
# best_line = np.stack([centroid, centroid+best_vec_selected*line_length])
# fig = go.Figure()
# fig.add_trace(go.Scatter3d(x=coords_selected[:,0], y=coords_selected[:,1], z=coords_selected[:,2], mode='markers', marker=dict(size=sizes, color=colors, opacity=0.7, line=dict(width=0)), text=atoms_selected, name='Atoms', hoverinfo='text'))
# fig.add_trace(go.Scatter3d(x=directions[:,0]+centroid[0], y=directions[:,1]+centroid[1], z=directions[:,2]+centroid[2], mode='markers', marker=dict(size=4, color='blue', opacity=0.5), name='Directions'))
# fig.add_trace(go.Scatter3d(x=best_line[:,0], y=best_line[:,1], z=best_line[:,2], mode='lines', line=dict(color='red', width=6), name='Best Vector'))
# fig.add_trace(go.Scatter3d(x=[best_line[1,0]], y=[best_line[1,1]], z=[best_line[1,2]], mode='markers', marker=dict(size=10, color='red'), name='Best Vector Tip'))
# fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), title=f'Plotly: Directions, Best Vector, and Atoms for {selected_xyz_file}', legend=dict(itemsizing='constant'))
# fig.show()

#volume of molecule, surface area, molecular stuff
#centroid different calculations
#molecular descriptors  