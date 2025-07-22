from pywindow.molecular import Molecule

mol = Molecule(file_path="structures/mop_output.xyz")
mol.full_analysis()

print("Number of windows:", mol.number_of_windows)
print("Window diameters:", mol.window_diameter)
print("Interior diameter:", mol.interior_diameter)
