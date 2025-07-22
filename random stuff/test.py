from pywindow.molecular import Molecule
mol = Molecule.from_xyz("test.xyz")
mol.calculate_volume()
mol.calculate_windows()
