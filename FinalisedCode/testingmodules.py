import os
import numpy as np
import csv
import math
from rdkit.Chem import GetPeriodicTable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Explicit import for 3D plotting
PERIODIC_TABLE = GetPeriodicTable()
atom = 'Chlorine'
r = PERIODIC_TABLE.GetVdwRad(atom)
print(f"Van der Waals radius of {atom}: {r}")
r = PERIODIC_TABLE.GetRcovalent(atom)
print(f"Covalent radius of {atom}: {r}")