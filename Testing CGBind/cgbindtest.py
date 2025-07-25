import cgbind

linker = cgbind.Linker(
    smiles='C1(C#CC2=CC=CC(C#CC3=CC=CN=C3)=C2)=CC=CN=C1',  # EZEVAI linker
    arch_name='m2l4'
)

cage = cgbind.Cage(linker=linker, metal='Pd')
print(cage)
cage.print_xyz_file(filename="mop_output.xyz")

m_m_dist = cage.get_m_m_dist()
vol = cage.get_cavity_vol()
escape_sphere = cage.get_max_escape_sphere()
escape_volume = (4 / 3) * 3.14159 * escape_sphere**3

print(f"\nPd–Pd Distance: {m_m_dist:.2f} Å")
print(f"Cavity Volume: {vol:.2f} Å³")
print(f"Maximum Escape Sphere Radius: {escape_sphere:.2f} Å")
print(f"Escape Sphere Volume: {escape_volume:.2f} Å³")

