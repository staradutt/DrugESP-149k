import json
import sys

local_json = sys.argv[1]
mol_index  = int(sys.argv[2])
output_inp = sys.argv[3]

with open(local_json) as f:
    data = json.load(f)

mol = data[mol_index]

with open(output_inp, 'w') as f:
    f.write("! wB97X-D3 6-31G* TightSCF CHELPG\n")
    f.write("\n")
    f.write("%chelpg\n")
    f.write("  GRID 0.2\n")
    f.write("  RMAX 2.8\n")
    f.write("  VDWRADII COSMO\n")
    f.write("  DIPOLE TRUE\n")
    f.write("end\n")
    f.write("\n")
    f.write("%elprop\n")
    f.write("  Dipole     true\n")
    f.write("  Quadrupole true\n")
    f.write("  Polar      true\n")
    f.write("end\n")
    f.write("\n")
    f.write("%scf\n")
    f.write("  MaxIter 200\n")
    f.write("end\n")
    f.write("\n")
    f.write("* xyz 0 1\n")
    for sym, (x, y, z) in zip(mol["species"], mol["coords"]):
        f.write(f"  {sym}  {x:.6f}  {y:.6f}  {z:.6f}\n")
    f.write("*\n")
