"""
ORCA Input Generatorfor use in mega_job.sh

Author: Taradutt Pattnaik
Date: 2026

Description:

Generates ORCA quantum chemistry input files from molecular
geometry entries stored in the MOSES json. To be used in mega_job.sh

Method:
- DFT: B3LYP
- Basis set: 6-31G*
- Tight SCF convergence
- CHELPG charge calculation
- Dipole, quadrupole, and polarizability properties

Usage(as used in mega_job.sh):
python gen_input.py <local_json> <mol_index> <output_inp>

Arguments:
1. local_json  : JSON chunk containing molecule entries
2. mol_index   : Index of molecule within chunk
3. output_inp  : Output ORCA  input file path
"""
import json
import sys

# ====================
# Command-line arguments
# ====================

local_json = sys.argv[1]
mol_index  = int(sys.argv[2])
output_inp = sys.argv[3]

# =====================
# Load molecule from local chunk
# =====================
with open(local_json) as f:
    data = json.load(f)

mol = data[mol_index]
# =====================
# Write ORCA input file
# =====================

with open(output_inp, 'w') as f:
    f.write("! B3LYP 6-31G* TightSCF CHELPG\n") #main electronic methods
    f.write("\n")
    #CHELGP Charge calculation method block start
    f.write("%chelpg\n")
    f.write("  GRID 0.2\n")
    f.write("  RMAX 2.8\n")
    f.write("  VDWRADII COSMO\n")
    f.write("  DIPOLE TRUE\n")
    f.write("end\n")
    f.write("\n")
    #CHELGP charge calculation block end
    
    #Response Properties block start 
    f.write("%elprop\n")
    f.write("  Dipole     true\n")
    f.write("  Quadrupole true\n")
    f.write("  Polar      true\n")
    f.write("end\n")
    #End of response properties
    f.write("\n")
    #scf convergence 
    f.write("%scf\n")
    f.write("  MaxIter 200\n")
    f.write("end\n")
    f.write("\n")
    # Molecular geometry block
    f.write("* xyz 0 1\n")
    for sym, (x, y, z) in zip(mol['species'], mol['coords']):
        f.write(f"  {sym}  {x:.6f}  {y:.6f}  {z:.6f}\n")
    f.write("*\n")
