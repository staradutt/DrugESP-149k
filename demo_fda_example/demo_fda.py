"""
Demo: FDA-Approved Molecule -> ORCA Input Generation
Author: Taradutt Pattnaik
Date: 2026

Description:
Self-contained demonstration of ORCA input generation for the FDA
reference set. Picks the first neutral molecule in the FDA dataset and
writes an ORCA input file at the main dataset's gas-phase level of
theory (B3LYP/6-31G*, no CPCM -- the FDA-neutral subset matches the
core dataset's level of theory, unlike the charged extension).

Usage:
    Edit DATASET_PATH below if needed, then run:
    python3 demo_fda_neutral_orca_input.py
    (or paste into a Jupyter cell / %run it -- no command-line
    arguments needed)

Output (written to ./demo_fda_example/):
    fda_neutral.inp
    summary.txt
"""

import os
import json

# Edit this path directly rather than passing it as a command-line
# argument -- avoids Jupyter's kernel connection file showing up in
# sys.argv and being mistaken for the dataset path.
DATASET_PATH = "Drug_ESP_FDA.json"
OUT_DIR = "demo_fda_example"


def write_orca_input(path, species, coords, charge, mult=1):
    with open(path, "w") as f:
        f.write("! B3LYP 6-31G* TightSCF CHELPG\n\n")
        f.write("%chelpg\n  GRID 0.2\n  RMAX 2.8\n  VDWRADII COSMO\n  DIPOLE TRUE\nend\n\n")
        f.write("%elprop\n  Dipole     true\n  Quadrupole true\n  Polar      true\nend\n\n")
        f.write("%scf\n  MaxIter 200\nend\n\n")
        f.write(f"* xyz {charge} {mult}\n")
        for sym, (x, y, z) in zip(species, coords):
            f.write(f"  {sym}  {x:.6f}  {y:.6f}  {z:.6f}\n")
        f.write("*\n")


def main():
    print(f"Loading {DATASET_PATH}...")
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    print(f"  {len(dataset):,} molecules")

    chosen = next((e for e in dataset if e.get("charge", 0) == 0), None)
    if chosen is None:
        print("No neutral entry found -- exiting.")
        return

    print(f"\nChosen molecule: mol_id={chosen['mol_id']}")
    if "drug_name" in chosen:
        print(f"  drug_name: {chosen['drug_name']}")
    if "fda_appl_no" in chosen:
        print(f"  fda_appl_no: {chosen['fda_appl_no']} ({chosen.get('fda_appl_type')})")
        print(f"  fda_trade_name: {chosen.get('fda_trade_name')}")
    print(f"  smiles: {chosen['smiles']}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "fda_neutral.inp")
    write_orca_input(out_path, chosen["species"], chosen["coords"], chosen.get("charge", 0))
    print(f"\nWrote: {out_path}")

    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("FDA-approved molecule ORCA input demo\n")
        f.write("=" * 40 + "\n")
        f.write(f"Source dataset: {DATASET_PATH}\n")
        f.write(f"Chosen mol_id: {chosen['mol_id']}\n")
        if "drug_name" in chosen:
            f.write(f"drug_name: {chosen['drug_name']}\n")
        if "fda_appl_no" in chosen:
            f.write(f"fda_appl_no: {chosen['fda_appl_no']} ({chosen.get('fda_appl_type')})\n")
            f.write(f"fda_trade_name: {chosen.get('fda_trade_name')}\n")
        f.write(f"smiles: {chosen['smiles']}\n\n")
        f.write("Level of theory: B3LYP/6-31G*, TightSCF, CHELPG (gas-phase)\n")
        f.write("(matches the main DrugESP-149k dataset's level of theory)\n")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
