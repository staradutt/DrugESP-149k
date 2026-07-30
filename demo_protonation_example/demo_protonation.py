"""
Demo: Protonation State Enumeration -> ORCA Input Generation
Author: Taradutt Pattnaik
Date: 2026

Description:
Self-contained demonstration of the charged-extension methodology used
to build the protonated/deprotonated ionization states in this dataset.
Given the main dataset, this script:

  1. Picks one parent molecule (deterministically, first suitable match)
  2. Enumerates protonation/deprotonation states at physiological pH
     (6.4-8.4) using dimorphite-dl
  3. Applies the same chemical-sanity filters used for the full dataset:
       - rejects amide/lactam/anilide nitrogen ionization (a known
         dimorphite-dl false-positive pattern -- amide N is neither
         meaningfully basic nor acidic under physiological pH)
       - restricts +1 states to "well-populated" basic sites (aliphatic
         amines, pyridine-type aromatic N, imidazole-type aromatic N),
         excluding weakly-basic azoles (tetrazole/triazole/oxadiazole/
         pyrazole/thiazole/oxazole/diazines)
  4. Generates 3D geometry (ETKDGv3 + MMFF94) for the neutral parent,
     one +1 variant, and one -1 variant
  5. Writes ORCA input files (B3LYP/6-31+G*/CPCM(water), TightSCF,
     CHELPG) for all three

Requires: rdkit, dimorphite-dl (pip install dimorphite-dl)

Usage:
    Edit DATASET_PATH below if needed, then run:
    python3 demo_protonation_orca_inputs.py
    (or paste into a Jupyter cell / %run it -- no command-line
    arguments needed)

Output (written to ./demo_protonation_example/):
    neutral.inp, pos1.inp, neg1.inp
    summary.txt (which molecule was chosen and why)
"""

import os
import json

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from dimorphite_dl import protonate_smiles

RDLogger.DisableLog('rdApp.*')

# Edit this path directly rather than passing it as a command-line
# argument -- avoids Jupyter's kernel connection file showing up in
# sys.argv and being mistaken for the dataset path.
DATASET_PATH = "Drug_ESP_149k.json"
OUT_DIR = "demo_protonation_example"
PH_MIN, PH_MAX = 6.4, 8.4
RANDOM_SEED = 42

AMIDE_N = Chem.MolFromSmarts("[#7]C(=O)")
SULFONAMIDE_N = Chem.MolFromSmarts("[#7]S(=O)(=O)")


def passes_amide_lactam_filter(mol):
    charged_atoms = [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
    if not charged_atoms:
        return False
    amide_n_idx = {m[0] for m in mol.GetSubstructMatches(AMIDE_N)}
    sulfonamide_n_idx = {m[0] for m in mol.GetSubstructMatches(SULFONAMIDE_N)}
    for atom in charged_atoms:
        if atom.GetSymbol() != "N":
            continue
        idx = atom.GetIdx()
        if idx in amide_n_idx:
            if idx in sulfonamide_n_idx and atom.GetFormalCharge() == -1:
                continue
            return False
    return True


def is_well_populated_basic_site(mol, atom):
    if not atom.GetIsAromatic():
        return True
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if atom.GetIdx() not in ring:
            continue
        ring_size = len(ring)
        ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
        n_positions = [i for i, a in enumerate(ring_atoms) if a.GetSymbol() == "N"]
        other_hetero = sum(1 for a in ring_atoms if a.GetSymbol() not in ("C", "N"))
        if ring_size == 6 and len(n_positions) == 1 and other_hetero == 0:
            return True
        if ring_size == 5 and len(n_positions) == 2 and other_hetero == 0:
            pos_diff = abs(n_positions[0] - n_positions[1])
            adjacent = pos_diff == 1 or pos_diff == ring_size - 1
            if not adjacent:
                return True
        return False
    return False


def passes_pos1_site_filter(mol):
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() == 1 and atom.GetSymbol() == "N":
            if not is_well_populated_basic_site(mol, atom):
                return False
    return True


def find_valid_variants(smiles):
    try:
        variants = protonate_smiles(smiles, ph_min=PH_MIN, ph_max=PH_MAX, precision=1.0)
    except Exception:
        return None, None

    pos1, neg1 = None, None
    for v in variants:
        mol = Chem.MolFromSmiles(v)
        if mol is None:
            continue
        q = Chem.GetFormalCharge(mol)
        if q == 1 and pos1 is None:
            if passes_amide_lactam_filter(mol) and passes_pos1_site_filter(mol):
                pos1 = Chem.MolToSmiles(mol)
        elif q == -1 and neg1 is None:
            if passes_amide_lactam_filter(mol):
                neg1 = Chem.MolToSmiles(mol)
    return pos1, neg1


def generate_3d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    params.enforceChirality = True
    if AllChem.EmbedMolecule(mol, params) == -1:
        return None
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    if mp is None:
        return None
    ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
    if ff is None or ff.Minimize(maxIts=2000) != 0:
        return None
    conf = mol.GetConformer()
    species = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = conf.GetPositions().tolist()
    return species, coords


def write_orca_input(path, species, coords, charge, mult=1):
    with open(path, "w") as f:
        f.write("! B3LYP 6-31+G* CPCM(water) TightSCF CHELPG\n\n")
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

    print("\nSearching for a suitable demo molecule "
          f"(one with both a valid +1 and -1 site, pH {PH_MIN}-{PH_MAX})...")
    chosen = None
    pos1_smiles = neg1_smiles = None
    for entry in dataset:
        pos1, neg1 = find_valid_variants(entry["smiles"])
        if pos1 and neg1:
            chosen = entry
            pos1_smiles, neg1_smiles = pos1, neg1
            break

    if chosen is None:
        print("No molecule with both +1 and -1 sites found in this dataset -- exiting.")
        return

    print(f"\nChosen molecule: mol_id={chosen['mol_id']}")
    print(f"  Neutral: {chosen['smiles']}")
    print(f"  +1:      {pos1_smiles}")
    print(f"  -1:      {neg1_smiles}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\nGenerating 3D geometries (ETKDGv3 + MMFF94)...")
    results = {}
    for label, smi, charge in [
        ("neutral", chosen["smiles"], 0),
        ("pos1", pos1_smiles, 1),
        ("neg1", neg1_smiles, -1),
    ]:
        geom = generate_3d(smi)
        if geom is None:
            print(f"  {label}: geometry generation failed")
            continue
        species, coords = geom
        out_path = os.path.join(OUT_DIR, f"{label}.inp")
        write_orca_input(out_path, species, coords, charge)
        results[label] = (smi, charge)
        print(f"  {label}: wrote {out_path}")

    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Protonation state enumeration demo\n")
        f.write("=" * 40 + "\n")
        f.write(f"Source dataset: {DATASET_PATH}\n")
        f.write(f"Chosen mol_id: {chosen['mol_id']}\n")
        f.write(f"pH window: {PH_MIN}-{PH_MAX} (dimorphite-dl)\n\n")
        for label, (smi, charge) in results.items():
            f.write(f"{label} (charge {charge:+d}): {smi}\n")
        f.write("\nLevel of theory: B3LYP/6-31+G*/CPCM(water), TightSCF, CHELPG\n")
        f.write("(matches the charged-extension production level of theory)\n")
    print(f"\nWrote: {summary_path}")


if __name__ == "__main__":
    main()
