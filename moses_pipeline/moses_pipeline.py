"""
MOSES 150K 3D Data generation Pipeline
Author: Taradutt Pattnaik
Date: 2026

Description:
This script filters molecules from the MOSES dataset, performs diversity sampling,
and generates optimized 3D conformations using RDKit (ETKDGv3 + MMFF94 optimization).

Output:
- geometry_checkpoint.json (intermediate output)
- MOSES_150k_Master.json (final output)
"""

import os
import json
import random
import collections
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from joblib import Parallel, delayed
from tqdm import tqdm
# =====================
# --- CONFIGURATION ---
# =====================

TARGET_MOLECULES = 150000
RANDOM_SEED = 42 #  this controls REPRODUCIBILITY
CHECKPOINT_EVERY = 10000
N_JOBS = -1 

MASTER_PATH = "MOSES_150k_Master.json"
CHECKPOINT_PATH = "geometry_checkpoint.json"

# Original Chemical Constraints (allowed atoms and sizes)
ALLOWED_ATOMIC_NUMS = {1, 6, 7, 8, 9, 16, 17}  # H C N O F S Cl
MIN_HEAVY, MAX_HEAVY = 10, 35 #number of heavy atoms
MIN_MW, MAX_MW = 150, 500 #molecular weight range 

random.seed(RANDOM_SEED)
# =====================
# --- Chemical filtering  ---
# =====================
def passes_filters(smi):
    """
    Applies chemical filters to SMILES strings ..

    Filters:
    - Valid RDKit parsing

    - No multi-fragment molecules 
    - Allowed atomic set only (described above for B3LYP)
    - Heavy atom count range
    - Molecular weight min/max
    - Neutral charge only
    - Must contain at least one heteroatom (non C/H)
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return False, "parse_failed"
    
    if '.' in smi: return False, "fragment"
    atomic_nums = {a.GetAtomicNum() for a in mol.GetAtoms()}
    
    if not atomic_nums.issubset(ALLOWED_ATOMIC_NUMS):
        return False, "bad_elements"
    
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < MIN_HEAVY or n_heavy > MAX_HEAVY:
        return False, "size"
    
    mw = Descriptors.MolWt(mol)
    if mw < MIN_MW or mw > MAX_MW:
        return False, "mw"
    
    if Chem.GetFormalCharge(mol) != 0:
        return False, "charged"
    
    non_ch = {a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum() not in {1, 6}}
    if len(non_ch) == 0:
        return False, "hydrocarbon"
    return True, "ok"
# =====================
# --- 3D Structure generation ---
# =====================
def generate_single(mol_id, smi):
    """
    Generates optimized 3D struture  for a molecule

    Steps:
    - SMILES -> RDKit molecule
    - Adds hydrogens
    - Embed using ETKDGv3
    - Optimize with MMFF94 

    Returns:
    (result_dict, None) on success
    (None, error_info) on failure
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_SEED
        params.enforceChirality = True
        if AllChem.EmbedMolecule(mol, params) == -1:
            return None, (mol_id, smi, "embed_failed")
        if AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94", maxIters=2000) == -1:
            return None, (mol_id, smi, "mmff_failed")
        conf = mol.GetConformer()
        return {
            "mol_id": mol_id,
            "smiles": smi,
            "species": [a.GetSymbol() for a in mol.GetAtoms()],
            "coords": conf.GetPositions().tolist(),
            "n_atoms": mol.GetNumAtoms(),
            "n_heavy": mol.GetNumHeavyAtoms(),
            "mw": round(Descriptors.MolWt(Chem.MolFromSmiles(smi)), 3),
            "charge": 0,
            "status": "geometry_ok"
        }, None
    except Exception as e:
        return None, (mol_id, smi, str(e))

# =====================
# --- MAIN PIPELINE ---
# =====================
def main():
    # =====================
    # ---Step 1. Load  MOSES ---
    # =====================
    print("[1/5] Downloading MOSES dataset files...")
    train_url = "https://github.com/molecularsets/moses/raw/master/data/train.csv"
    test_url = "https://github.com/molecularsets/moses/raw/master/data/test.csv"
    
    train_df = pd.read_csv(train_url)
    test_df = pd.read_csv(test_url)
    all_smiles = list(train_df['SMILES']) + list(test_df['SMILES'])
    print(f"Loaded {len(all_smiles):,} total SMILES.")

    # =====================
    # ---Step 2. Filtering ---
    # =====================
    print(f"[2/5] Applying chemical filters...")
    passed = []
    for smi in tqdm(all_smiles, desc="Filtering"):
        ok, _ = passes_filters(smi)
        if ok: passed.append(smi)
    print(f"Passed filters: {len(passed):,}")

    # =====================
    # ---Step 3. Diversity Sampling ---
    # =====================
    
    print(f"[3/5] Diversity sampling {TARGET_MOLECULES:,} molecules...")
    mol_by_size = collections.defaultdict(list)
    for smi in tqdm(passed, desc="Grouping by size"):
        mol = Chem.MolFromSmiles(smi)
        mol_by_size[mol.GetNumHeavyAtoms()].append(smi)
    
    selected = []
    total_passed = len(passed)
    for size in sorted(mol_by_size.keys()):
        smis = mol_by_size[size]
        n_select = int(len(smis) / total_passed * TARGET_MOLECULES)
        selected.extend(random.sample(smis, min(n_select, len(smis))))
    
    already = set(selected)
    remainder = [s for s in passed if s not in already]
    random.shuffle(remainder)
    needed = TARGET_MOLECULES - len(selected)
    selected.extend(remainder[:needed])
    random.shuffle(selected)
    print(f"Selected {len(selected):,} molecules.")

   # =====================
    # ---Step 4. Generating optimized 3D structures with checkpoints---
    # =====================
    print("[4/5] Generating 3D structures...")
    molecule_bank = []
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            molecule_bank = json.load(f)
        done_ids = {e['mol_id'] for e in molecule_bank}
        remaining = [(i, smi) for i, smi in enumerate(selected) if i not in done_ids]
        print(f"Resuming: {len(molecule_bank):,} done, {len(remaining):,} remaining")
    else:
        remaining = list(enumerate(selected))
        print(f"Starting fresh: {len(remaining):,} molecules")

    for chunk_start in range(0, len(remaining), CHECKPOINT_EVERY):
        chunk = remaining[chunk_start:chunk_start + CHECKPOINT_EVERY]
        print(f"Processing Chunk {chunk_start//CHECKPOINT_EVERY + 1}...")
        results = Parallel(n_jobs=N_JOBS, return_as="generator")(
            delayed(generate_single)(mol_id, smi) for mol_id, smi in chunk
        )
        for entry, fail in tqdm(results, total=len(chunk)):
            if entry: molecule_bank.append(entry)
        
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(molecule_bank, f)

    # =====================
    # ---Step 5. Final Json Export ---
    # =====================
    print(f"[5/5] Saving master JSON to {MASTER_PATH}")
    with open(MASTER_PATH, 'w') as f:
        json.dump(molecule_bank, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()