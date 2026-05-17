"""
Collection + Final Filtering Script

Author: Taradutt Pattnaik 
Date: 2026

Description:
Collects parsed ORCA results from parallel worker outputs,
merge them with molecular metadata(smiles x,y,z etc), apply final quality
filters, and export the final compact dataset

Inputs:
- MOSES_150k_Master.json
- parsed_results/*.jsonl

Output:
- DrugESP_149k_compact.json

Filtering:
- Only converged ORCA calculations
- Only fully parsed entries
- Removes extreme CHELPG charge outliers

Notes:
- Worker outputs are stored in JSONL format 
- Final output is compact/minified JSON
"""
import json
import os
import glob

# Configuration
MASTER_META = "MOSES_150k_Master.json"
WORKER_DIR  = "parsed_results/*.jsonl"
FINAL_NAME  = "DrugESP_149k_compact.json"
CHARGE_THRESHOLD = 3.0

# Main collection routine
def run_collection():
    # Load master metadata
    print("Loading master metadata...")
    
    with open(MASTER_META) as f:
        master = {m['mol_id']: m for m in json.load(f)} # Lookup by mol_id for fast merging
    # Collect converged worker results
    print("Collecting worker results...")
    converged_data = {}
    for jf in glob.glob(WORKER_DIR):
        with open(jf) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get('status') == 'converged' and entry.get('parse_ok'):
                    converged_data[entry['mol_id']] = entry

    print(f"Total converged: {len(converged_data):,}")

    # Merge and Filter
    
    
    # Merge metadata + parsed properties
    final_list = []
    excluded_count = 0

    for mid, parsed in converged_data.items():
        meta = master.get(mid)
        if not meta: continue

        # Physical Outlier Filter->Removes molecules with unrealistic CHELPG charges
        max_q = max(abs(q) for q in parsed['chelpg_charges'])
        if max_q > CHARGE_THRESHOLD:
            excluded_count += 1
            continue

        # Create Compact Dataset Entry
        final_list.append({
            'mol_id': mid,
            'smiles': meta['smiles'],
            'species': meta['species'],
            'coords': meta['coords'],
            'energy_hartree': parsed['energy_hartree'],
            'dipole_debye': parsed['dipole_debye'],
            'chelpg_charges': parsed['chelpg_charges'],
            'gap_ev': parsed['gap_ev'],
            'polarizability_iso': parsed['polarizability_iso']
        })
    # Sort for reproducibility
    final_list.sort(key=lambda x: x['mol_id'])
    

    # Save compact dataset
    with open(FINAL_NAME, 'w', separators=(',', ':')) as f:
        json.dump(final_list, f)
    # Final summary
    print(f" Collection complete.")
    print(f"   Final count: {len(final_list):,}")
    print(f"   Excluded (bad charges): {excluded_count}")

if __name__ == "__main__":
    run_collection()