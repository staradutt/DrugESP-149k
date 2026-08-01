# DrugESP — electrostatic & electronic properties for drug-like molecules

Dataset and code supporting DrugESP: a quantum-chemical dataset of electrostatic and electronic properties for drug-like molecules, plus protonated/deprotonated and FDA-approved reference extensions. 

DOI (dataset JSON files on Zenodo): https://doi.org/10.5281/zenodo.21709579

What this is
------------
DrugESP provides DFT-computed electrostatic potentials, atomic partial charges (CHELPG), and a set of electronic properties for ~149k drug-like molecules, along with matched charged/protonated variants
- data processing and molecule collection scripts,
- HPC/ORCA job generation and parsing tools,
- validation and analysis scripts,
- example ORCA inputs and small case studies,
- SchNet training/evaluation scripts and a pretrained checkpoint.

Stack
-----
- Language(s): Python (primary)
- Tooling / runtime: ORCA (for quantum chemistry), SLURM (common HPC scheduler) — used by the provided job scripts
- Notable libraries you will likely use: RDKit (SMILES, molecule handling), NumPy, SciPy/Matplotlib (analysis/plots), PyTorch (SchNet training/evaluation)

Repository layout (top-level)
-----------------------------
```
DrugESP_149k_moses_mapping.json   # mapping to MOSES indices/splits (mol_id -> source)
ESP_case_study/                   # CHELPG-vs-QM ESP validation scripts and data
fig_method_validation/            # method-validation figure bundle (functional & geometry sensitivity)
QMHPC_workflow/                   # ORCA/HPC production pipeline: input generation, job scripts, parsing, QC
moses_pipeline/                   # molecule collection, filtering, and geometry generation
demo_protonation_example/         # example protonation-state enumeration + ORCA inputs
demo_fda_example/                 # ORCA input generation for FDA-approved reference set
SchNetHPC/                        # SchNet training, evaluation, and a trained model checkpoint
USAGE.md                          # usage examples for loading & working with the JSON dataset
README.md                         # this file
LICENSE                           # MIT license
oxcarbazepine_sample.inp          # example ORCA input
```

Note: The `Func_Val/` and `Geom_Val/` folders previously used for functional- and geometry-sensitivity validation were removed; their validation summary and the reproducibility bundle for the method-validation figure are now available in `fig_method_validation/` (see below).

How it fits together
--------------------
- moses_pipeline/ gathers and filters source molecules (MOSES mapping), prepares initial geometries and SMILES for DFT runs.
- QMHPC_workflow/ turns those molecules into ORCA input files, contains a submission helper (mega_job.sh) and parsers that extract energies, CHELPG charges and other properties from ORCA output.
- ESP_case_study/ contains code and data used to validate CHELPG-derived ESPs vs direct QM ESPs (example script: `esp_compare.py`, supporting data and figures included).
- fig_method_validation/ contains the reproducibility bundle and summary for the method-validation figure, which captures both the functional sensitivity (B3LYP vs ωB97X-D3) and geometry sensitivity (MMFF94 vs DFT-optimized) comparisons used in the paper; this replaces the older Func_Val/ and Geom_Val/ folders.
- SchNetHPC/ contains code to train and evaluate a SchNet model to predict CHELPG charges and related properties; a pretrained checkpoint (schnet_chelpg_best.pt) and evaluation scripts are included.
- demo_protonation_example/ and demo_fda_example/ are small runnable examples showing how protonation/ionization variants and FDA subset inputs were generated and tested.

Quick start (local, analysis & demo)
-----------------------------------
1. Clone the repo:
```bash
git clone https://github.com/staradutt/DrugESP-149k.git
cd DrugESP-149k
```

2. Create a Python environment and install common analysis dependencies. RDKit is recommended via conda:
```bash
# example (conda recommended for RDKit)
conda create -n drugesp python=3.8
conda activate drugesp
conda install -c conda-forge rdkit numpy scipy matplotlib
pip install torch torchvision  # or install PyTorch per your CUDA / CPU setup
```

3. (Optional) Download the dataset JSON files from Zenodo (DOI above) into the repo root or a data/ directory:
- Drug_ESP_149k.json
- Drug_ESP_charged.json
- Drug_ESP_protonated_neutral.json
- Drug_ESP_FDA.json
- dataset_metadata.json

Usage examples (loading & basic queries)
---------------------------------------
The `USAGE.md` contains several short examples; here are the most useful snippets.

- Load a dataset file:
```python
import json

with open("Drug_ESP_149k.json") as f:
    dataset = json.load(f)

print(f"{len(dataset):,} molecules")
print(dataset[0].keys())
```

- Look up a molecule by mol_id:
```python
with open("Drug_ESP_149k.json") as f:
    main_dataset = json.load(f)
main_by_id = {e["mol_id"]: e for e in main_dataset}
mol = main_by_id[0]
print(mol["smiles"], mol["species"])
```

- Find by SMILES using RDKit canonicalization:
```python
from rdkit import Chem

def find_by_smiles(dataset, target_smiles):
    target = Chem.MolToSmiles(Chem.MolFromSmiles(target_smiles))
    for entry in dataset:
        m = Chem.MolFromSmiles(entry["smiles"])
        if m and Chem.MolToSmiles(m) == target:
            return entry
    return None

mol = find_by_smiles(main_dataset, "CCO")
```

- Load stored geometry into an RDKit molecule (coords are stored per-entry):
```python
def to_rdkit_mol(entry):
    m = Chem.MolFromSmiles(entry["smiles"])
    m = Chem.AddHs(m)
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, (x, y, z) in enumerate(entry["coords"]):
        conf.SetAtomPosition(i, (x, y, z))
    m.AddConformer(conf)
    return m
```

Charged / protonated / FDA subsets
----------------------------------
- Charged dataset (`Drug_ESP_charged.json`) entries include `charge` (+1 or -1) and `parent_mol_id`.
- Protonated-neutral dataset (`Drug_ESP_protonated_neutral.json`) contains neutral recomputations matched to charged partners for direct comparison.
- FDA subset (`Drug_ESP_FDA.json`) includes provenance fields: `drug_name`, `fda_appl_no`, `fda_appl_type`, `fda_trade_name`, `fda_approval_date`.

HPC / ORCA production pipeline
------------------------------
- Use `QMHPC_workflow/gen_input.py` to generate ORCA input files from SMILES/geometry.
- `QMHPC_workflow/mega_job.sh` is an example driver for batching and submitting jobs to an HPC scheduler (SLURM-style). Adjust to match your cluster.
- `QMHPC_workflow/parse_mol.py` and `compute_surfce_extrema.py` parse ORCA outputs and post-process electrostatic potentials / surface extrema.

Note: ORCA (the quantum chemistry package) is required to run the DFT calculations. The repository contains example ORCA input templates (e.g. `oxcarbazepine_sample.inp`) 

Validation, benchmarks and case studies
--------------------------------------
- ESP_case_study/ contains code and data used to validate CHELPG-derived ESPs vs direct QM ESPs (example script: `esp_compare.py`, supporting data and figures included).
- The method-validation figure (functional and geometry sensitivity checks) is bundled in `fig_method_validation/` and includes a standalone plotting script plus the NumPy arrays needed to exactly reproduce the figure comparing:
  - Functional choice: B3LYP vs ωB97X-D3 (CHELPG charges, dipole, polarizability)
  - Geometry source: MMFF94 vs DFT-optimized (CHELPG charges, dipole)

  See `fig_method_validation/README.md` (and the bundled `method_validation_data.npz` and `plot_figure.py`) for details and reproducible plotting.
- For historical reference, the older folders `Func_Val/` and `Geom_Val/` were used during development; their key summary is captured in `fig_method_validation/`.

SchNet model training & evaluation
---------------------------------
- Training script: `SchNetHPC/train_schnet.py` — trains a SchNet model on the dataset features (modify paths/hyperparams inside the script).
- Evaluation: `SchNetHPC/schnet_eval.py` evaluates models / writes metrics.
- A pretrained checkpoint is included: `SchNetHPC/schnet_chelpg_best.pt` to reproduce reported results or as a starting point for fine-tuning.


Contributing, licensing & citation
---------------------------------
- License: MIT (see LICENSE).
- If you use the dataset or code, please cite the dataset DOI above and include a citation to any paper(s) associated with DrugESP (citation to be added in the repo).
- Contributions: open an issue describing your proposed change or improvement; for code contributions, open a Pull Request with tests or example reproductions where appropriate.

Contact / Support
-----------------
Open issues in this repository for bug reports, data problems, or questions about reproducing the results. For questions about dataset provenance or experimental details, reference the relevant folder[...]
