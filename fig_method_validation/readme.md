# Method Validation Figure — Reproducibility Bundle
 
This folder contains everything needed to regenerate the method-validation
figure comparing (1) DFT functional choice (B3LYP vs ωB97X-D3) and
(2) geometry source (MMFF94 vs DFT-optimized) for CHELPG partial charges,
dipole moment, and isotropic polarizability.
 
No HPC access, ORCA installation, or the full dataset is required — all
data needed to reproduce the exact figure are bundled as flat NumPy arrays.
 
## Contents
 
```
fig_method_validation/
├── method_validation_data.npz   # bundled per-atom / per-molecule arrays
├── plot_figure.py                # standalone plotting script
└── README.md                     # this file
```
 
## Requirements
 
```
python >= 3.8
numpy
matplotlib
```
 
Install with:
 
```bash
pip install numpy matplotlib
```
 
## Usage
 
```bash
python plot_figure.py
```
 
This will produce `fig_method_validation.png` in the same folder (300 dpi,
2×3 panel layout) and print the R², MAE, and RMSE statistics for every
panel to the console.
 
## What's in `method_validation_data.npz`
 
All arrays are NumPy float arrays. CHELPG arrays are **per-atom** (flattened
across all molecules in the comparison); dipole, polarizability, and RMSD
arrays are **per-molecule**.
 
| Key              | Description                                              | Units   |
|------------------|-----------------------------------------------------------|---------|
| `wb97_b3lyp_q`   | CHELPG charges, B3LYP/6-31G* reference                    | e       |
| `wb97_wb97_q`    | CHELPG charges, ωB97X-D3/6-31G*, same geometry             | e       |
| `wb97_b3lyp_dip` | Dipole moment, B3LYP/6-31G*                                | Debye   |
| `wb97_wb97_dip`  | Dipole moment, ωB97X-D3/6-31G*, same geometry               | Debye   |
| `wb97_b3lyp_pol` | Isotropic polarizability, B3LYP/6-31G*                     | a.u.    |
| `wb97_wb97_pol`  | Isotropic polarizability, ωB97X-D3/6-31G*, same geometry    | a.u.    |
| `geom_mmff_q`    | CHELPG charges, single-point B3LYP on MMFF94 geometry       | e       |
| `geom_dft_q`     | CHELPG charges, B3LYP on DFT-optimized geometry              | e       |
| `geom_mmff_dip`  | Dipole moment, single-point B3LYP on MMFF94 geometry         | Debye   |
| `geom_dft_dip`   | Dipole moment, B3LYP on DFT-optimized geometry                | Debye   |
| `rmsd_values`    | Kabsch-aligned all-atom RMSD, MMFF94 vs DFT-optimized geometry | Å    |
 
Corresponding array pairs (e.g. `wb97_b3lyp_q` and `wb97_wb97_q`) are
index-matched — element `i` in one corresponds to element `i` in the other.
 
## Methods summary
 
- **Functional comparison**: CHELPG charges, dipole moment, and
  polarizability recomputed at ωB97X-D3/6-31G* on the same MMFF94-optimized
  geometries used for the main dataset (B3LYP/6-31G*), for 612 molecules.
- **Geometry comparison**: 450 molecules underwent full geometry
  optimization at B3LYP/6-31G* (`! B3LYP 6-31G* Opt TightSCF CHELPG`, ORCA
  default optimization convergence criteria). CHELPG charges and dipole
  moment were compared between single-point calculations on the original
  MMFF94 geometry and on the resulting DFT-optimized geometry. Structural
  agreement is reported as the Kabsch-aligned all-atom RMSD between the two
  geometries.
- All CHELPG calculations used: grid spacing 0.2 Å, RMAX 2.8 Å,
  VDWRADII COSMO.
## Citation
 
If you use this figure or data, please cite the accompanying dataset paper
(DrugESP-149K).
