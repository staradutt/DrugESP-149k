# ESP Case Study

Validates CHELPG-derived electrostatic potential against ab initio QM ESP,
for a single representative molecule (oxcarbazepine) and pooled across 500
randomly sampled molecules from DrugESP-149K.

## Files

- `esp_compare.py` -- run this
- `molecule_data.json` -- oxcarbazepine geometry and CHELPG charges
- `mol_148052.scfp.mol_148052.vpot` -- oxcarbazepine's ORCA QM ESP grid
- `esp_validation_500_pooled_points.npz` -- pooled CHELPG/QM ESP values across 500 molecules

## Usage

```bash
pip install numpy scipy matplotlib
python3 esp_compare.py
```

Produces `figure_esp_validation_2x2.png`: (a) oxcarbazepine CHELPG-vs-QM
scatter, (b) oxcarbazepine ESP distribution, (c) pooled 500-molecule
scatter, (d) pooled ESP error distribution.
