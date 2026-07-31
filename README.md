# DrugESP-149K

Code accompanying the DrugESP-149K dataset: a quantum chemical dataset of
electrostatic and electronic properties for drug-like molecules, plus
protonated/deprotonated and FDA-approved reference extensions.

Dataset (JSON files) hosted on Zenodo: https://doi.org/10.5281/zenodo.21709579

## Repository structure

- `moses_pipeline/` -- molecule collection, filtering, and geometry generation for the core dataset
- `QMHPC_workflow/` -- ORCA/HPC production pipeline (input generation, SLURM job, parsing, QC, Vmin/Vmax)
- `demo_protonation_example/` -- example protonation-state enumeration and ORCA input generation
- `demo_fda_example/` -- example ORCA input generation for the FDA-approved reference set
- `ESP_case_study/` -- CHELPG-vs-QM electrostatic potential validation (oxcarbazepine & 500 molecules)
- `Func_Val/` -- functional (B3LYP vs wB97X-D3) sensitivity validation
- `Geom_Val/` -- geometry (MMFF94 vs DFT-optimized) sensitivity validation
- `SchNetHPC/` -- SchNet GNN training, evaluation, and trained model
- `DrugESP_149k_moses_mapping.json` -- mol_id to original MOSES SMILES/index/split

## License

MIT License (see `LICENSE`).

## Citation

If you use this dataset or code, please cite:

[citation to be added]
