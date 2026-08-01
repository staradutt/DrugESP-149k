# Using DrugESP in Python

Quick reference for loading the dataset and pulling individual molecules,
covering all four dataset files.

## Loading a dataset file

```python
import json

with open("Drug_ESP_149k.json") as f:
    dataset = json.load(f)

print(f"{len(dataset):,} molecules")
print(dataset[0].keys())
```

Same pattern works for `Drug_ESP_charged.json`, `Drug_ESP_protonated_neutral.json`,
and `Drug_ESP_FDA.json` -- just swap the filename. See `dataset_metadata.json`
for the full field reference (types, units, descriptions), including which
fields are common to all four files versus specific to one.

---

## Main dataset (Drug_ESP_149k.json)

### Look up a molecule by mol_id

```python
with open("Drug_ESP_149k.json") as f:
    main_dataset = json.load(f)

main_by_id = {e["mol_id"]: e for e in main_dataset}
mol = main_by_id[0]

print(mol["smiles"])
print(mol["species"])
print(mol["coords"])
print(mol["chelpg_charges"])
```

### Look up a molecule by SMILES

Use canonical SMILES matching via RDKit, since string comparison alone can
miss equivalent SMILES written differently.

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

### Load geometry into RDKit

```python
def to_rdkit_mol(entry):
    m = Chem.MolFromSmiles(entry["smiles"])
    m = Chem.AddHs(m)
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, (x, y, z) in enumerate(entry["coords"]):
        conf.SetAtomPosition(i, (x, y, z))
    m.AddConformer(conf)
    return m

rdkit_mol = to_rdkit_mol(mol)
```

Note: atom order in `coords`/`species` matches the order RDKit assigns after
`AddHs()` on the stored SMILES, not necessarily the SMILES atom order before
hydrogens are added. This applies to all four dataset files -- charged
species' SMILES already encode formal charge (e.g. `[N-]`, `[NH+]`), so
`Chem.MolFromSmiles()` picks up the correct charge automatically without any
extra handling.

### Filter by property

```python
subset = [e for e in main_dataset if e["mw"] < 300]
high_dipole = [e for e in main_dataset if e["dipole_debye"] > 8.0]
by_gap = sorted(main_dataset, key=lambda e: e["gap_ev"])
```

### Extract a field across the whole dataset

```python
import numpy as np

mw_values = np.array([e["mw"] for e in main_dataset])
print(f"Mean MW: {mw_values.mean():.1f} Da")
```

---

## Charged dataset (Drug_ESP_charged.json)

Each entry has `charge` (+1 or -1) and `parent_mol_id` (the `mol_id` of the
neutral parent in `Drug_ESP_149k.json` this ion was generated from).

### Look up by mol_id

```python
with open("Drug_ESP_charged.json") as f:
    charged_dataset = json.load(f)

charged_by_id = {e["mol_id"]: e for e in charged_dataset}
ion = charged_by_id["pos1_000000"]

print(ion["smiles"], ion["charge"], ion["parent_mol_id"])
```

### Find all ions generated from a given parent molecule

A single parent can have multiple ionization-site variants, so this returns
a list, not a single entry.

```python
def ions_for_parent(charged_dataset, parent_mol_id):
    return [e for e in charged_dataset if e["parent_mol_id"] == parent_mol_id]

variants = ions_for_parent(charged_dataset, 19)
for v in variants:
    print(v["mol_id"], v["charge"], v["smiles"])
```

### Split into cations and anions

```python
cations = [e for e in charged_dataset if e["charge"] == 1]
anions = [e for e in charged_dataset if e["charge"] == -1]
print(f"{len(cations):,} cations, {len(anions):,} anions")
```

---

## Protonated-neutral dataset (Drug_ESP_protonated_neutral.json)

Neutral parent molecules recomputed at the same level of theory as the
charged dataset (B3LYP/6-31+G*/CPCM(water)), for direct charge-state
comparison. Each entry also carries `parent_mol_id`.

### Build a matched neutral/+1/-1 triplet

```python
def get_triplet(parent_mol_id, protonated_neutral_dataset, charged_dataset):
    neutral = next((e for e in protonated_neutral_dataset
                     if e["parent_mol_id"] == parent_mol_id), None)
    ions = ions_for_parent(charged_dataset, parent_mol_id)
    pos1 = next((e for e in ions if e["charge"] == 1), None)
    neg1 = next((e for e in ions if e["charge"] == -1), None)
    return {"neutral": neutral, "pos1": pos1, "neg1": neg1}

with open("Drug_ESP_protonated_neutral.json") as f:
    protonated_neutral_dataset = json.load(f)

triplet = get_triplet(19, protonated_neutral_dataset, charged_dataset)
for state, entry in triplet.items():
    if entry:
        print(f"{state}: {entry['smiles']} (charge {entry['charge']:+d})")
    else:
        print(f"{state}: not available for this parent")
```

Not every parent has all three states available -- some only have a +1 or
-1 partner, or (for the protonated-neutral set specifically) no charged
partner survived QC filtering. Always check for `None` before using a
result.

---

## FDA dataset (Drug_ESP_FDA.json)

Each entry carries FDA Orange Book provenance: `drug_name`, `fda_appl_no`,
`fda_appl_type`, `fda_trade_name`, `fda_approval_date`.

### Look up by drug name

```python
with open("Drug_ESP_FDA.json") as f:
    fda_dataset = json.load(f)

def find_by_drug_name(fda_dataset, name):
    name = name.lower()
    return [e for e in fda_dataset if e.get("drug_name", "").lower() == name]

matches = find_by_drug_name(fda_dataset, "emtricitabine")
```

### Look up by FDA application number

```python
fda_by_appl_no = {e["fda_appl_no"]: e for e in fda_dataset if e.get("fda_appl_no")}
drug = fda_by_appl_no["021500"]
print(drug["drug_name"], drug["fda_trade_name"])
```

### Filter by application type

```python
# 'N' = New Drug Application, 'A' = Abbreviated New Drug Application (generic)
ndas = [e for e in fda_dataset if e.get("fda_appl_type") == "N"]
andas = [e for e in fda_dataset if e.get("fda_appl_type") == "A"]
```
