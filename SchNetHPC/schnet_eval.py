#!/usr/bin/env python3
"""
SchNet Evaluation for CHELPG Charge and Dipole Prediction
Author: Taradutt Pattnaik
Date: 2026

Description:
Evaluates a trained SchNet model on the test set

Calculates:
- per -  atom CHELPG charge prediction accuracy (MAE, R2correlation)
- Per-element charge errors 
- Dipole moments (magnitude + components)
- parity plots etc.

Inputs:
- test.json (as created by test_train_split.py)
- schnet_chelpg_best.pt (trained model ...assuming training is complete..also provided)

Outputs:
- SchNet_Benchmark_Extended.png  ( figure with all theh analysis )

Extra note:
- send it off with the same jobscript as the one used to train the schnet model
"""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
import torch.nn as nn
from torch_geometric.nn import SchNet


# Plot configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False



# Paths (change as per requirement )

BASE      = Path("data_root")
MODEL_DIR = BASE / "models"
DATA_DIR  = BASE / "training_data"
FIGDIR    = BASE / "figures"
FIGDIR.mkdir(exist_ok=True)


# Constants

ELEM_TO_Z = {"H":1,"C":6,"N":7,"O":8,"F":9,"S":16,"Cl":17}
CUTOFF    = 5.0

# Unit conversions
CONV = 4.80320427   # e.A -> Debye
AU2D = 2.541746     # atomic units -> Debye


# Model definition
# similar to training ....copy paste 
class SchNetCharge(nn.Module):
    def __init__(self, hidden_channels=256, num_interactions=6):
        super().__init__()

        
        self.schnet = SchNet(
            hidden_channels  = hidden_channels,
            num_filters      = hidden_channels,
            num_interactions = num_interactions,
            num_gaussians    = 50,
            cutoff           = CUTOFF,
            max_num_neighbors= 32,
        )

        self.charge_head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        
        )

    def forward(self, z, pos, batch_idx):
        h = self.schnet.embedding(z)
        edge_index, edge_weight = self.schnet.interaction_graph(pos, batch_idx)
        edge_attr = self.schnet.distance_expansion(edge_weight)

        
        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        charges = self.charge_head(h).squeeze(-1)

        # enforce charge neutrality per molecule
        mol_sum = scatter(charges, batch_idx, dim=0, reduce="sum")
        n_atoms = scatter(torch.ones_like(charges), batch_idx, dim=0, reduce="sum")
        charges = charges - mol_sum[batch_idx] / n_atoms[batch_idx]

        return charges


# Data utilities
def mol_to_data(mol):
    coords  = torch.tensor(mol["coords"], dtype=torch.float32)
    
    charges = torch.tensor(mol["chelpg_charges"], dtype=torch.float32)
    z       = torch.tensor([ELEM_TO_Z[s] for s in mol["species"]],
                            dtype=torch.long)

    return Data(
        z=z,
        pos=coords,
        y=charges,
        mol_id=mol["mol_id"]
    )


# Load test set
print("Loading test set...")

with open(DATA_DIR / "test.json") as f:
    test_mols = json.load(f)

print(f"Test molecules: {len(test_mols):,}")

mol_lookup = {m["mol_id"]: m for m in test_mols}

test_data = [mol_to_data(m) for m in test_mols]
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = SchNetCharge().to(device)
ckpt = torch.load(MODEL_DIR / "schnet_chelpg_best.pt", map_location=device)

model.load_state_dict(ckpt["model"])
model.eval()


# Eval...

print("\nEvaluating model..")

all_errors, all_pred, all_true = [], [], []
elem_errors = {e: [] for e in ELEM_TO_Z.keys()}

dip_pred_all = []
dip_qm_all   = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred  = model(batch.z, batch.pos, batch.batch)

        errors = (pred - batch.y).abs()

        all_errors.extend(errors.cpu().numpy())
        all_pred.extend(pred.cpu().numpy())
        all_true.extend(batch.y.cpu().numpy())

        # per-element errors
        z_cpu = batch.z.cpu().numpy()
        err_cpu = errors.cpu().numpy()

        for z_val, err in zip(z_cpu, err_cpu):
            for elem, z in ELEM_TO_Z.items():
                if z == z_val:
                    elem_errors[elem].append(err)
                    break

        # dipole reconstruction sum(q_i*pos)
        dip = pred[:, None] * batch.pos
        dip_mol = scatter(dip, batch.batch, dim=0, reduce="sum")
        dip_mol = dip_mol * CONV

        dip_pred_all.append(dip_mol.cpu().numpy())

        qm = []
        for mol_id in batch.mol_id:
            mol = mol_lookup[int(mol_id)]
            qm.append(np.array(mol["dipole_vector"]) * AU2D)

        dip_qm_all.append(np.array(qm))


# Final arrays
all_errors = np.array(all_errors)
all_pred   = np.array(all_pred)
all_true   = np.array(all_true)

dip_pred_all = np.vstack(dip_pred_all)
dip_qm_all   = np.vstack(dip_qm_all)


# Plots

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# a.. parity
ax = axes[0,0]
idx = np.random.choice(len(all_true), min(50000, len(all_true)), replace=False)
ax.scatter(all_true[idx], all_pred[idx], alpha=0.08, s=1.5)
ax.plot([-2.5,2.5],[-2.5,2.5],'r--')

ax.text(0.05, 0.95,
        f"R = {np.corrcoef(all_true, all_pred)[0,1]:.4f}\n"
        f"MAE = {all_errors.mean():.4f} e",
        transform=ax.transAxes, va='top',
        bbox=dict(facecolor='white', alpha=0.7))

ax.set_title("(a) Charge Parity")
ax.set_xlabel("DFT CHELPG (e)")
ax.set_ylabel("Predicted (e)")

# b.. per element MAE
ax = axes[0,1]
labels = [e for e in ELEM_TO_Z if elem_errors[e]]
means  = [np.mean(elem_errors[e]) for e in labels]

ax.bar(labels, means)
ax.set_title("(b) Per-element MAE")

#c.. dipole magnitude
ax = axes[0,2]
pred_mag = np.linalg.norm(dip_pred_all, axis=1)
true_mag = np.linalg.norm(dip_qm_all, axis=1)

ax.scatter(true_mag, pred_mag, s=5, alpha=0.1)

ax.set_title("(c) Dipole Magnitude")

# (d,e,f)... dipole components lol
for i in range(3):
    ax = axes[1,i]
    ax.scatter(dip_qm_all[:,i], dip_pred_all[:,i], s=5, alpha=0.1)
    ax.set_title(f"Dipole {['x','y','z'][i]}")

plt.tight_layout()

out_img = FIGDIR / "SchNet_Benchmark_Extended.png"
plt.savefig(out_img, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_img}")
