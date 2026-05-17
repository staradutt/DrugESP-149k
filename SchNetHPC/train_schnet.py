#!/usr/bin/env python3

"""
SchNet Training for CHELPG Charge Prediction

Author: Taradutt Pattnaik
Date: 2026

Description:
Trains a SchNet-based GGNN to predict
per-atom CHELPG partial charges from molecule geometry

The model uses SchNet atom embeddings followed by a custom
atomwise regression head. Charge conservation is enforced
during prediction by enforcing constraint: the sum of atomic charges
within each molecule be zero.

Inputs:
- training_data/train.json (created by test_train_split.py )
- training_data/val.json
- training_data/test.json

Outputs:
- models/schnet_chelpg_best.pt
- models/schnet_epoch*.pt
- logs/schnet_train_detail.log

Model Features:
- schNet base
- Atomwise charge regression
- charge conservationconstraint 
- MAE optimization with L1 loss

"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from torch_scatter import scatter


# Paths
BASE = Path.cwd()

TRAIN_DIR = BASE / "training_data"
MODEL_DIR = BASE / "models"
LOG_DIR   = BASE / "logs"

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# Logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "schnet_train_detail.log")
    ]
)

log = logging.getLogger(__name__)


# Reproducibility



torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# Constants, only these elements in the Drug ESP dataset


ELEM_TO_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "S": 16,
    "Cl": 17
}

CUTOFF     = 5.0
BATCH_SIZE =32
N_EPOCHS   =   200
LR         =  1e-3


# Dataset Conversion


def mol_to_data(mol):

    coords = torch.tensor(
        mol["coords"],
        dtype=torch.float32
    )

    charges = torch.tensor(
        mol["chelpg_charges"],
        dtype=torch.float32
    )

    z = torch.tensor(
        [ELEM_TO_Z[s] for s in mol["species"]],
        dtype=torch.long
    )

    return Data(
        z=z,
        pos=coords,
        y=charges
    )


# Data Loading ...


def load_split(name):

    log.info(f"Loading {name} split...")

    with open(TRAIN_DIR / f"{name}.json") as f:
        mols = json.load(f)

    data_list = []

    for mol in mols:

        try:
            
            data_list.append(mol_to_data(mol))

        except Exception as e:
            log.warning(
                f"Skipping molecule {mol.get('mol_id')} "
                f"due to error: {e}"
            )

    log.info(f"  {name}: {len(data_list):,} molecules")

    return data_list


# SchNet Charge Model


class SchNetCharge(nn.Module):
    """
    SchNet with atomwise charge prediction head.
    """

    def __init__(
        self,
        hidden_channels=256,
        num_interactions=6
    ):

        super().__init__()

        self.schnet = SchNet(
            hidden_channels   = hidden_channels,
            num_filters       = hidden_channels,
            num_interactions  = num_interactions,
            num_gaussians     = 50,
            cutoff            = CUTOFF,
            max_num_neighbors = 32,
        )

        # Atomwise charge regression head
        self.charge_head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, z, pos, batch_idx):

        # Initial atomic embeddings
        h = self.schnet.embedding(z)

        # Neighbor graph
        edge_index, edge_weight = (
            self.schnet.interaction_graph(pos, batch_idx)
        )

        edge_attr = self.schnet.distance_expansion(edge_weight)

        
        # Interaction blocks
        for interaction in self.schnet.interactions:
            h = h + interaction(
                h,
                edge_index,
                edge_weight,
                edge_attr
            )

        # Atomwise charge prediction
        charges = self.charge_head(h).squeeze(-1)

        # Enforce charge conservation:
        # sum (q_i) = 0 for each molecule
        mol_sum = scatter(
            charges,
            batch_idx,
            dim=0,
            reduce="sum"
        )

        n_atoms = scatter(
            torch.ones_like(charges),
            batch_idx,
            dim=0,
            reduce="sum"
        )

        charges = (
            charges
            - mol_sum[batch_idx] / n_atoms[batch_idx]
        )

        return charges


# Evaluation


def evaluate(model, loader, device):

    model.eval()

    total_mae = 0.0
    total_n   = 0

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            pred = model(
                batch.z,
                batch.pos,
                batch.batch
            )

            total_mae += (
                (pred - batch.y).abs().sum().item()
            )

            total_n += batch.y.numel()

    return total_mae / total_n


# Main Training
def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    log.info(f"Device: {device}")

    if torch.cuda.is_available():
        log.info(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # Load datasets
    train_data = load_split("train")
    val_data   = load_split("val")
    test_data  = load_split("test")

    # Data loaders
    train_loader = DataLoader(
        train_data,
        batch_size =BATCH_SIZE,
        shuffle=True,
        num_workers =4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size =64,
        shuffle =False,
        num_workers=4,
        pin_memory=True
    )

    # Model
    model = SchNetCharge().to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    log.info(
        f"SchNetCharge parameters: {n_params:,}"
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=8,
        factor=0.5,
        min_lr=1e-6
    )

    best_val_mae = float("inf")

    log.info("Starting training...")

 
    # Training Loop
    for epoch in range(1, N_EPOCHS + 1):

        model.train()

        train_mae = 0.0
        train_n   = 0

        t0 = time.time()

        for batch in train_loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            pred = model(
                batch.z,
                batch.pos,
                batch.batch
            )

            loss = F.l1_loss(pred, batch.y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                10.0
            )

            optimizer.step()

            train_mae += (
                loss.item() * batch.y.numel()
            )

            train_n += batch.y.numel()

        train_mae /= train_n

        val_mae = evaluate(
            model,
            val_loader,
            device
        )

        scheduler.step(val_mae)

        t1 = time.time()

        # Save best model
        if val_mae < best_val_mae:

            best_val_mae = val_mae

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_mae": val_mae,
                },
                MODEL_DIR / "schnet_chelpg_best.pt"
            )

            flag = " <- best"

        else:
            flag = ""

        log.info(
            f"Epoch {epoch:3d}/{N_EPOCHS} | "
            f"Train: {train_mae:.4f} e | "
            f"Val: {val_mae:.4f} e | "
            f"Best: {best_val_mae:.4f} e | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
            f"{t1-t0:.0f}s{flag}"
        )

        # Periodic checkpoints
        if epoch % 10 == 0:

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                },
                MODEL_DIR / f"schnet_epoch{epoch}.pt"
            )

        # Early stopping
        if optimizer.param_groups[0]["lr"] <= 1e-6:

            log.info(
                f"Early stopping at epoch {epoch}"
            )

            break


    # Final Test Evaluation
    ckpt = torch.load(
        MODEL_DIR / "schnet_chelpg_best.pt", #saves the best model for later use (we provide a copy as well)
        map_location=device
    )

    model.load_state_dict(ckpt["model"])

    test_mae = evaluate(
        model,
        test_loader,
        device
    )

    log.info(f"Test MAE: {test_mae:.4f} e")
    log.info(f"Best val MAE: {best_val_mae:.4f} e")
    log.info("Training complete.")


if __name__ == "__main__":
    main()
