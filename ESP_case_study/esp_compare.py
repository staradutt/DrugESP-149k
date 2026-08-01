"""
ESP Validation Figure: Oxcarbazepine (single molecule) vs 500-molecule Pool
Author: Taradutt Pattnaik
Date: 2026

Description:
2x2 publication figure combining the single-molecule oxcarbazepine
validation with the 500-molecule aggregate validation:
  (a) Oxcarbazepine: CHELPG vs QM ESP scatter
  (b) Oxcarbazepine: ESP error histogram (CHELPG - QM), centered at zero
  (c) 500 molecules, pooled: CHELPG vs QM ESP scatter
  (d) 500 molecules, pooled: ESP error histogram (CHELPG - QM)

The error histograms (b, d) show whether the CHELPG approximation is
unbiased (centered near zero) and narrow (tight spread), which is more
informative alongside the scatter plots than a second correlation
metric (e.g. an R^2 histogram) would be.

All molecular-structure/3D-rendering code (PyVista ball-and-stick
visualization) from the original single-molecule script has been
removed -- this version is statistics/figure only.

Input:
  molecule_data.json                      (oxcarbazepine geometry + CHELPG charges)
  mol_147130.scfp.mol_147130.vpot          (oxcarbazepine ORCA QM ESP grid)
  esp_validation_500_pooled_points.npz     (500-molecule pooled surface points)
Output:
  figure_esp_validation_2x2.png (600 DPI)
"""

import numpy as np
import json
from scipy.interpolate import griddata
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

VDW_RADII = {
    'H': 1.30, 'C': 2.00, 'N': 1.83, 'O': 1.72,
    'F': 1.72, 'S': 2.16, 'Cl': 2.05,
}
BOHR = 0.529177
BOHR_TO_ANGSTROM = 0.529177249
AU_TO_KCAL = 627.509
N_POINTS_PER_ATOM = 200


def create_vdw_surface(coords, species, n_points_per_atom=N_POINTS_PER_ATOM):
    radii = np.array([VDW_RADII.get(s, 1.70) for s in species])
    surface_pts = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for center, r in zip(coords, radii):
        for k in range(n_points_per_atom):
            y = 1.0 - (k / (n_points_per_atom - 1)) * 2.0
            radius_at_y = np.sqrt(max(1.0 - y * y, 0.0))
            theta = phi * k
            pt = center + r * np.array([np.cos(theta) * radius_at_y, y, np.sin(theta) * radius_at_y])
            dists = np.linalg.norm(coords - pt, axis=1)
            if np.all(dists >= radii * 0.99):
                surface_pts.append(pt)
    return np.array(surface_pts)


def esp_chelpg(points, coords, charges):
    points_bohr = points / BOHR
    coords_bohr = coords / BOHR
    diff = points_bohr[:, np.newaxis, :] - coords_bohr[np.newaxis, :, :]
    dists = np.maximum(np.linalg.norm(diff, axis=2), 1e-6)
    return (charges[np.newaxis, :] / dists).sum(axis=1)


def read_orca_vpot(filename):
    with open(filename) as f:
        lines = f.readlines()
    n_atoms, n_points = map(int, lines[0].split())
    data_start = n_atoms + 1
    vpot_data = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) == 4:
            esp, x, y, z = map(float, parts)
            vpot_data.append([x * BOHR_TO_ANGSTROM, y * BOHR_TO_ANGSTROM,
                               z * BOHR_TO_ANGSTROM, esp])
    return np.array(vpot_data)


def main():
    print("Loading oxcarbazepine data...")
    with open("molecule_data.json") as f:
        mol_data = json.load(f)

    coords = np.array(mol_data['coords'])
    charges = np.array(mol_data['chelpg_charges'])
    species = mol_data['species']
    print(f"  {mol_data['name']} (ID: {mol_data['mol_id']}), {len(coords)} atoms")

    vdw_points = create_vdw_surface(coords, species)
    esp_chelpg_ox = esp_chelpg(vdw_points, coords, charges) * AU_TO_KCAL

    vpot_filename = f"mol_{mol_data['mol_id']}.scfp.mol_{mol_data['mol_id']}.vpot"
    vpot_data = read_orca_vpot(vpot_filename)
    esp_orca_ox = griddata(vpot_data[:, :3], vpot_data[:, 3], vdw_points,
                            method='nearest') * AU_TO_KCAL

    r_ox, _ = pearsonr(esp_chelpg_ox, esp_orca_ox)
    r2_ox = r_ox ** 2
    mae_ox = np.abs(esp_chelpg_ox - esp_orca_ox).mean()
    delta_ox = esp_chelpg_ox - esp_orca_ox

    print(f"  Surface points: {len(vdw_points)}")
    print(f"  R^2 = {r2_ox:.4f}, MAE = {mae_ox:.2f} kcal/mol")
    print(f"  Delta ESP: mean = {delta_ox.mean():.3f}, std = {delta_ox.std():.2f} kcal/mol")

    print("\nLoading 500-molecule pooled data...")
    pooled = np.load("esp_validation_500_pooled_points.npz")
    chelpg_500 = pooled["chelpg_esp_kcal"]
    orca_500 = pooled["orca_esp_kcal"]
    print(f"  {len(chelpg_500):,} pooled surface points")

    r_500, _ = pearsonr(chelpg_500, orca_500)
    r2_500 = r_500 ** 2
    mae_500 = np.abs(chelpg_500 - orca_500).mean()
    delta_500 = chelpg_500 - orca_500

    print(f"  R^2 = {r2_500:.4f}, MAE = {mae_500:.2f} kcal/mol")
    print(f"  Delta ESP: mean = {delta_500.mean():.3f}, std = {delta_500.std():.2f} kcal/mol")

    print("\nBuilding 2x2 figure...")
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    ax = axes[0, 0]
    ax.scatter(esp_orca_ox, esp_chelpg_ox, alpha=0.4, s=8, c='#2E86AB', edgecolors='none')
    lims = [min(esp_orca_ox.min(), esp_chelpg_ox.min()), max(esp_orca_ox.max(), esp_chelpg_ox.max())]
    ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5, label='x = y')
    ax.plot([], [], ' ', label=f'R² = {r2_ox:.4f}\nMAE = {mae_ox:.2f} kcal/mol')
    ax.set_xlabel('ORCA QM ESP (kcal/mol)', fontsize=13)
    ax.set_ylabel('CHELPG ESP (kcal/mol)', fontsize=13)
    ax.set_title('a', fontsize=16, fontweight='bold', loc='left')
    ax.legend(fontsize=10, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[0, 1]
    bins_ox = np.linspace(min(esp_chelpg_ox.min(), esp_orca_ox.min()),
                           max(esp_chelpg_ox.max(), esp_orca_ox.max()), 50)
    ax.hist(esp_chelpg_ox, bins=bins_ox, alpha=0.6,
            label=f'CHELPG (Vmin={esp_chelpg_ox.min():.1f}, Vmax={esp_chelpg_ox.max():.1f})',
            color='#2E86AB', edgecolor='black', linewidth=0.5)
    ax.hist(esp_orca_ox, bins=bins_ox, alpha=0.6,
            label=f'ORCA QM (Vmin={esp_orca_ox.min():.1f}, Vmax={esp_orca_ox.max():.1f})',
            color='#A23B72', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('ESP (kcal/mol)', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.set_title('b', fontsize=16, fontweight='bold', loc='left')
    ax.legend(fontsize=9, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1, 0]
    plot_n = min(len(chelpg_500), 200_000)
    idx = (np.random.choice(len(chelpg_500), plot_n, replace=False)
           if len(chelpg_500) > plot_n else np.arange(len(chelpg_500)))
    ax.scatter(orca_500[idx], chelpg_500[idx], alpha=0.15, s=3, c='#2E86AB',
               edgecolors='none', rasterized=True)
    lims = [min(orca_500.min(), chelpg_500.min()), max(orca_500.max(), chelpg_500.max())]
    ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5, label='x = y')
    ax.plot([], [], ' ', label=f'Pooled R² = {r2_500:.4f}\nMAE = {mae_500:.2f} kcal/mol')
    ax.set_xlabel('ORCA QM ESP (kcal/mol)', fontsize=13)
    ax.set_ylabel('CHELPG ESP (kcal/mol)', fontsize=13)
    ax.set_title('c', fontsize=16, fontweight='bold', loc='left')
    ax.legend(fontsize=10, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1, 1]
    ax.hist(delta_500, bins=80, color='#A23B72', edgecolor='black', linewidth=0.3,
            label=f'mean = {delta_500.mean():.2f} kcal/mol\nstd = {delta_500.std():.2f} kcal/mol')
    ax.set_xlabel(r'$\Delta$ESP = ESP$_{CHELPG}$ - ESP$_{QM}$ (kcal/mol)', fontsize=13)
    ax.set_ylabel('Number of surface points', fontsize=13)
    ax.set_title('d', fontsize=16, fontweight='bold', loc='left')
    ax.legend(fontsize=10, frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("figure_esp_validation_2x2.png", dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("\nSaved: figure_esp_validation_2x2.png (600 DPI)")


if __name__ == "__main__":
    main()
