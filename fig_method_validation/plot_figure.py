"""
plot_figure.py
--------------
Regenerates the method-validation figure (functional choice and geometry
optimization effects on CHELPG charges, dipole moment, and polarizability)
from the bundled data file `method_validation_data.npz`.

Usage:
    python plot_figure.py

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# --- FONT CONFIGURATION (Helvetica-compatible) ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / 'method_validation_data.npz'
OUT_FILE  = HERE / 'fig_method_validation.png'


def panel_label(ax, letter):
    ax.text(0.02, 1.02, letter, transform=ax.transAxes, fontsize=14,
             fontweight='bold', va='bottom', ha='left')


def calc_stats(x, y):
    x = np.asarray(x); y = np.asarray(y)
    mae  = np.mean(np.abs(x - y))
    rmse = np.sqrt(np.mean((x - y) ** 2))
    r    = np.corrcoef(x, y)[0, 1]
    r2   = r ** 2
    return mae, rmse, r, r2


def scatter_panel(ax, x, y, xlabel, ylabel, color, unit,
                   max_points=50000, force_zero_origin=False, seed=0):
    rng = np.random.default_rng(seed)
    n_plot = min(max_points, len(x))
    idx = rng.choice(len(x), n_plot, replace=False) if len(x) > n_plot else np.arange(len(x))
    ax.scatter(np.asarray(x)[idx], np.asarray(y)[idx],
               alpha=0.15 if len(x) < 5000 else 0.05,
               s=8 if len(x) < 5000 else 1, color=color, rasterized=True)

    mae, rmse, r, r2 = calc_stats(x, y)

    if force_zero_origin:
        lo = 0
        hi = max(np.max(x), np.max(y)) * 1.05
    else:
        mn = min(np.min(x), np.min(y))
        lo = mn * 1.05 if mn < 0 else mn * 0.95
        hi = max(np.max(x), np.max(y)) * 1.05

    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.2)

    textstr = f'R\u00b2 = {r2:.4f}\nMAE = {mae:.4f} {unit}'
    ax.text(0.04, 0.96, textstr, transform=ax.transAxes, fontsize=10,
            va='top', ha='left')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_aspect('equal')

    return mae, rmse, r, r2, len(x)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. Make sure method_validation_data.npz "
            f"is in the same folder as this script."
        )

    d = np.load(DATA_FILE)

    wb97_b3lyp_q   = d['wb97_b3lyp_q']
    wb97_wb97_q    = d['wb97_wb97_q']
    wb97_b3lyp_dip = d['wb97_b3lyp_dip']
    wb97_wb97_dip  = d['wb97_wb97_dip']
    wb97_b3lyp_pol = d['wb97_b3lyp_pol']
    wb97_wb97_pol  = d['wb97_wb97_pol']
    geom_mmff_q    = d['geom_mmff_q']
    geom_dft_q     = d['geom_dft_q']
    geom_mmff_dip  = d['geom_mmff_dip']
    geom_dft_dip   = d['geom_dft_dip']
    rmsd_values    = d['rmsd_values']

    fig = plt.figure(figsize=(19, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # (a) CHELPG B3LYP vs wB97X-D3
    ax = fig.add_subplot(gs[0, 0])
    scatter_panel(ax, wb97_b3lyp_q, wb97_wb97_q,
                  'B3LYP CHELPG charge (e)', '\u03c9B97X-D3 CHELPG charge (e)',
                  '#3498db', 'e')
    panel_label(ax, 'a')

    # (b) Dipole moment B3LYP vs wB97X-D3
    ax = fig.add_subplot(gs[0, 1])
    scatter_panel(ax, wb97_b3lyp_dip, wb97_wb97_dip,
                  'B3LYP dipole moment (Debye)', '\u03c9B97X-D3 dipole moment (Debye)',
                  '#2ecc71', 'D', force_zero_origin=True)
    panel_label(ax, 'b')

    # (c) Polarizability B3LYP vs wB97X-D3
    ax = fig.add_subplot(gs[0, 2])
    scatter_panel(ax, wb97_b3lyp_pol, wb97_wb97_pol,
                  'B3LYP polarizability (a.u.)', '\u03c9B97X-D3 polarizability (a.u.)',
                  '#e74c3c', 'a.u.')
    panel_label(ax, 'c')

    # (d) CHELPG MMFF94 vs DFT-optimized
    ax = fig.add_subplot(gs[1, 0])
    scatter_panel(ax, geom_mmff_q, geom_dft_q,
                  'MMFF94 geometry CHELPG charge (e)', 'DFT-optimized CHELPG charge (e)',
                  '#9b59b6', 'e')
    panel_label(ax, 'd')

    # (e) RMSD distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(rmsd_values, bins=40, color='#f39c12', alpha=0.85, edgecolor='white')
    ax.axvline(rmsd_values.mean(), color='black', linewidth=2,
               label=f'Mean = {rmsd_values.mean():.3f} \u00c5')
    ax.set_xlabel('All-atom RMSD (\u00c5)', fontsize=11)
    ax.set_ylabel('Number of molecules', fontsize=11)
    ax.legend(fontsize=9, loc='upper right', frameon=False)
    panel_label(ax, 'e')

    # (f) Dipole moment MMFF94 vs DFT-optimized
    ax = fig.add_subplot(gs[1, 2])
    scatter_panel(ax, geom_mmff_dip, geom_dft_dip,
                  'MMFF94 geometry dipole moment (Debye)', 'DFT-optimized dipole moment (Debye)',
                  '#1abc9c', 'D', force_zero_origin=True)
    panel_label(ax, 'f')

    fig.savefig(OUT_FILE, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Figure saved to: {OUT_FILE}")

    # Print stats for reference
    print("\n" + "=" * 60)
    print("Panel statistics")
    print("=" * 60)
    for label, x, y, unit in [
        ('(a) CHELPG, B3LYP vs wB97X-D3',   wb97_b3lyp_q,   wb97_wb97_q,   'e'),
        ('(b) Dipole, B3LYP vs wB97X-D3',   wb97_b3lyp_dip, wb97_wb97_dip, 'D'),
        ('(c) Polarizability, B3LYP vs wB97X-D3', wb97_b3lyp_pol, wb97_wb97_pol, 'a.u.'),
        ('(d) CHELPG, MMFF94 vs DFT-opt',   geom_mmff_q,    geom_dft_q,    'e'),
        ('(f) Dipole, MMFF94 vs DFT-opt',   geom_mmff_dip,  geom_dft_dip,  'D'),
    ]:
        mae, rmse, r, r2 = calc_stats(x, y)
        print(f"{label:40s} n={len(x):>7,}  MAE={mae:.4f} {unit:5s} RMSE={rmse:.4f} {unit:5s} R2={r2:.4f}")

    print(f"(e) RMSD (MMFF94 vs DFT-opt geometry){' '*4}"
          f"n={len(rmsd_values):>7,}  mean={rmsd_values.mean():.4f} \u00c5  "
          f"median={np.median(rmsd_values):.4f} \u00c5")


if __name__ == '__main__':
    main()
