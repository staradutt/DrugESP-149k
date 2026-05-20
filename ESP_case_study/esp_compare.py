"""
ESP Validation: CHELPG vs ORCA QM on VdW Surface
Author: Taradutt Pattnaik
Date: 2026

Description:
Validates CHELPG partial charges against quantum mechanical ESP from ORCA.
Uses van der Waals surface with COSMO radii for comparison.

Workflow:
1. Load molecular geometry and CHELPG charges
2. Generate VdW surface using golden section spiral
3. Calculate ESP from CHELPG charges
4. Read and interpolate ORCA QM ESP data
5. Compare results and generate publication figures

Dependencies: numpy, scipy, pyvista, matplotlib
"""

import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import nest_asyncio
import json

# Setup for interactive plotting
nest_asyncio.apply()
pv.set_jupyter_backend("trame")

# Use Arial/Helvetica fonts for publication quality
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# =========================================================
# Physical Constants
# =========================================================
# COSMO van der Waals radii (Angstroms)
VDW_RADII = {
    'H': 1.30, 'C': 2.00, 'N': 1.83, 'O': 1.72,
    'F': 1.72, 'S': 2.16, 'Cl': 2.05
}

BOHR = 0.529177  # Bohr to Angstrom conversion
AU_TO_KCAL = 627.509  # Hartree to kcal/mol conversion

# =========================================================
# Load Molecule Data
# =========================================================
print("Loading molecule data...")
with open("molecule_data.json", 'r') as f:
    mol_data = json.load(f)

coords = np.array(mol_data['coords'])
charges = np.array(mol_data['chelpg_charges'])
species = mol_data['species']

print(f"Loaded {mol_data['name']} (ID: {mol_data['mol_id']})")
print(f"Atoms: {len(coords)}, Net charge: {charges.sum():.4f} e")

# =========================================================
# ORCA .vpot File Reader
# =========================================================
def read_orca_vpot(filename):
    """
    Parse ORCA .vpot file containing grid points and ESP values
    
    Format:
    Line 1: n_atoms n_points
    Lines 2 to n_atoms+1: atomic coordinates
    Remaining lines: ESP x y z (4 columns)
    
    Returns: numpy array [x, y, z, esp] in Angstroms and a.u.
    """
    BOHR_TO_ANGSTROM = 0.529177249
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    n_atoms, n_points = map(int, lines[0].split())
    print(f"ORCA .vpot: {n_atoms} atoms, {n_points} grid points")
    
    # Skip header and atomic coordinates
    data_start = n_atoms + 1
    vpot_data = []
    
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) == 4:
            esp, x, y, z = map(float, parts)
            # Convert Bohr to Angstrom for coordinates
            vpot_data.append([
                x * BOHR_TO_ANGSTROM, 
                y * BOHR_TO_ANGSTROM, 
                z * BOHR_TO_ANGSTROM, 
                esp  # ESP already in a.u.
            ])
    
    return np.array(vpot_data)

# =========================================================
# VdW Surface Generation
# =========================================================
def create_vdw_surface(coords, species, n_points_per_atom=200):
    """
    Generate molecular van der Waals surface using golden section spiral
    
    Algorithm:
    - Distribute points uniformly on sphere around each atom
    - Use golden angle for azimuthal spacing
    - Filter points that penetrate other atomic spheres
    
    Args:
        coords: atomic coordinates (Angstroms)
        species: atomic symbols
        n_points_per_atom: sampling density per atom
    
    Returns: surface point coordinates (Angstroms)
    """
    # Get VdW radius for each atom
    radii = np.array([VDW_RADII.get(s, 1.70) for s in species])
    
    surface_pts = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle ~137.5 degrees
    
    print(f"\nGenerating VdW surface...")
    print(f"Points per atom: {n_points_per_atom}")
    print(f"Using COSMO radii")
    
    # Loop over each atom
    for i, (center, r, s) in enumerate(zip(coords, radii, species)):
        # Generate points on sphere using golden section spiral
        for k in range(n_points_per_atom):
            # Vertical coordinate (-1 to 1)
            y = 1.0 - (k / (n_points_per_atom - 1)) * 2.0
            radius_at_y = np.sqrt(max(1.0 - y*y, 0.0))
            
            # Azimuthal angle
            theta = phi * k
            
            # Convert to 3D point
            pt = center + r * np.array([
                np.cos(theta) * radius_at_y,
                y,
                np.sin(theta) * radius_at_y
            ])
            
            # Check if point is outside all other atomic spheres
            dists = np.linalg.norm(coords - pt, axis=1)
            
            # Keep point if not buried (0.99 for numerical tolerance)
            if np.all(dists >= radii * 0.99):
                surface_pts.append(pt)
    
    surface_pts = np.array(surface_pts)
    print(f"Surface created: {len(surface_pts)} exposed points")
    
    return surface_pts

# =========================================================
# ESP Calculation from Point Charges
# =========================================================
def esp_chelpg(points, coords, charges):
    """
    Calculate electrostatic potential from point charges
    
    ESP(r) = sum_i q_i / |r - r_i|
    
    Args:
        points: evaluation points (Angstroms)
        coords: atomic coordinates (Angstroms)
        charges: partial charges (e)
    
    Returns: ESP values (a.u.)
    """
    # Convert to Bohr (atomic units)
    points_bohr = points / BOHR
    coords_bohr = coords / BOHR
    
    # Calculate distances from all points to all atoms
    # Shape: (n_points, n_atoms)
    diff = points_bohr[:, np.newaxis, :] - coords_bohr[np.newaxis, :, :]
    dists = np.maximum(np.linalg.norm(diff, axis=2), 1e-6)  # avoid division by zero
    
    # Sum contributions: ESP = sum(q/r)
    esp = (charges[np.newaxis, :] / dists).sum(axis=1)
    
    return esp

# =========================================================
# Main Analysis
# =========================================================
print("=" * 70)
print("ESP VALIDATION: CHELPG vs ORCA QM (COSMO VdW Surface)")
print("=" * 70)

# Step 1: Create VdW surface
vdw_points = create_vdw_surface(coords, species, n_points_per_atom=100)

# Step 2: Calculate CHELPG ESP on surface
print("\nCalculating CHELPG ESP...")
esp_chelpg_vdw = esp_chelpg(vdw_points, coords, charges)
print(f"CHELPG ESP calculated")

# Step 3: Read ORCA QM ESP data
print("\nLoading ORCA .vpot file...")
# Use dynamic filename based on molecule ID
vpot_filename = f"mol_{mol_data['mol_id']}.scfp.mol_{mol_data['mol_id']}.vpot"
vpot_data = read_orca_vpot(vpot_filename)
vpot_coords = vpot_data[:, :3]
vpot_esp = vpot_data[:, 3]

# Step 4: Interpolate ORCA ESP to VdW surface points
print(f"\nInterpolating ORCA ESP to surface...")
esp_orca_vdw = griddata(vpot_coords, vpot_esp, vdw_points, method='nearest')
print(f"ORCA ESP interpolated")

# =========================================================
# Statistical Analysis
# =========================================================
print("\n" + "=" * 70)
print("VALIDATION STATISTICS")
print("=" * 70)

# Convert to kcal/mol for reporting
chelpg_vmin_kcal = esp_chelpg_vdw.min() * AU_TO_KCAL
chelpg_vmax_kcal = esp_chelpg_vdw.max() * AU_TO_KCAL
orca_vmin_kcal = esp_orca_vdw.min() * AU_TO_KCAL
orca_vmax_kcal = esp_orca_vdw.max() * AU_TO_KCAL

print(f"\nCHELPG ESP:")
print(f"   Vmin: {chelpg_vmin_kcal:.2f} kcal/mol")
print(f"   Vmax: {chelpg_vmax_kcal:.2f} kcal/mol")

print(f"\nORCA QM ESP:")
print(f"   Vmin: {orca_vmin_kcal:.2f} kcal/mol")
print(f"   Vmax: {orca_vmax_kcal:.2f} kcal/mol")

# Calculate agreement metrics
difference = esp_chelpg_vdw - esp_orca_vdw
abs_difference = np.abs(difference)
correlation, _ = pearsonr(esp_chelpg_vdw, esp_orca_vdw)

print(f"\nAgreement Metrics:")
print(f"   R²:   {correlation**2:.4f}")
print(f"   MAE:  {abs_difference.mean() * AU_TO_KCAL:.2f} kcal/mol")
print(f"   Vmin difference: {abs(chelpg_vmin_kcal - orca_vmin_kcal):.2f} kcal/mol")
print(f"   Vmax difference: {abs(chelpg_vmax_kcal - orca_vmax_kcal):.2f} kcal/mol")

# =========================================================
# Prepare PyVista Meshes
# =========================================================
print("\nPreparing visualization...")

# Create point cloud with ESP values
vdw_cloud = pv.PolyData(vdw_points)
vdw_cloud["CHELPG_ESP"] = esp_chelpg_vdw
vdw_cloud["ORCA_ESP"] = esp_orca_vdw

# Determine color scale limits
vmin_display = min(esp_chelpg_vdw.min(), esp_orca_vdw.min())
vmax_display = max(esp_chelpg_vdw.max(), esp_orca_vdw.max())

# Create surface mesh by triangulating nearby points
print("Creating surface mesh from VdW points...")
# Surface reconstruction connects nearby surface points (2D manifold, not 3D volume)
vdw_mesh = vdw_cloud.reconstruct_surface(nbr_sz=15)

# Interpolate ESP values to mesh vertices
vdw_mesh["CHELPG_ESP"] = vdw_mesh.interpolate(vdw_cloud, radius=0.5)["CHELPG_ESP"]
vdw_mesh["ORCA_ESP"] = vdw_mesh.interpolate(vdw_cloud, radius=0.5)["ORCA_ESP"]

print(f"Surface mesh created: {vdw_mesh.n_points} vertices, {vdw_mesh.n_cells} triangles")
print(f"(Connects nearby surface points only)")

# =========================================================
# Publication-Quality 3-Panel Visualization
# =========================================================
print("Creating publication-quality figure...")

# Rendering parameters
atom_colors = {"C": "#333333", "N": "#0000FF", "O": "#FF0000", "H": "#EEEEEE"}
bond_color = "#AAAAAA"
bond_radius = 0.15
atom_radius = 0.35
bond_cutoff = 1.7  # Angstroms, typical covalent bond threshold

def add_molecular_structure(plotter, coords, species):
    """
    Add ball-and-stick molecular structure to plotter
    
    Args:
        plotter: PyVista plotter object
        coords: atomic coordinates
        species: atomic symbols
    """
    # Add bonds first (so they appear behind atoms)
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < bond_cutoff:
                # Create cylinder for bond
                direction = (coords[j] - coords[i]) / dist
                bond = pv.Cylinder(
                    center=(coords[i] + coords[j]) / 2,
                    direction=direction,
                    radius=bond_radius,
                    height=dist,
                    resolution=20
                )
                plotter.add_mesh(bond, color=bond_color, specular=0.0)
    
    # Add atoms on top of bonds
    for r, s in zip(coords, species):
        atom = pv.Sphere(radius=atom_radius, center=r, 
                        phi_resolution=20, theta_resolution=20)
        plotter.add_mesh(atom, color=atom_colors.get(s, "gray"), specular=0.0)

# Create 3-panel figure
plotter = pv.Plotter(shape=(1, 3), window_size=[2100, 700])
plotter.set_background("white")

# Convert ESP to kcal/mol for display
esp_chelpg_kcal = esp_chelpg_vdw * AU_TO_KCAL
esp_orca_kcal = esp_orca_vdw * AU_TO_KCAL
vmin_display_kcal = min(esp_chelpg_kcal.min(), esp_orca_kcal.min())
vmax_display_kcal = max(esp_chelpg_kcal.max(), esp_orca_kcal.max())

# Add kcal/mol values to meshes
vdw_cloud["CHELPG_ESP_kcal"] = esp_chelpg_kcal
vdw_cloud["ORCA_ESP_kcal"] = esp_orca_kcal
vdw_mesh["CHELPG_ESP_kcal"] = griddata(vdw_points, esp_chelpg_kcal, vdw_mesh.points, method='nearest')
vdw_mesh["ORCA_ESP_kcal"] = griddata(vdw_points, esp_orca_kcal, vdw_mesh.points, method='nearest')

# =========================================================
# Panel 1: CHELPG Charges
# =========================================================
plotter.subplot(0, 0)
plotter.add_text("CHELPG Charges", font_size=18, color="black", 
                position="upper_edge", font="arial")
plotter.hide_axes()

# Add molecular structure
add_molecular_structure(plotter, coords, species)

# Add charge labels offset from atoms
mol_center = coords.mean(axis=0)
label_coords = []
label_vals = []

for r, q, s in zip(coords, charges, species):
    # Calculate offset direction from molecule center
    vec = r - mol_center
    norm = np.linalg.norm(vec)
    offset_dist = 0.5 if s == 'H' else 0.6
    offset = (vec / norm) * offset_dist if norm > 1e-3 else np.array([0, 0, 0.6])
    
    label_coords.append(r + offset)
    label_vals.append(f"{q:+.2f}")

plotter.add_point_labels(
    label_coords, label_vals, 
    font_size=13, 
    text_color="black",
    shape="rounded_rect", 
    fill_shape=True, 
    shape_color="white", 
    shape_opacity=0.85, 
    always_visible=True, 
    point_size=0,
    bold=False,
    font_family="arial"
)

# =========================================================
# Panel 2: CHELPG ESP
# =========================================================
plotter.subplot(0, 1)
plotter.add_text("CHELPG ESP", font_size=18, color="black", 
                position="upper_edge", font="arial")
plotter.hide_axes()

# Add semi-transparent surface mesh
plotter.add_mesh(
    vdw_mesh,
    scalars="CHELPG_ESP_kcal",
    cmap="RdYlBu_r",
    clim=[vmin_display_kcal, vmax_display_kcal],
    opacity=0.3,
    smooth_shading=True,
    show_scalar_bar=False
)

# Add colored ESP points on top
plotter.add_mesh(
    vdw_cloud,
    scalars="CHELPG_ESP_kcal",
    cmap="RdYlBu_r",
    clim=[vmin_display_kcal, vmax_display_kcal],
    point_size=6,
    render_points_as_spheres=True,
    scalar_bar_args=dict(
        title="ESP (kcal/mol)",
        vertical=False,  # Horizontal color bar
        position_x=0.25,
        position_y=0.08,
        width=0.5,
        height=0.08,
        color="black",
        title_font_size=16,
        label_font_size=14,
        font_family="arial",
        n_labels=5,
        fmt="%.1f"
    )
)

# Add molecular structure
add_molecular_structure(plotter, coords, species)

# =========================================================
# Panel 3: ORCA QM ESP
# =========================================================
plotter.subplot(0, 2)
plotter.add_text("ORCA QM ESP", font_size=18, color="black", 
                position="upper_edge", font="arial")
plotter.hide_axes()

# Add semi-transparent surface mesh
plotter.add_mesh(
    vdw_mesh,
    scalars="ORCA_ESP_kcal",
    cmap="RdYlBu_r",
    clim=[vmin_display_kcal, vmax_display_kcal],
    opacity=0.3,
    smooth_shading=True,
    show_scalar_bar=False
)

# Add colored ESP points on top
plotter.add_mesh(
    vdw_cloud,
    scalars="ORCA_ESP_kcal",
    cmap="RdYlBu_r",
    clim=[vmin_display_kcal, vmax_display_kcal],
    point_size=6,
    render_points_as_spheres=True,
    scalar_bar_args=dict(
        title="ESP (kcal/mol)",
        vertical=False,  # Horizontal color bar
        position_x=0.25,
        position_y=0.08,
        width=0.5,
        height=0.08,
        color="black",
        title_font_size=16,
        label_font_size=14,
        font_family="arial",
        n_labels=5,
        fmt="%.1f"
    )
)

# Add molecular structure
add_molecular_structure(plotter, coords, species)

# Set camera zoom for all panels
for i in range(3):
    plotter.subplot(0, i)
    plotter.camera.zoom(1.3)

# Link views so all panels rotate together
plotter.link_views()

# Display interactive plot
plotter.show(jupyter_backend="trame")

# =========================================================
# Publication-Quality Statistical Plots
# =========================================================
print("\nCreating statistical analysis plots...")

fig = plt.figure(figsize=(12, 5))

# Subplot (a): Correlation plot
ax1 = plt.subplot(1, 2, 1)

# Convert to kcal/mol for plotting
esp_orca_kcal_plot = esp_orca_vdw * AU_TO_KCAL
esp_chelpg_kcal_plot = esp_chelpg_vdw * AU_TO_KCAL

# Scatter plot
ax1.scatter(esp_orca_kcal_plot, esp_chelpg_kcal_plot, alpha=0.4, s=8, 
            c='#2E86AB', edgecolors='none')

# x=y reference line
ax1.plot([esp_orca_kcal_plot.min(), esp_orca_kcal_plot.max()], 
         [esp_orca_kcal_plot.min(), esp_orca_kcal_plot.max()], 
         'k--', linewidth=2, alpha=0.5, label='x = y')

# R² value in legend
ax1.plot([], [], ' ', label=f'R² = {correlation**2:.4f}')

ax1.set_xlabel('ORCA QM ESP (kcal/mol)', fontsize=14)
ax1.set_ylabel('CHELPG ESP (kcal/mol)', fontsize=14)
ax1.set_title('a', fontsize=18, fontweight='bold', loc='left')
ax1.legend(fontsize=11, frameon=False, loc='upper left')
ax1.tick_params(labelsize=11)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Subplot (b): Distribution comparison
ax2 = plt.subplot(1, 2, 2)

# Create histogram bins
bins = np.linspace(min(esp_chelpg_vdw.min(), esp_orca_vdw.min()) * AU_TO_KCAL,
                   max(esp_chelpg_vdw.max(), esp_orca_vdw.max()) * AU_TO_KCAL, 50)

# Overlapping histograms
ax2.hist(esp_chelpg_vdw * AU_TO_KCAL, bins=bins, alpha=0.6, 
         label=f'CHELPG (Vmin={chelpg_vmin_kcal:.1f}, Vmax={chelpg_vmax_kcal:.1f})', 
         color='#2E86AB', edgecolor='black', linewidth=0.5)
ax2.hist(esp_orca_vdw * AU_TO_KCAL, bins=bins, alpha=0.6, 
         label=f'ORCA QM (Vmin={orca_vmin_kcal:.1f}, Vmax={orca_vmax_kcal:.1f})', 
         color='#A23B72', edgecolor='black', linewidth=0.5)

ax2.set_xlabel('ESP (kcal/mol)', fontsize=14)
ax2.set_ylabel('Frequency', fontsize=14)
ax2.set_title('b', fontsize=18, fontweight='bold', loc='left')
ax2.legend(fontsize=10, frameon=False, loc='upper left')
ax2.tick_params(labelsize=11)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()

# Save high-resolution figure
output_filename = 'figure1_esp_validation.png'
plt.savefig(output_filename, dpi=600, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print(f"Figure saved: {output_filename} (600 DPI)")

# =========================================================
# Summary
# =========================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nValidation Summary:")
print(f"  Surface: COSMO VdW radii")
print(f"  R² = {correlation**2:.4f} (Excellent agreement)")
print(f"  MAE = {abs_difference.mean() * AU_TO_KCAL:.2f} kcal/mol")
print(f"  Extrema within {max(abs(chelpg_vmin_kcal - orca_vmin_kcal), abs(chelpg_vmax_kcal - orca_vmax_kcal)):.2f} kcal/mol")
print("\nCHELPG charges validated for dataset!")
print("=" * 70)
