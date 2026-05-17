"""
ESP Surface Extrema (Vmin/Vmax) Calculation

Author: Taradutt Pattnaik 
Date: 2026

Description:
Computes approximate electrostatic potential extrema
(Vmin / Vmax) on the molecular van der Waals surface
using CHELPG atomic charges.

Method used:
- Generate surface points using a Golden Section Spiral
- Remove buried/internal points
- Compute electrostatic potential at exposed points
- Store minimum and maximum ESP values

Input:
- DrugESP_149k_compact.json

Output:
- Updated DrugESP_149k_compact.json
  containing:
    - vmin_kcal
    - vmax_kcal

Notes:
- ESP values are reported in kcal/mol
- Parallelized using multiprocessing
"""
import numpy as np
import json
import multiprocessing as mp

# Constants
VDW_RADII = {'H':1.20,'C':1.70,'N':1.55,'O':1.52,'F':1.47,'S':1.80,'Cl':1.75} # van der Waals radii (Angstrom)
BOHR = 0.529177
AU_TO_KCAL = 627.509 #atomic unit to kcal
# ESP extrema calculation
def compute_vmin_vmax(mol):
    coords  = np.array(mol['coords'])
    charges = np.array(mol['chelpg_charges'])
    radii   = np.array([VDW_RADII.get(s, 1.70) for s in mol['species']])
    
    # Generate points on VdW surface (Golden Section Spiral)
    n_pts = 100
    surface_pts = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    
    for i in range(len(mol['species'])):
        center, r = coords[i], radii[i]
        for k in range(n_pts):
            y = 1.0 - (k / (n_pts - 1)) * 2.0
            radius_at_y = np.sqrt(max(1.0 - y*y, 0.0))
            theta = phi * k
            pt = center + r * np.array([np.cos(theta)*radius_at_y, y, np.sin(theta)*radius_at_y])
            
            # Check if point is buried
            dists = np.linalg.norm(coords - pt, axis=1)
            if np.all(dists >= radii * 0.99):
                surface_pts.append(pt)

    if not surface_pts: return None

    # Calculate ESP at surface
    
    surface_bohr = np.array(surface_pts) / BOHR
    coords_bohr  = coords / BOHR
    diff = surface_bohr[:, np.newaxis, :] - coords_bohr[np.newaxis, :, :]
    dists = np.maximum(np.linalg.norm(diff, axis=2), 1e-6)
    esp = (charges[np.newaxis, :] / dists).sum(axis=1)
    # Return extrema
    return {
        'mol_id': mol['mol_id'],
        'vmin_kcal': float(esp.min() * AU_TO_KCAL),
        'vmax_kcal': float(esp.max() * AU_TO_KCAL)
    }
#main execution 
def main():
    # Load compact dataset
    with open("DrugESP_149k_compact.json") as f:
        dataset = json.load(f)
    # Parallel ESP calculations
    with mp.Pool(processes=mp.cpu_count()) as pool:
        extrema = list(pool.map(compute_vmin_vmax, dataset))

    # Merge back into dataset
    lookup = {r['mol_id']: r for r in extrema if r}
    for mol in dataset:
        res = lookup.get(mol['mol_id'])
        if res:
            mol.update({'vmin_kcal': res['vmin_kcal'], 'vmax_kcal': res['vmax_kcal']})
    # Save updated dataset
    with open("DrugESP_149k_compact.json", 'w', separators=(',', ':')) as f:
        json.dump(dataset, f)
    print("Vmin/Vmax updated in compact JSON.")

if __name__ == "__main__":
    main()