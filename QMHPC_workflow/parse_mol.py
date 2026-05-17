"""
ORCA Output Parser

Author: Taradutt Pattnaik
Date: 2026

Description:
Parser used in mega_job.sh to extract out computed properties
from ORCA DFT output file

Extracted Properties:
- Total electronic energy
- Dipole moment
- CHELPG atomic charges
- Mulliken atomic charges
- Quadrupole tensor
- Polarizability tensor
- HOMO/LUMO energies
- HOMO-LUMO gap
- Runtime statistics

Usage:
python parse_mol.py <mol_id> <out_file> <results_json>

Arguments:
1. mol_id       : Molecule identifier
2. out_file     : ORCA output file
3. results_json : Output JSONL file

Notes:
- Results are appended in JSONL format
- Partial parsing failures are tracked using parse_errors
- Non-converged calculations are marked as failed
"""
import sys
import re
import json
import os

def parse_orca_output(content):
    lines   = content.split('\n')
    result  = {}
    errors  = []

    if 'ORCA TERMINATED NORMALLY' not in content:
        return {'status': 'failed', 'errors': ['not_converged']}

    
    # Energy
    m = re.search(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)', content)
    if m:
        result['energy_hartree'] = float(m.group(1))
    else:
        errors.append('energy_missing')

    # Dipole 
    m = re.search(
        r'Total Dipole Moment\s*:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
        content)
    
    if m:
        result['dipole_vector'] = [float(m.group(1)),
                                   float(m.group(2)),
                                   float(m.group(3))]
        m2 = re.search(r'Magnitude \(Debye\)\s*:\s*([-\d.]+)', content)
        if m2:
            result['dipole_debye'] = float(m2.group(1))
    else:
        errors.append('dipole_missing')

    # CHELPG charges
    chelpg   = []
    in_block = False
    for line in lines:
        if 'CHELPG Charges' in line:
            in_block = True
            continue
        if in_block:
            m = re.match(r'\s*\d+\s+\w+\s*:\s*([-\d.]+)', line)
            if m:
                chelpg.append(float(m.group(1)))
            if 'Total charge' in line:
                in_block = False
                break
    if chelpg:
        result['chelpg_charges'] = chelpg
        result['chelpg_sum']     = round(sum(chelpg), 6)
        result['n_atoms']        = len(chelpg)
    else:
        errors.append('chelpg_missing')

    # Mulliken charges
    
    mulliken = []
    in_block = False
    for line in lines:
        if 'MULLIKEN ATOMIC CHARGES' in line:
            in_block = True
            continue
        if in_block:
            m = re.match(r'\s*\d+\s+\w+\s*:\s*([-\d.]+)', line)
            if m:
                mulliken.append(float(m.group(1)))
            if 'Sum of atomic charges' in line:
                in_block = False
                break
    if mulliken:
        result['mulliken_charges'] = mulliken
    else:
        errors.append('mulliken_missing')

    # Quadrupole tensor
    
    m = re.search(
        r'TOT\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
        r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+\(a\.u\.\)',
        content)
    if m:
        result['quadrupole_tensor'] = {
            'xx': float(m.group(1)), 'yy': float(m.group(2)),
            'zz': float(m.group(3)), 'xy': float(m.group(4)),
            'xz': float(m.group(5)), 'yz': float(m.group(6))
        }
    else:
        errors.append('quadrupole_missing')

    # Polarizability
    
    m_iso = re.search(r'Isotropic polarizability\s*:\s*([-\d.]+)', content)
    if m_iso:
        result['polarizability_iso'] = float(m_iso.group(1))
        m_t = re.search(
            r'The raw cartesian tensor \(atomic units\):\s*\n'
            r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n'
            r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n'
            r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
            content)
        
        if m_t:
            result['polarizability_tensor'] = {
                'xx': float(m_t.group(1)),
                'xy': float(m_t.group(2)),
                'xz': float(m_t.group(3)),
                'yy': float(m_t.group(5)),
                'yz': float(m_t.group(6)),
                'zz': float(m_t.group(9))
            
            }
    else:
        errors.append('polarizability_missing')

    # HOMO LUMO
    homo, lumo  = None, None
    in_orb      = False
    for line in lines:
        if 'ORBITAL ENERGIES' in line:
            in_orb = True
            continue
        if in_orb:
            m = re.match(r'\s*\d+\s+([\d.]+)\s+([-\d.]+)', line)
            if m:
                occ = float(m.group(1))
                eng = float(m.group(2))
                if occ > 0.5:
                    homo = eng
                elif occ < 0.5 and homo is not None and lumo is None:
                    lumo = eng
            if lumo is not None:
                break
    if homo and lumo:
        result['homo_ev'] = round(homo * 27.2114, 4)#hartree to ev
        result['lumo_ev'] = round(lumo * 27.2114, 4)#hartree to ev
        result['gap_ev']  = round((lumo - homo) * 27.2114, 4)#hartree to ev
    else:
        errors.append('homo_lumo_missing')

    # Timings/Run time stats
    timings = {}
    m = re.search(
        r'TOTAL RUN TIME: \d+ days \d+ hours (\d+) minutes (\d+) seconds',
        content)
    if m:
        timings['total_sec'] = int(m.group(1)) * 60 + int(m.group(2))
    m = re.search(r'SCF iterations\s+\.\.\.\s+([\d.]+) sec', content)
    if m: timings['scf_sec'] = float(m.group(1))
    m = re.search(r'SCF Response\s+\.\.\.\s+([\d.]+) sec', content)
    if m: timings['scf_response_sec'] = float(m.group(1))
    result['timings']      = timings
    result['status']       = 'converged'
    result['parse_errors'] = errors
    result['parse_ok']     = len(errors) == 0

    return result

#main execution ..used by mega_job.sh
if __name__ == '__main__':
    # Usage: python parse_mol.py <mol_id> <out_file> <results_json>
    mol_id      = int(sys.argv[1])
    out_file    = sys.argv[2]
    results_json = sys.argv[3]

    with open(out_file) as f:
        content = f.read()

    parsed           = parse_orca_output(content)
    parsed['mol_id'] = mol_id

    # Append to worker results JSON (one entry per line = JSONL format)
    with open(results_json, 'a') as f:
        f.write(json.dumps(parsed) + '\n')
