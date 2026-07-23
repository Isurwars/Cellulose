# @file explore_pdos.py
# @copyright Copyright © 2026 Isaías Rodríguez (isurwars@gmail.com)

"""
explore_pdos.py — Reads CASTEP pdos_bin and outputs a human-readable text report and CSV mapping.
"""

import castepxbin
import numpy as np
import os
import csv

from extract_pdos import parse_manual_castep

def main():
    pdos_bin_path = "pdos_results/frame_000.pdos_bin"
    castep_path = "pdos_results/frame_000.castep"
    bands_path = "pdos_results/frame_000.bands"
    
    if not os.path.exists(pdos_bin_path) or not os.path.exists(castep_path):
        print("Missing frame_000 files in pdos_results directory.")
        return
        
    print("Loading data...")
    pdos_data = castepxbin.read_pdos_bin(pdos_bin_path)
    weights = pdos_data['pdos_weights'][:, :, 0, 0] # shape (204, 250)
    ions = pdos_data['ion'] # shape (204,)
    am_channel = pdos_data['am_channel'] # shape (204,)
    
    atoms_species = parse_manual_castep(castep_path)['species']
    species_indices_pdos = pdos_data['species']
    
    unique_species = []
    for s in atoms_species:
        if s not in unique_species:
            unique_species.append(s)
    species_indices = {s: [idx for idx, sym in enumerate(atoms_species) if sym == s] for s in unique_species}
    
    orbital_names = []
    current_atom_channel_count = {}
    
    for i in range(len(ions)):
        sp_idx = species_indices_pdos[i] - 1
        sym = unique_species[sp_idx]
        ion_idx = ions[i] - 1
        atom_idx = species_indices[sym][ion_idx]
        ch = am_channel[i]
        
        ch_name = "s" if ch == 0 else "p"
        if ch_name == "p":
            key = (atom_idx, ch)
            cnt = current_atom_channel_count.get(key, 0)
            p_sublabels = ["px", "py", "pz"]
            ch_name = p_sublabels[cnt % 3]
            current_atom_channel_count[key] = cnt + 1
            
        orbital_names.append((atom_idx + 1, sym, ch_name))
        
    report_path = "pdos_readable_summary.txt"
    print(f"Writing readable summary to {report_path}...")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CASTEP PDOS BINARY EXPLORATION AND ORBITAL PROJECTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Source PDOS Bin: {pdos_bin_path}\n")
        f.write(f"Source CASTEP:   {castep_path}\n")
        f.write(f"Source Bands:    {bands_path}\n\n")
        
        f.write(f"Number of Orbitals: {len(ions)}\n")
        f.write(f"Number of Atoms:    {len(atoms_species)}\n")
        f.write(f"Number of Bands:    {weights.shape[1]}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("1. ORBITAL-BY-ORBITAL DETAIL (First 50 orbitals)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'OrbitalIdx':<12}{'AtomIdx':<10}{'Element':<10}{'Orbital':<10}{'SummedWeight':<15}{'Band0':<12}{'Band100':<12}\n")
        f.write("-" * 80 + "\n")
        for i in range(min(50, len(ions))):
            atom_num, sym, orb = orbital_names[i]
            sum_w = np.sum(weights[i, :])
            w0 = weights[i, 0]
            w100 = weights[i, 100]
            f.write(f"{i:<12}{atom_num:<10}{sym:<10}{orb:<10}{sum_w:<15.4f}{w0:<12.6f}{w100:<12.6f}\n")
        f.write("... (truncated, see CSV output for complete listing) ...\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("2. ATOM-BY-ATOM PROJECTED WEIGHTS SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'AtomIdx':<10}{'Element':<10}{'Total Weight':<15}{'s-channel':<12}{'p-channel':<12}\n")
        f.write("-" * 80 + "\n")
        
        for atom_idx in range(len(atoms_species)):
            sym = atoms_species[atom_idx]
            orb_indices = [i for i, names in enumerate(orbital_names) if names[0] == atom_idx + 1]
            total_w = np.sum(weights[orb_indices, :])
            
            s_indices = [i for i in orb_indices if am_channel[i] == 0]
            p_indices = [i for i in orb_indices if am_channel[i] == 1]
            
            s_w = np.sum(weights[s_indices, :]) if s_indices else 0.0
            p_w = np.sum(weights[p_indices, :]) if p_indices else 0.0
            
            f.write(f"{atom_idx+1:<10}{sym:<10}{total_w:<15.4f}{s_w:<12.4f}{p_w:<12.4f}\n")
            
    csv_path = "pdos_readable_all_orbitals.csv"
    print(f"Writing complete CSV database to {csv_path}...")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["OrbitalIndex", "AtomIndex", "Element", "OrbitalName", "SummedWeight"] + [f"Band_{b}" for b in range(weights.shape[1])]
        writer.writerow(header)
        for i in range(len(ions)):
            atom_num, sym, orb = orbital_names[i]
            sum_w = np.sum(weights[i, :])
            row = [i, atom_num, sym, orb, sum_w] + list(weights[i, :])
            writer.writerow(row)
            
    print("Done writing exploration exports!")

if __name__ == '__main__':
    main()
