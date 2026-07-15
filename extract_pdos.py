# @file extract_pdos.py
# @copyright Copyright © 2026 Isaías Rodríguez (isurwars@gmail.com)
# @par License
# SPDX-License-Identifier: AGPL-3.0-only

"""
extract_pdos.py — Extract CASTEP, bands, and pdos bin files to create cellulose.db and cellulose.xyz with logit-transformed pDoS weights.
"""

import argparse
import glob
import os
import re
import numpy as np
import ase.db
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import castepxbin

def parse_manual_castep(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    lattice = []
    species = []
    coords_frac = []
    forces = []
    energy = None

    reading_lattice = False
    reading_positions = False
    reading_forces = False

    for line in lines:
        clean = re.sub(r"\\", "", line).strip()

        if "Final energy, E" in clean:
            energy_match = re.search(r"=\s+(-?\d+\.\d+)", clean)
            if energy_match:
                energy = float(energy_match.group(1))

        if "Real Lattice(A)" in clean:
            reading_lattice = True
            continue
        if reading_lattice:
            parts = clean.split()
            if len(parts) >= 3 and len(lattice) < 3:
                lattice.append([float(x) for x in parts[:3]])
            elif len(lattice) == 3:
                reading_lattice = False

        if "Fractional coordinates of atoms" in clean:
            reading_positions = True
            continue
        if reading_positions:
            if "x---" in clean or "Element" in clean:
                continue
            if "xxxxxxxxxx" in clean or "No user defined" in clean:
                reading_positions = False
                continue

            parts = clean.replace('x', '').split()
            if len(parts) >= 5:
                species.append(parts[0])
                coords_frac.append([float(parts[2]), float(parts[3]), float(parts[4])])

        if "Cartesian components (eV/A)" in clean:
            reading_forces = True
            continue
        if reading_forces:
            if "---" in clean or "x  y  z" in clean:
                continue
            if "*" not in clean:
                reading_forces = False
                continue

            parts = clean.replace('*', '').split()
            if len(parts) >= 4:
                forces.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])

    lattice_np = np.array(lattice)
    coords_np = np.array(coords_frac)
    positions_abs = coords_np @ lattice_np
    forces_np = np.array(forces)

    return {
        "energy": energy,
        "lattice": lattice_np,
        "species": species,
        "positions": positions_abs,
        "forces": forces_np
    }

def parse_manual_bands(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    eigenvalues = []
    reading_eigenvalues = False

    for line in lines:
        clean = re.sub(r"\\", "", line).strip()

        if "Spin component 1" in clean:
            reading_eigenvalues = True
            continue

        if reading_eigenvalues:
            if "K-point" in clean or not clean:
                if len(eigenvalues) >= 250:
                    reading_eigenvalues = False
                continue

            try:
                val = float(clean)
                eigenvalues.append(val)
            except ValueError:
                continue

    return np.array(eigenvalues)

def transform_target(y):
    y = np.clip(y, 1e-4, 1 - 1e-4)
    return np.log(y / (1 - y))

def write_mace_dataset(filename, dataset):
    """Writes a multi-frame dataset to Extended XYZ format."""
    print(f"Writing Extended XYZ file to '{filename}'...")
    with open(filename, 'w') as f:
        for frame_data, eigen_map, eigen_energies in dataset:
            num_atoms = len(frame_data['species'])
            lattice_str = " ".join(map(str, frame_data['lattice'].flatten()))
            properties = "species:S:1:pos:R:3:forces:R:3:eigen_weights:R:250"
            energies_str = " ".join([f"{e:.6f}" for e in eigen_energies])

            f.write(f"{num_atoms}\n")
            comment = (
                f'Lattice="{lattice_str}" '
                f'Properties={properties} '
                f'energy={frame_data["energy"]:.8f} '
                f'eigen_energies="{energies_str}" '
                f'pbc="T T T"'
            )
            f.write(f"{comment}\n")

            for i in range(num_atoms):
                s = frame_data['species'][i]
                px, py, pz = frame_data['positions'][i]
                fx, fy, fz = frame_data['forces'][i]
                w_str = " ".join([f"{w:.6f}" for w in eigen_map[i, :]])

                f.write(f"{s} {px:12.8f} {py:12.8f} {pz:12.8f} "
                        f"{fx:12.8f} {fy:12.8f} {fz:12.8f} "
                        f"{w_str}\n")

def create_orb_database(db_filename, dataset):
    """Creates the ASE SQLite database."""
    if os.path.exists(db_filename):
        print(f"Removing existing database file: {db_filename}")
        os.remove(db_filename)
        
    print(f"Writing ASE Database file to '{db_filename}'...")
    db = ase.db.connect(db_filename)

    for frame_data, eigen_map, eigen_energies in dataset:
        atoms = Atoms(
            symbols=frame_data['species'],
            positions=frame_data['positions'],
            cell=frame_data['lattice'],
            pbc=True
        )

        calc = SinglePointCalculator(
            atoms=atoms,
            energy=frame_data['energy'],
            forces=frame_data['forces']
        )
        atoms.calc = calc

        data_payload = {
            'eigenvalues': eigen_energies,
            'weights': eigen_map
        }

        db.write(atoms, data=data_payload)

def main():
    parser = argparse.ArgumentParser(description="Extract CASTEP, bands, and pdos bin data into transformed database and xyz format")
    parser.add_argument("--pdos_dir", default="pdos_results", help="Directory containing pdos results")
    parser.add_argument("--out_db", default="cellulose.db", help="Path to output ASE SQLite database")
    parser.add_argument("--out_xyz", default="cellulose.xyz", help="Path to output Extended XYZ file")
    parser.add_argument("--no-logit", action="store_true", help="Extract raw weights without logit transformation")
    args = parser.parse_args()

    frame_files = sorted(glob.glob(os.path.join(args.pdos_dir, "frame_*.castep")))
    print(f"Found {len(frame_files)} CASTEP files in '{args.pdos_dir}'. Starting batch processing...")

    all_frames = []

    for castep_path in frame_files:
        base_name = os.path.splitext(os.path.basename(castep_path))[0]
        folder = os.path.dirname(castep_path)

        bands_path = os.path.join(folder, f"{base_name}.bands")
        pdos_bin_path = os.path.join(folder, f"{base_name}.pdos_bin")

        if not os.path.exists(bands_path) or not os.path.exists(pdos_bin_path):
            print(f"Skipping {base_name}: Missing .bands or .pdos_bin")
            continue

        try:
            # 1. Parse structural data
            frame_data = parse_manual_castep(castep_path)

            # 2. Parse eigenvalues
            frame_eigenvalues = parse_manual_bands(bands_path)

            # 3. Parse and map PDOS weights
            pdos_data = castepxbin.read_pdos_bin(pdos_bin_path)
            weights = pdos_data['pdos_weights']
            ions = pdos_data['ion']
            species_indices_pdos = pdos_data['species']

            # Reconstruct global atom index by mapping local (species, ion) to global index
            atoms_species = frame_data['species']
            unique_species = []
            for s in atoms_species:
                if s not in unique_species:
                    unique_species.append(s)
            species_indices = {s: [idx for idx, sym in enumerate(atoms_species) if sym == s] for s in unique_species}

            # Create the (72, 250) map for this specific frame
            current_map = np.zeros((72, 250))
            for orbital_idx in range(204):
                sp_idx = species_indices_pdos[orbital_idx] - 1
                sym = unique_species[sp_idx]
                ion_idx = ions[orbital_idx] - 1
                atom_idx = species_indices[sym][ion_idx]
                current_map[atom_idx, :] += weights[orbital_idx, :, 0, 0]

            if args.no_logit:
                transformed_map = current_map
            else:
                # Transform weights to logit space
                transformed_map = transform_target(current_map)

            # Convert eigenvalues from Hartree to eV for the energy axis
            eigen_energies_ev = frame_eigenvalues * 27.2114

            all_frames.append((frame_data, transformed_map, eigen_energies_ev))

        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    print(f"Successfully processed {len(all_frames)} frames.")

    # Write outputs
    write_mace_dataset(args.out_xyz, all_frames)
    create_orb_database(args.out_db, all_frames)
    print("Done!")

if __name__ == "__main__":
    main()
