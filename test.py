# @file test.py
# @copyright Copyright © 2026 Isaías Rodríguez (isurwars@gmail.com)
# @par License
# SPDX-License-Identifier: AGPL-3.0-only

import torch
import torch.nn as nn
import numpy as np
import ase.db
import matplotlib.pyplot as plt
from orb_models.forcefield.pretrained import orb_v3_conservative_omol

# ==========================================
# --- EVALUATION CONFIGURATION TOGGLES ---
# ==========================================
TEST_MODE = "electronic"  # Set to "physics" or "electronic"
CHECKPOINT_PATH = "ckpts_electronic/checkpoint_epoch100.ckpt" # Update to your target checkpoint

SAVE_PLOTS = True
EXPORT_CSV = False

latent_dim = 256 
device = "cuda"

print(f"--- Initializing Evaluation in {TEST_MODE.upper()} Mode ---")

# 1. Load Base Model (Always required for the GNN Feature Extractor)
model, atoms_adapter = orb_v3_conservative_omol(train=False, device=device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# 2. Conditionally Load Electronic Heads
if TEST_MODE == "electronic":
    eigenvalue_head = nn.Sequential(
        nn.Linear(latent_dim, 1024),
        nn.SiLU(),
        nn.Linear(1024, 250)
    ).to(device)

    weight_head = nn.Sequential(
        nn.Linear(latent_dim, 1024),
        nn.SiLU(),
        nn.Linear(1024, 1024),
        nn.SiLU(),
        nn.Linear(1024, 250),
        nn.Softplus()
    ).to(device)
    
    eigenvalue_head.load_state_dict(checkpoint["eigenvalue_head_state"])
    weight_head.load_state_dict(checkpoint["weight_head_state"])
    eigenvalue_head.eval()
    weight_head.eval()

# 3. Data Containers
results = {
    "energy_true": [], "energy_pred": [],
    "forces_true": [], "forces_pred": [],
    "eigs_true": [], "eigs_pred": [],
    "weights_true": [], "weights_pred": []
}

# 4. Evaluation Loop
db = ase.db.connect('cellulose_finetuning.db')
print(f"Evaluating {len(db)} frames...")

for row in db.select():
    test_atoms = row.toatoms()
    
    single_graph = atoms_adapter.from_ase_atoms(test_atoms)
    inputs = atoms_adapter.batch([single_graph]).to(device)
    inputs.positions.requires_grad_(True)
    
    inputs.system_features = {
        "total_charge": torch.tensor([0.0], dtype=torch.float32, device=device),
        "spin_multiplicity": torch.tensor([1.0], dtype=torch.float32, device=device),
        "cell": torch.tensor(test_atoms.get_cell().array, dtype=torch.float32, device=device).unsqueeze(0)
    }

    # ---------------------------------------------------------
    # CONDITIONAL INFERENCE
    # ---------------------------------------------------------
    if TEST_MODE == "physics":
        # PHYSICS: We MUST leave autograd enabled so the conservative model 
        # can calculate forces as the derivative of energy (-dE/dR)
        base_out = model(inputs)
        
        results["energy_true"].append(row.energy)
        results["energy_pred"].append(base_out["energy"].detach().cpu().numpy().item())
        results["forces_true"].append(row.forces)
        results["forces_pred"].append(base_out["grad_forces"].detach().cpu().numpy())
            
    elif TEST_MODE == "electronic":
        # ELECTRONIC: No calculus needed. Disable gradients for maximum speed and memory.
        with torch.no_grad():
            gnn_out = model.model(inputs)
            node_feats = gnn_out["node_features"]
            graph_feats = node_feats.mean(dim=0, keepdim=True)
            
            pred_eigs = eigenvalue_head(graph_feats).cpu().numpy().flatten()
            pred_weights = weight_head(node_feats).cpu().numpy().flatten()
            
            true_weights_flat = np.array(row.data["weights"]).flatten()
            
            results["eigs_true"].append(row.data["eigenvalues"])
            results["eigs_pred"].append(pred_eigs)
            results["weights_true"].append(true_weights_flat)
            results["weights_pred"].append(pred_weights)

# 5. Metrics & Exports
def get_rmse(true, pred):
    return np.sqrt(np.mean((np.array(true) - np.array(pred))**2))

print(f"\n--- Final Metrics ({TEST_MODE.upper()}) ---")

if TEST_MODE == "physics":
    # Calculate Physics Metrics
    energy_true = np.array(results["energy_true"])
    energy_pred = np.array(results["energy_pred"])
    baseline_offset = np.mean(energy_true - energy_pred)
    aligned_energy_pred = energy_pred + baseline_offset
    
    energy_rmse = get_rmse(energy_true, aligned_energy_pred)
    f_true = np.concatenate(results["forces_true"]).flatten()
    f_pred = np.concatenate(results["forces_pred"]).flatten()
    forces_rmse = get_rmse(f_true, f_pred)
    
    print(f"Energy RMSE: {energy_rmse:.4f} eV")
    print(f"Forces RMSE: {forces_rmse:.4f} eV/Å")
    print(f"Baseline Offset: {baseline_offset:.4f} eV")

    # Plot Physics
    if SAVE_PLOTS:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(energy_true, aligned_energy_pred, alpha=0.5)
        ax[0].plot([energy_true.min(), energy_true.max()], [energy_true.min(), energy_true.max()], 'r--')
        ax[0].set_title(f"Energy (RMSE: {energy_rmse:.3f} eV)")
        ax[0].set_xlabel("DFT Energy (eV)")
        ax[0].set_ylabel("ML Predicted (Aligned) (eV)")

        ax[1].scatter(f_true, f_pred, alpha=0.3, s=1)
        ax[1].plot([f_true.min(), f_true.max()], [f_true.min(), f_true.max()], 'r--')
        ax[1].set_title(f"Forces (RMSE: {forces_rmse:.3f} eV/Å)")
        ax[1].set_xlabel("DFT Forces (eV/Å)")
        ax[1].set_ylabel("ML Predicted (eV/Å)")
        
        plt.tight_layout()
        plt.savefig("cellulose_physics_parity.png")
        print("Saved cellulose_physics_parity.png")

    # Export Physics CSVs
    if EXPORT_CSV:
        np.savetxt("cellulose_energy.csv", np.column_stack((energy_true, aligned_energy_pred)), delimiter=",", header="Energy_True_eV,Energy_Pred_eV", comments="")
        np.savetxt("cellulose_forces.csv", np.column_stack((f_true, f_pred)), delimiter=",", header="Forces_True_eV_A,Forces_Pred_eV_A", comments="")
        with open("cellulose_physics_metrics.txt", "w") as f:
            f.write(f"Energy RMSE: {energy_rmse:.4f} eV\nForces RMSE: {forces_rmse:.4f} eV/A\nOffset: {baseline_offset:.4f} eV\n")

elif TEST_MODE == "electronic":
    # Calculate Electronic Metrics
    eig_true = np.array(results["eigs_true"]).flatten()
    eig_pred = np.array(results["eigs_pred"]).flatten()
    eigs_rmse = get_rmse(eig_true, eig_pred)
    
    # Weights are already flattened during the inference loop
    w_true = np.concatenate(results["weights_true"])
    w_pred = np.concatenate(results["weights_pred"])
    weights_rmse = get_rmse(w_true, w_pred)
    
    print(f"Eigenvalues RMSE: {eigs_rmse:.4f} eV")
    print(f"Weights RMSE: {weights_rmse:.4f} eV")

    # Plot Electronic
    if SAVE_PLOTS:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(eig_true, eig_pred, alpha=0.1, s=0.5)
        ax[0].plot([eig_true.min(), eig_true.max()], [eig_true.min(), eig_true.max()], 'r--')
        ax[0].set_title(f"Eigenvalues (RMSE: {eigs_rmse:.3f} eV)")
        ax[0].set_xlabel("DFT Eigenvalues (eV)")
        ax[0].set_ylabel("ML Predicted (eV)")

        ax[1].scatter(w_true, w_pred, alpha=0.1, s=0.5)
        ax[1].plot([w_true.min(), w_true.max()], [w_true.min(), w_true.max()], 'r--')
        ax[1].set_title(f"Weights (RMSE: {weights_rmse:.3f} eV)")
        ax[1].set_xlabel("DFT PDOS Weights")
        ax[1].set_ylabel("ML Predicted")
        
        plt.tight_layout()
        plt.savefig("cellulose_electronic_parity.png")
        print("Saved cellulose_electronic_parity.png")

    # Export Electronic CSVs
    if EXPORT_CSV:
        np.savetxt("cellulose_eigenvalues.csv", np.column_stack((eig_true, eig_pred)), delimiter=",", header="Eigenvalues_True_eV,Eigenvalues_Pred_eV", comments="")
        np.savetxt("cellulose_weights.csv", np.column_stack((w_true, w_pred)), delimiter=",", header="Weights_True,Weights_Pred", comments="")
        with open("cellulose_electronic_metrics.txt", "w") as f:
            f.write(f"Eigenvalues RMSE: {eigs_rmse:.4f} eV\nWeights RMSE: {weights_rmse:.4f} eV\n")