# @file test.py
# @copyright Copyright © 2026 Isaías Rodríguez (isurwars@gmail.com)
# @par License
# SPDX-License-Identifier: AGPL-3.0-only

import torch
import torch.nn as nn
import numpy as np
import ase.db
import matplotlib.pyplot as plt
import glob
import os
import re
from orb_models.forcefield.pretrained import orb_v3_conservative_omol, orb_v3_direct_omol

# ==========================================
# --- EVALUATION CONFIGURATION TOGGLES ---
# ==========================================
CHECKPOINT_PATH = "ckpts_electronic/" # Update to target file or folder

EVALUATE_ALL = True  # Set to True to evaluate all checkpoints in the directory of CHECKPOINT_PATH
SAVE_PLOTS = True
EXPORT_CSV = False

latent_dim = 256 
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- Initializing Evaluation ({device}) ---")

# 1. Initialize Base Model (Always required for GNN Feature Extractor structure mapping)
model, atoms_adapter = orb_v3_direct_omol(train=False, device=device)

# 2. Get the list of checkpoints to evaluate
if EVALUATE_ALL:
    if os.path.isdir(CHECKPOINT_PATH):
        ckpt_dir = CHECKPOINT_PATH
    else:
        ckpt_dir = os.path.dirname(CHECKPOINT_PATH) or "."
    
    # Sort files naturally by epoch number
    checkpoint_paths = sorted(
        glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch*.ckpt")),
        key=lambda x: int(re.search(r'epoch(\d+)', x).group(1)) if re.search(r'epoch(\d+)', x) else 0
    )
    if not checkpoint_paths:
        checkpoint_paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
else:
    checkpoint_paths = [CHECKPOINT_PATH]

if not checkpoint_paths:
    raise ValueError(f"No checkpoint file(s) found matching: {CHECKPOINT_PATH}")

print(f"Found {len(checkpoint_paths)} checkpoint(s) to evaluate.")

# 3. Preprocess and Cache Database Frames (Speeds up evaluation by 10-20x)
db = ase.db.connect('cellulose_finetuning.db')
print(f"Preprocessing and caching {len(db)} database frames...")

cached_frames = []
for row in db.select():
    test_atoms = row.toatoms()
    
    # Run the CPU-based adapter step to build graphs once
    single_graph = atoms_adapter.from_ase_atoms(test_atoms)
    
    # Cache the ground truth properties and structure metadata
    gt = {
        "energy": row.energy if hasattr(row, "energy") else None,
        "forces": row.forces if hasattr(row, "forces") else None,
        "eigenvalues": row.data.get("eigenvalues") if "eigenvalues" in row.data else None,
        "weights": row.data.get("weights") if "weights" in row.data else None,
        "cell": test_atoms.get_cell().array
    }
    cached_frames.append((single_graph, gt))

print("Preprocessing complete!")

# 4. Evaluation Loop
def get_rmse(true, pred):
    return np.sqrt(np.mean((np.array(true) - np.array(pred))**2))

summary_results = []
# Initialize eigenvalue and weight heads (defined once outside the loop)
eigenvalue_head = nn.Sequential(
    nn.LayerNorm(latent_dim),
    nn.Linear(latent_dim, 1024),
    nn.SiLU(),
    nn.LayerNorm(1024),
    nn.Linear(1024, 1024),
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

for ckpt_idx, ckpt_path in enumerate(checkpoint_paths):
    print(f"\n[{ckpt_idx + 1}/{len(checkpoint_paths)}] Evaluating: {ckpt_path}")
    
    # Load parameters from checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    try:
        eigenvalue_head.load_state_dict(checkpoint["eigenvalue_head_state"])
        current_eigenvalue_head = eigenvalue_head
        eigenvalue_head.eval()
    except RuntimeError:
        print("  [Warning] Detected older checkpoint architecture. Temporarily falling back to old eigenvalue head structure.")
        old_eigenvalue_head = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 250)
        ).to(device)
        old_eigenvalue_head.load_state_dict(checkpoint["eigenvalue_head_state"])
        current_eigenvalue_head = old_eigenvalue_head
        current_eigenvalue_head.eval()
        
    weight_head.load_state_dict(checkpoint["weight_head_state"])
    weight_head.eval()
    
    results = {
        "forces_true": [], "forces_pred": [],
        "eigs_true": [], "eigs_pred": [],
        "weights_true": [], "weights_pred": []
    }
    
    # Run predictions on cached frames
    is_conservative = model.__class__.__name__ == "ConservativeForcefieldRegressor"
    for single_graph, gt in cached_frames:
        inputs = atoms_adapter.batch([single_graph]).to(device)
        
        inputs.system_features = {
            "total_charge": torch.tensor([0.0], dtype=torch.float32, device=device),
            "spin_multiplicity": torch.tensor([1.0], dtype=torch.float32, device=device),
            "cell": torch.tensor(gt["cell"], dtype=torch.float32, device=device).unsqueeze(0)
        }
        
        # 1. Physics Evaluation (Forces)
        if is_conservative:
            with torch.set_grad_enabled(True):
                inputs.positions.requires_grad_(True)
                base_out = model(inputs)
                pred_forces = base_out["grad_forces"]
        else:
            with torch.no_grad():
                base_out = model(inputs)
                pred_forces = base_out["forces"]
            
        results["forces_true"].append(gt["forces"])
        results["forces_pred"].append(pred_forces.detach().cpu().numpy())
            
        # 2. Electronic Structure Evaluation (Eigenvalues & PDOS Weights)
        with torch.no_grad():
            gnn_out = model.model(inputs)
            node_feats = gnn_out["node_features"]
            graph_feats = node_feats.mean(dim=0, keepdim=True)
            
            pred_eigs = current_eigenvalue_head(graph_feats).cpu().numpy().flatten()
            pred_weights = weight_head(node_feats).cpu().numpy().flatten()
            
            results["eigs_true"].append(gt["eigenvalues"])
            results["eigs_pred"].append(pred_eigs)
            results["weights_true"].append(np.array(gt["weights"]).flatten())
            results["weights_pred"].append(pred_weights)
            
    # Calculate Metrics and Export Plots
    ckpt_dir = os.path.dirname(ckpt_path) or "."
    epoch_match = re.search(r'epoch(\d+)', ckpt_path)
    suffix = f"_epoch{epoch_match.group(1)}" if epoch_match else f"_{os.path.basename(ckpt_path)}"
    
    # Physics calculations (Energy calculation is disabled)
    
    f_true = np.concatenate(results["forces_true"]).flatten()
    f_pred = np.concatenate(results["forces_pred"]).flatten()
    forces_rmse = get_rmse(f_true, f_pred)
    
    # Electronic structure calculations
    eig_true = np.array(results["eigs_true"]).flatten()
    eig_pred = np.array(results["eigs_pred"]).flatten()
    eigs_rmse = get_rmse(eig_true, eig_pred)
    
    w_true = np.concatenate(results["weights_true"])
    w_pred = np.concatenate(results["weights_pred"])
    weights_rmse = get_rmse(w_true, w_pred)
    
    print(f"  Eigenvalues RMSE: {eigs_rmse:.4f} eV")
    print(f"  Weights RMSE:     {weights_rmse:.4f}")
    print(f"  Forces RMSE:      {forces_rmse:.4f} eV/Å")
    
    summary_results.append({
        "checkpoint": os.path.basename(ckpt_path),
        "eigs_rmse": eigs_rmse,
        "weights_rmse": weights_rmse,
        "forces_rmse": forces_rmse
    })
    
    if SAVE_PLOTS:
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Eigenvalues Parity Plot
        ax[0].scatter(eig_true, eig_pred, alpha=0.1, s=0.5)
        ax[0].plot([eig_true.min(), eig_true.max()], [eig_true.min(), eig_true.max()], 'r--')
        ax[0].set_title(f"Eigenvalues (RMSE: {eigs_rmse:.3f} eV)")
        ax[0].set_xlabel("DFT Eigenvalues (eV)")
        ax[0].set_ylabel("ML Predicted (eV)")
        
        # 2. PDOS Weights Parity Plot
        ax[1].scatter(w_true, w_pred, alpha=0.1, s=0.5)
        ax[1].plot([w_true.min(), w_true.max()], [w_true.min(), w_true.max()], 'r--')
        ax[1].set_title(f"PDOS Weights (RMSE: {weights_rmse:.3f})")
        ax[1].set_xlabel("DFT PDOS Weights")
        ax[1].set_ylabel("ML Predicted")
        
        # 3. Forces Parity Plot
        ax[2].scatter(f_true, f_pred, alpha=0.3, s=1)
        ax[2].plot([f_true.min(), f_true.max()], [f_true.min(), f_true.max()], 'r--')
        ax[2].set_title(f"Forces (RMSE: {forces_rmse:.3f} eV/Å)")
        ax[2].set_xlabel("DFT Forces (eV/Å)")
        ax[2].set_ylabel("ML Predicted (eV/Å)")
        
        plt.tight_layout()
        plot_path = os.path.join(ckpt_dir, f"cellulose_{suffix}.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"  Saved combined plot: {plot_path}")
        
    if EXPORT_CSV:
        np.savetxt(os.path.join(ckpt_dir, f"cellulose_eigenvalues{suffix}.csv"), np.column_stack((eig_true, eig_pred)), delimiter=",", header="Eigenvalues_True_eV,Eigenvalues_Pred_eV", comments="")
        np.savetxt(os.path.join(ckpt_dir, f"cellulose_weights{suffix}.csv"), np.column_stack((w_true, w_pred)), delimiter=",", header="Weights_True,Weights_Pred", comments="")
        np.savetxt(os.path.join(ckpt_dir, f"cellulose_forces{suffix}.csv"), np.column_stack((f_true, f_pred)), delimiter=",", header="Forces_True_eV_A,Forces_Pred_eV_A", comments="")

# 5. Print Final Summary Table
print("\n" + "=" * 76)
print("                            EVALUATION SUMMARY TABLE")
print("=" * 76)
print(f"{'Checkpoint Filename':<30} | {'Eigs RMSE':<11} | {'Weights RMSE':<12} | {'Forces RMSE':<11}")
print("-" * 76)
for res in summary_results:
    print(f"{res['checkpoint']:<30} | {res['eigs_rmse']:<11.4f} | {res['weights_rmse']:<12.4f} | {res['forces_rmse']:<11.4f}")
print("=" * 76)