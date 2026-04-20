import torch
import torch.nn as nn
import numpy as np
import ase.db
from ase.data import atomic_numbers as SYMBOL_TO_NUM
import matplotlib.pyplot as plt
from orb_models.forcefield.pretrained import orb_v3_conservative_omol

# 1. Architecture Setup (Must match your finetune.py)
latent_dim = 256 
device = "cuda"

eigenvalue_head = nn.Sequential(
    nn.Linear(latent_dim, 1024),
    nn.SiLU(),
    nn.Linear(1024, 250)
).to(device)

weight_head = nn.Sequential(
    nn.Linear(latent_dim, 1024),
    nn.SiLU(),
    nn.Linear(1024, 250)
).to(device)

# 2. Load Model and Checkpoint
model, atoms_adapter = orb_v3_conservative_omol(train=False, device=device)
checkpoint = torch.load("ckpts/checkpoint_epoch100.ckpt", map_location=device)

model.load_state_dict(checkpoint["state_dict"])
eigenvalue_head.load_state_dict(checkpoint["eigenvalue_head_state"])
weight_head.load_state_dict(checkpoint["weight_head_state"])

model.eval()
eigenvalue_head.eval()
weight_head.eval()

# 3. Data Containers
results = {
    "energy_true": [], "energy_pred": [],
    "forces_true": [], "forces_pred": [],
    "eigs_true": [], "eigs_pred": [],
    "weights_true": [], "weights_pred": []
}

# 4. Loop over entire DB
db = ase.db.connect('cellulose_finetuning.db')
print(f"Starting evaluation over {len(db)} frames...")

for row in db.select():
    test_atoms = row.toatoms()
    
    # Prepare Inputs
    single_graph = atoms_adapter.from_ase_atoms(test_atoms)
    inputs = atoms_adapter.batch([single_graph]).to(device)

    inputs.positions.requires_grad_(True)
    
    # MANUALLY INJECT the required electronic and structural properties
    # The 'cell' is critical for force calculations in Orb-v3
    inputs.system_features = {
        "total_charge": torch.tensor([0.0], dtype=torch.float32, device=device),
        "spin_multiplicity": torch.tensor([1.0], dtype=torch.float32, device=device),
        "cell": torch.tensor(test_atoms.get_cell().array, dtype=torch.float32, device=device).unsqueeze(0)
    }

    # Perform Inference
    base_out = model(inputs)

    with torch.no_grad():
        # Get custom head predictions
        gnn_out = model.model(inputs)
        node_feats = gnn_out["node_features"]
        graph_feats = node_feats.mean(dim=0, keepdim=True)
        
        pred_eigs = eigenvalue_head(graph_feats).cpu().numpy().flatten()
        pred_weights = weight_head(node_feats).cpu().numpy().flatten()
        pred_energy = base_out["energy"].cpu().numpy().item()
        pred_forces = base_out["grad_forces"].cpu().numpy()

    # Store results

    # Get the atomic numbers (e.g., [6, 6, 8, 1, ...]) for the current frame
    atomic_numbers = test_atoms.get_atomic_numbers()
    
    # Sum the reference energies for these specific atoms
    # Using .get(z, 0.0) safely defaults to 0 if an element is missing from refs
    #ref_sum = sum(ref_energies.get(z, 0.0) for z in atomic_numbers)
    
    # Shift the true energy to match the ML target
    #true_shifted_energy = row.energy - ref_sum
    
    results["energy_true"].append(row.energy)
    results["energy_pred"].append(pred_energy)
    
    results["forces_true"].append(row.forces)
    results["forces_pred"].append(pred_forces)
    
    # Custom properties stay in row.data
    results["eigs_true"].append(row.data["eigenvalues"])
    results["eigs_pred"].append(pred_eigs)

    results["weights_true"].append(row.data["weights"])
    results["weights_pred"].append(pred_weights)

# 5. Calculate Metrics
energy_true = np.array(results["energy_true"])
energy_pred = np.array(results["energy_pred"])
baseline_offset = np.mean(energy_true - energy_pred)
aligned_energy_pred = energy_pred + baseline_offset

def get_rmse(true, pred):
    return np.sqrt(np.mean((np.array(true) - np.array(pred))**2))

energy_rmse = get_rmse(results["energy_true"], aligned_energy_pred)
forces_rmse = get_rmse(np.concatenate(results["forces_true"]), np.concatenate(results["forces_pred"]))
eigs_rmse = get_rmse(results["eigs_true"], results["eigs_pred"])

# Flatten both arrays completely so they are simple 1D lists of ~18 million numbers
w_true_flat = np.array(results["weights_true"]).flatten()
w_pred_flat = np.array(results["weights_pred"]).flatten()

weights_rmse = get_rmse(w_true_flat, w_pred_flat)

print(f"\n--- Final Metrics ---")
print(f"Energy RMSE: {energy_rmse:.4f} eV")
print(f"Forces RMSE: {forces_rmse:.4f} eV/Å")
print(f"Eigenvalues RMSE: {eigs_rmse:.4f} eV")
print(f"Weights RMSE: {weights_rmse:.4f} eV")

# 6. Plotting Parity Graphs
fig, ax = plt.subplots(1, 4, figsize=(24, 5))

# Energy Parity
ax[0].scatter(energy_true, aligned_energy_pred, alpha=0.5)
ax[0].plot([energy_true.min(), energy_true.max()], 
           [energy_true.min(), energy_true.max()], 'r--')
ax[0].set_title(f"Energy (RMSE: {energy_rmse:.3f} eV)")
ax[0].set_xlabel("DFT Energy (eV)")
ax[0].set_ylabel("ML Predicted (Aligned) (eV)")

# Forces Parity (Flattened)
f_true = np.concatenate(results["forces_true"]).flatten()
f_pred = np.concatenate(results["forces_pred"]).flatten()
ax[1].scatter(f_true, f_pred, alpha=0.3, s=1)
ax[1].plot([f_true.min(), f_true.max()], [f_true.min(), f_true.max()], 'r--')
ax[1].set_title(f"Forces (RMSE: {forces_rmse:.3f} eV/Å)")
ax[1].set_xlabel("DFT Forces")
ax[1].set_ylabel("ML Predicted")

# Eigenvalue Parity
eig_true = np.array(results["eigs_true"]).flatten()
eig_pred = np.array(results["eigs_pred"]).flatten()
ax[2].scatter(eig_true, eig_pred, alpha=0.1, s=0.5)
ax[2].plot([eig_true.min(), eig_true.max()], [eig_true.min(), eig_true.max()], 'r--')
ax[2].set_title(f"Eigenvalues (RMSE: {eigs_rmse:.3f} eV)")
ax[2].set_xlabel("DFT Eigenvalues")
ax[2].set_ylabel("ML Predicted")

# Weights Parity
weights_true = np.array(results["weights_true"]).flatten()
weights_pred = np.array(results["weights_pred"]).flatten()
ax[3].scatter(weights_true, weights_pred, alpha=0.1, s=0.5)
ax[3].plot([weights_true.min(), weights_true.max()], [weights_true.min(), weights_true.max()], 'r--')
ax[3].set_title(f"Weights (RMSE: {weights_rmse:.3f} eV)")
ax[3].set_xlabel("DFT Weights")
ax[3].set_ylabel("ML Predicted")

plt.tight_layout()
plt.savefig("cellulose_evaluation_parity.png")

# 7. Exporting Data to CSV for OriginPro
print("\nExporting raw data to CSV files...")

# 1. Energy CSV (1 row per frame)
np.savetxt(
    "cellulose_energy.csv", 
    np.column_stack((energy_true, aligned_energy_pred)), 
    delimiter=",", 
    header="Energy_True_eV,Energy_Pred_eV", 
    comments="" # Removes the '#' at the start of the header
)

# 2. Forces CSV (Flattened: 1 row per force component across all frames)
forces_true_flat = np.concatenate(results["forces_true"]).flatten()
forces_pred_flat = np.concatenate(results["forces_pred"]).flatten()
np.savetxt(
    "cellulose_forces.csv", 
    np.column_stack((forces_true_flat, forces_pred_flat)), 
    delimiter=",", 
    header="Forces_True_eV_A,Forces_Pred_eV_A", 
    comments=""
)

# 3. Eigenvalues CSV (Flattened: 1 row per eigenvalue across all frames)
eigs_true_flat = np.array(results["eigs_true"]).flatten()
eigs_pred_flat = np.array(results["eigs_pred"]).flatten()
np.savetxt(
    "cellulose_eigenvalues.csv", 
    np.column_stack((eigs_true_flat, eigs_pred_flat)), 
    delimiter=",", 
    header="Eigenvalues_True_eV,Eigenvalues_Pred_eV", 
    comments=""
)

# 4. Weights CSV (Flattened: 1 row per weight across all frames)
weights_true_flat = np.array(results["weights_true"]).flatten()
weights_pred_flat = np.array(results["weights_pred"]).flatten()
np.savetxt(
    "cellulose_weights.csv", 
    np.column_stack((weights_true_flat, weights_pred_flat)), 
    delimiter=",", 
    header="Weights_True_eV,Weights_Pred_eV", 
    comments=""
)

# Save the quick summary text file
with open("cellulose_evaluation_metrics.txt", "w") as f:
    f.write("--- Final MLIP Evaluation Metrics ---\n")
    f.write(f"Energy RMSE: {energy_rmse:.4f} eV\n")
    f.write(f"Forces RMSE: {forces_rmse:.4f} eV/A\n")
    f.write(f"Eigenvalues RMSE: {eigs_rmse:.4f} eV\n")
    f.write(f"Weights RMSE: {weights_rmse:.4f} eV\n")
    f.write(f"Baseline Offset Applied: {baseline_offset:.4f} eV\n")

print("Data successfully saved to 3 CSV files and the metrics summary!")

plt.show()