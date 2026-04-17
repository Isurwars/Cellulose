import torch
import torch.nn as nn
import numpy as np
import ase.io
from orb_models.forcefield.pretrained import orb_v3_conservative_omol
from torch_scatter import scatter_mean

# 1. Recreate the Spectral Head Architecture
# This MUST match exactly what you added to finetune.py
latent_dim = 256 # Standard for orb_v3_conservative_omol
spectral_head = nn.Sequential(
    nn.Linear(latent_dim, 1024),
    nn.SiLU(),
    nn.Linear(1024, 250)
).to("cuda")

# 2. Load the Base Model and the Checkpoint
model, atoms_adapter = orb_v3_conservative_omol(train=False, device="cuda")

# Load your specific checkpoint
checkpoint_path = "ckpts/checkpoint_epoch99.ckpt" # Adjust epoch number if needed
checkpoint = torch.load(checkpoint_path, map_location="cuda")

# Load weights into both models
model.load_state_dict(checkpoint["state_dict"])
spectral_head.load_state_dict(checkpoint["spectral_head_state"])

# Set both to evaluation mode
model.eval()
spectral_head.eval()

# 3. Prepare Test Data
# You can load from an xyz or directly from your finetuning.db
# Let's assume you extract an ASE Atoms object from your db:
import ase.db
db = ase.db.connect('cellulose_finetuning.db')
row = db.get(id=1)
test_atoms = row.toatoms()

# Convert ASE Atoms to an Orb Graph, then batch it
single_graph = atoms_adapter.from_ase_atoms(test_atoms)
inputs = atoms_adapter.batch([single_graph]).to("cuda")

# MANUALLY INJECT the required electronic properties as tensors.
# The OMol model expects them in the `system_features` attribute as 1D tensors.
inputs.system_features = {
    "total_charge": torch.tensor([0.0], dtype=torch.float32, device="cuda"),
    "spin_multiplicity": torch.tensor([1.0], dtype=torch.float32, device="cuda")
}

# 4. Perform Inference
with torch.no_grad():
    # Get base GNN features
    gnn_out = model.model(inputs)
    node_features = gnn_out["node_features"]
    
    # Since we only have 1 test frame, pool the atoms using a standard PyTorch mean
    graph_features = node_features.mean(dim=0, keepdim=True)
        
    # Predict the Eigenvalues
    predicted_eigenvalues = spectral_head(graph_features)

# 5. Output and Verification
pred_eigs_np = predicted_eigenvalues.cpu().numpy()[0] # Remove batch dimension

print("--- Eigenvalue Prediction Test ---")
print(f"Shape of output: {pred_eigs_np.shape}")
print(f"First 5 Eigenvalues (eV): {pred_eigs_np[:5]}")
print(f"Last 5 Eigenvalues (eV): {pred_eigs_np[-5:]}")

# Optional: Compare against ground truth if you loaded from the DB
if 'eigenvalues' in row.data:
    true_eigs = row.data['eigenvalues']
    mse = np.mean((pred_eigs_np - true_eigs)**2)
    print(f"\nMean Squared Error vs Ground Truth: {mse:.4f}")