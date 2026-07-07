# @file train_electronic.py
# @copyright Copyright © 2026 Isaías Rodríguez (isurwars@gmail.com)
# @par License
# SPDX-License-Identifier: AGPL-3.0-only

"""
train_electronic.py — Electronic Structure Finetuning

Fine-tunes an Orb GNN backbone to predict two CASTEP electronic-structure
targets from first principles:

  * eigenvalues  (250 DFT Kohn-Sham band energies per structure, graph-level)
  * weights      (250 PDOS weights per atom, node-level)

The physics loss (energy / forces / stress) is intentionally suppressed so
the backbone learns to produce latent features that are useful for electronic
structure without drifting away from its interatomic potential pretraining.
Two lightweight MLP heads sit on top of the frozen backbone:
  - eigenvalue_head: mean-pooled node features → 250 band energies
  - weight_head:     per-node features → 250 PDOS weights (Softplus output)
"""

import argparse
import logging
import os
from collections.abc import Callable
from typing import Any

import ase
import torch
import torch.nn as nn
import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False

from orb_models.common.atoms.abstract_atoms_adapter import AbstractAtomsAdapter
from orb_models.common.dataset import augmentations, property_definitions
from orb_models.common.dataset.ase_sqlite_dataset import AseSqliteDataset
from orb_models.common.dataset.loaders import worker_init_fn
from orb_models.common.dataset.property_definitions import PROPERTIES, PropertyDefinition
from orb_models.common.models.base import ModelMixin
from orb_models.common.training.metrics import ScalarMetricTracker
from orb_models.common.training.util import get_optim, init_device
from orb_models.common.utils import seed_everything
from orb_models.forcefield import pretrained
from orb_models.common.dataset.property_definitions import PropertyDefinition
import numpy as np

def eigenvalues_row_fn(row, dataset: str):
    """Extract the 250-band eigenvalue array from an ASE SQLite row.

    Used as the ``row_to_property_fn`` for the temporary top-level
    PropertyDefinition registered before the DataLoader is built.  The final
    registry entry (inside ``run``) replaces this one with a picklable
    top-level function so multiprocessing workers can serialise it.
    """
    import torch
    val = row.data.get("eigenvalues")
    if val is None:
        raise ValueError(f"No eigenvalues in row {row.id}")
    return torch.from_numpy(np.array(val, dtype=np.float64))

# Pre-register eigenvalues so that any early property-config instantiation
# (e.g. inside AseSqliteDataset.__init__) can resolve the property name.
# This entry is overwritten inside run() with the fully picklable version.
property_definitions.PROPERTIES["eigenvalues"] = PropertyDefinition(
    name="eigenvalues",
    dim=250,
    domain="real",
    row_to_property_fn=eigenvalues_row_fn,
)

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Segment-mean pooling without requiring torch-scatter.

    Aggregates rows of ``src`` into ``dim_size`` output rows by averaging all
    source rows that share the same ``index`` value.  Avoids division-by-zero
    for segments with no contributing nodes via ``clamp(min=1)``.

    Args:
        src:   Node feature matrix of shape [N, F].
        index: Integer segment IDs of shape [N], mapping each node to its graph.
        dim:   Dimension along which to scatter (always 0 for node → graph).

    Returns:
        Tensor of shape [num_graphs, F] with per-graph mean features.
    """
    dim_size = int(index.max().item()) + 1
    out = src.new_zeros((dim_size, src.size(1)))
    out.index_add_(dim, index, src)
    count = src.new_zeros((dim_size, src.size(1)))
    count.index_add_(dim, index, torch.ones_like(src))
    return out / count.clamp(min=1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def prefix_keys[T](dict_to_prefix: dict[str, T], prefix: str, sep: str = "/") -> dict[str, T]:
    """Add a prefix to dictionary keys with a seperator."""
    return {f"{prefix}{sep}{k}": v for k, v in dict_to_prefix.items()}


def init_wandb_from_config(dataset: str, job_type: str, entity: str) -> Any:
    """Initialise wandb."""
    if not WANDB_AVAILABLE:
        raise ImportError(
            "wandb is not installed. Install with `pip install wandb` to enable logging."
        )

    wandb.init(  # type: ignore
        job_type=job_type,
        dir=os.path.join(os.getcwd(), "wandb"),
        name=f"{dataset}-{job_type}",
        project="orb-experiment",
        entity=entity,
        mode="online",
        sync_tensorboard=False,
    )
    assert wandb.run is not None
    return wandb.run


def finetune(
    model: ModelMixin,
    eigenvalue_head: nn.Module,
    weight_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    num_steps: int | None = None,
    clip_grad: float | None = None,
    log_freq: float = 10,
    device: torch.device = torch.device("cpu"),
    epoch: int = 0,
    accumulation_steps: int = 4,
    freeze_backbone: bool = True,
    eigenvalue_loss_weight: float = 1.0,
    weight_loss_weight: float = 1.0,
    energy_loss_weight: float = 0.0,
    forces_loss_weight: float = 0.0,
    is_conservative_model: bool = False,
):
    """Run one epoch of electronic-structure finetuning.

    Performs gradient-accumulation training over the dataloader.  The physics
    loss (energy/forces) is deliberately ignored; only the eigenvalue MSE and
    the peak-weighted PDOS weight loss are back-propagated.

    Args:
        model: The pretrained Orb GNN backbone (run in train mode).
        eigenvalue_head: MLP that maps mean-pooled node features to 250 band
            energies (graph-level target).
        weight_head: MLP that maps per-node features to 250 PDOS weights
            (node-level target, Softplus output).
        optimizer: Shared optimizer covering backbone + both heads.
        dataloader: PyTorch DataLoader; may be finite or stepped via num_steps.
        lr_scheduler: Optional LR scheduler, stepped after each optimizer update.
        num_steps: Hard cap on batches consumed per epoch (useful for large
            datasets where one full pass is prohibitively long).
        clip_grad: If set, gradient norms for all parameter groups are clipped
            to this value before each optimizer step.
        log_freq: Log aggregated metrics every this many steps.
        device: Device on which tensors live.
        epoch: Current epoch index (used to compute the global step for wandb).
        accumulation_steps: Number of forward passes whose gradients are summed
            before a single optimizer step (effective batch size multiplier).

    Returns:
        A dictionary of scalar metrics averaged over the epoch.
    """
    run: Any | None = wandb.run if WANDB_AVAILABLE else None

    metrics = ScalarMetricTracker()

    # Set the model to "train" mode (or "eval" if frozen).
    if freeze_backbone:
        model.eval()
    else:
        model.train()
    eigenvalue_head.train()
    weight_head.train()

    # Get tqdm for the training batches
    batch_generator = iter(dataloader)
    num_training_batches: int | float
    if num_steps is not None:
        num_training_batches = num_steps
    else:
        try:
            num_training_batches = len(dataloader)
        except TypeError:
            raise ValueError("Dataloader has no length, you must specify num_steps.")

    batch_generator_tqdm = tqdm.tqdm(batch_generator, total=num_training_batches)

    i = 0
    batch_iterator = iter(batch_generator_tqdm)
    while True:
        if num_steps and i == num_steps:
            break

        step_metrics = {
            "batch_size": 0.0,
            "batch_num_edges": 0.0,
            "batch_num_nodes": 0.0,
        }

        # Reset metrics so that it reports raw values for each step but still do averages on
        # the gradient accumulation.
        if i % log_freq == 0:
            metrics.reset()

        try:
            batch = next(batch_iterator)
        except StopIteration:
            break
        batch = batch.to(device)
        step_metrics["batch_size"] += len(batch.n_node)
        step_metrics["batch_num_edges"] += batch.n_edge.sum()
        step_metrics["batch_num_nodes"] += batch.n_node.sum()

        # Autocast is disabled (enabled=False) rather than using bfloat16.
        # The Orb model has float32 LayerNorm weights; enabling float16 autocast
        # feeds fp16 activations into those layers and triggers a kernel dispatch
        # mismatch.  bfloat16 would be safe, but keeping full float32 avoids any
        # precision loss when fine-tuning the sensitive electronic-structure heads.
        with torch.autocast("cuda", enabled=False):
            # --- 2. GNN Backbone forward pass ---
            gnn_out = model.model(batch)
            node_features = gnn_out["node_features"]          # [N_nodes, latent_dim]

            n_node = batch.n_node  # [num_graphs]
            graph_idx = torch.repeat_interleave(
                torch.arange(len(n_node), device=node_features.device), n_node
            )  # [total_nodes]
            graph_features = scatter_mean(node_features, graph_idx, dim=0)  # [num_graphs, latent_dim]

            # --- 3. Electronic structure predictions & loss ---
            pred_weights = weight_head(node_features)          # [N_nodes, 250] PDOS weights
            pred_eigenvalues = eigenvalue_head(graph_features) # [num_graphs, 250] band energies

            true_eigenvalues = batch.system_targets['eigenvalues']  # [num_graphs, 250]
            true_weights = batch.node_targets['weights']            # [N_nodes, 250]

            eig_loss = torch.nn.functional.mse_loss(pred_eigenvalues, true_eigenvalues)

            # Peak-weighted MSE for PDOS weights
            squared_errors = (pred_weights - true_weights) ** 2
            peak_multiplier = 1.0 + (true_weights * 20.0)
            magnitude_loss = torch.mean(squared_errors * peak_multiplier)

            # Masked Cramér (L2 Wasserstein) shape loss
            true_sums = true_weights.sum(dim=-1, keepdim=True)
            active_mask = (true_sums > 0.1).squeeze(-1)

            if active_mask.any():
                pred_weights_active = pred_weights[active_mask]
                true_weights_active = true_weights[active_mask]

                pred_pdf = pred_weights_active / (pred_weights_active.sum(dim=-1, keepdim=True) + 1e-8)
                true_pdf = true_weights_active / (true_weights_active.sum(dim=-1, keepdim=True) + 1e-8)
                pred_cdf = torch.cumsum(pred_pdf, dim=-1)
                true_cdf = torch.cumsum(true_pdf, dim=-1)

                cramer_loss = torch.mean((pred_cdf - true_cdf) ** 2)
            else:
                cramer_loss = torch.tensor(0.0, device=device)

            weight_loss = (3.0 * magnitude_loss) + (0.5 * cramer_loss)

            # --- 4. Physics predictions & loss (only if GNN is unfrozen) ---
            is_physics_active = (not freeze_backbone) and (energy_loss_weight > 0.0 or forces_loss_weight > 0.0)
            
            if is_physics_active:
                if is_conservative_model:
                    with torch.set_grad_enabled(True):
                        # Set requires_grad on positions to compute forces
                        batch.positions.requires_grad_(True)
                        
                        physics_out = model(batch)
                        pred_energy = physics_out["energy"]
                        pred_forces = physics_out["grad_forces"]
                else:
                    physics_out = model(batch)
                    pred_energy = physics_out["energy"]
                    pred_forces = physics_out["forces"]
                
                true_energy = batch.system_targets["energy"]
                true_forces = batch.node_targets["forces"]
                
                energy_loss = torch.nn.functional.mse_loss(pred_energy, true_energy)
                forces_loss = torch.nn.functional.huber_loss(pred_forces, true_forces, delta=0.15)
            else:
                energy_loss = torch.tensor(0.0, device=device)
                forces_loss = torch.tensor(0.0, device=device)

            # --- 5. Total loss ---
            total_loss = (
                (energy_loss_weight * energy_loss)
                + (forces_loss_weight * forces_loss)
                + (eigenvalue_loss_weight * eig_loss)
                + (weight_loss_weight * weight_loss)
            )
            scaled_loss = total_loss / accumulation_steps  # scale for gradient accumulation

            # --- 6. Logging ---
            batch_outputs = {}
            batch_outputs["loss/eigenvalues"] = eig_loss.detach()
            batch_outputs["loss/weights"] = weight_loss.detach()
            if is_physics_active:
                batch_outputs["loss/energy"] = energy_loss.detach()
                batch_outputs["loss/forces"] = forces_loss.detach()
            batch_outputs["loss/total"] = total_loss.detach()
            metrics.update(batch_outputs)

        if torch.isnan(scaled_loss):
            print(f"\n[Warning] NaN scaled_loss at step {i}. Skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            i += 1
            continue

        scaled_loss.backward()

        # Gradient accumulation: only update weights every `accumulation_steps`
        # batches (or on the final batch).  This simulates a larger effective
        # batch size without the memory overhead of a physically larger batch.
        if (i + 1) % accumulation_steps == 0 or (i + 1) == num_training_batches:
            if clip_grad is not None:
                # Clip gradients for all three parameter groups independently
                # to prevent instability when the heads first start training.
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                torch.nn.utils.clip_grad_norm_(eigenvalue_head.parameters(), clip_grad)
                torch.nn.utils.clip_grad_norm_(weight_head.parameters(), clip_grad)

            optimizer.step()

            # Release gradient tensors immediately to free memory for the next
            # accumulation window (set_to_none is more efficient than zeroing).
            optimizer.zero_grad(set_to_none=True)

        metrics.update(step_metrics)

        if i != 0 and i % log_freq == 0:
            metrics_dict = metrics.get_metrics()
            if run is not None:
                step = (epoch * num_training_batches) + i
                if run.sweep_id is not None:
                    run.log({"loss": metrics_dict["loss/total"]}, commit=False)
                run.log({"step": step}, commit=False)
                run.log(prefix_keys(metrics_dict, "finetune_step"), commit=True)

        i += 1

    return metrics.get_metrics()


def build_train_loader(
    dataset_name: str,
    dataset_path: str,
    num_workers: int,
    batch_size: int,
    atoms_adapter: AbstractAtomsAdapter,
    augmentation: bool | None = True,
    target_config: dict | None = None,
    **kwargs,
) -> DataLoader:
    """Builds the train dataloader from a config file.

    Args:
        dataset_name: The name of the dataset.
        dataset_path: Dataset path.
        num_workers: The number of workers for each dataset.
        batch_size: The batch_size config for each dataset.
        atoms_adapter: The atoms adapter for converting ase.Atoms to model-specific AbstractAtomBatch instances.
        augmentation: If rotation augmentation is used.
        target_config: The target config.

    Returns:
        The train Dataloader.
    """
    log_train = "Loading train datasets:\n"
    aug: list[Callable[[ase.Atoms], None]] = []
    if augmentation:
        aug = [augmentations.rotate_randomly]

    target_property_config = property_definitions.instantiate_property_config(target_config)
    dataset = AseSqliteDataset(
        dataset_name,
        dataset_path,
        atoms_adapter=atoms_adapter,
        target_config=target_property_config,
        augmentations=aug,
        **kwargs,
    )

    log_train += f"Total train dataset size: {len(dataset)} samples"
    logging.info(log_train)

    sampler = RandomSampler(dataset)

    batch_sampler = BatchSampler(
        sampler,
        batch_size=batch_size,
        drop_last=False,
    )

    train_loader: DataLoader = DataLoader(
        dataset,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=atoms_adapter.batch,
        batch_sampler=batch_sampler,
        timeout=10 * 60 if num_workers > 0 else 0,
    )
    return train_loader


def load_custom_reference_energies(filepath: str) -> torch.Tensor:
    """
    Load custom reference energies from a file.

    Supports two formats:
    1. JSON: {"1": -13.6, "6": -1030.5, ...} or {"H": -13.6, "C": -1030.5, ...}
    2. Text: One line per element: "element_number energy" or "element_symbol energy"

    Args:
        filepath: Path to the reference energies file

    Returns:
        Tensor of shape [118] with reference energies
    """
    import json

    # Element symbol to atomic number mapping
    ELEMENT_SYMBOLS = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
        # ... (rest of elements omitted for brevity, keep the original list in your file)
        "U": 92
    }

    ref_energies = torch.zeros(118)

    # Try to load as JSON first
    try:
        with open(filepath) as f:
            data = json.load(f)

        for key, value in data.items():
            try:
                z = int(key)
                if 1 <= z <= 118:
                    ref_energies[z] = float(value)
            except ValueError:
                if key in ELEMENT_SYMBOLS:
                    z = ELEMENT_SYMBOLS[key]
                    ref_energies[z] = float(value)
                else:
                    logging.warning(f"Unknown element symbol or invalid atomic number: {key}")

        logging.info(f"Loaded reference energies from JSON file: {filepath}")

    except json.JSONDecodeError:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) != 2:
                    logging.warning(f"Skipping invalid line: {line}")
                    continue

                element, energy = parts
                try:
                    z = int(element)
                    if 1 <= z <= 118:
                        ref_energies[z] = float(energy)
                except ValueError:
                    if element in ELEMENT_SYMBOLS:
                        z = ELEMENT_SYMBOLS[element]
                        ref_energies[z] = float(energy)
                    else:
                        logging.warning(f"Unknown element: {element}")

        logging.info(f"Loaded reference energies from text file: {filepath}")

    return ref_energies

# These extraction functions must be defined at module top-level (not as
# lambdas or nested functions) so that Python's multiprocessing 'spawn' mode
# can pickle them for DataLoader worker processes.

def extract_eigenvalues(row, dataset=None):
    """Return the 250 CASTEP Kohn-Sham eigenvalues for a structure as float32."""
    return torch.tensor(row.data["eigenvalues"], dtype=torch.float32)

def extract_weights(row, dataset=None):
    """Return the per-atom 250-band PDOS weight vector for a structure as float32."""
    return torch.tensor(row.data["weights"], dtype=torch.float32)

def evaluate_model(model, eigenvalue_head, weight_head, atoms_adapter, eval_frames, device, plot_path=None):
    """Evaluate current model checkpoint on cached validation frames."""
    model.eval()
    eigenvalue_head.eval()
    weight_head.eval()

    results = {
        "forces_true": [], "forces_pred": [],
        "eigs_true": [], "eigs_pred": [],
        "weights_true": [], "weights_pred": []
    }

    is_conservative = model.__class__.__name__ == "ConservativeForcefieldRegressor"

    for single_graph, gt in eval_frames:
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

            pred_eigs = eigenvalue_head(graph_feats).cpu().numpy().flatten()
            pred_weights = weight_head(node_feats).cpu().numpy().flatten()

            results["eigs_true"].append(gt["eigenvalues"])
            results["eigs_pred"].append(pred_eigs)
            results["weights_true"].append(np.array(gt["weights"]).flatten())
            results["weights_pred"].append(pred_weights)

    # Calculate RMSE
    f_true = np.concatenate(results["forces_true"]).flatten()
    f_pred = np.concatenate(results["forces_pred"]).flatten()
    forces_rmse = np.sqrt(np.mean((f_true - f_pred)**2))

    eig_true = np.array(results["eigs_true"]).flatten()
    eig_pred = np.array(results["eigs_pred"]).flatten()
    eigs_rmse = np.sqrt(np.mean((eig_true - eig_pred)**2))

    w_true = np.concatenate(results["weights_true"])
    w_pred = np.concatenate(results["weights_pred"])
    weights_rmse = np.sqrt(np.mean((w_true - w_pred)**2))

    if plot_path is not None:
        import matplotlib.pyplot as plt
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
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close(fig)
        logging.info(f"Saved parity plot to {plot_path}")

    return {
        "forces_rmse": forces_rmse,
        "eigs_rmse": eigs_rmse,
        "weights_rmse": weights_rmse
    }


def run(args):
    """Training Loop.

    Args:
        args: Config for training loop.
    """
    device = init_device(device_id=args.device_id)
    seed_everything(args.random_seed)

    precision = "float32-high"

    # Prepare loss weights if specified
    loss_weights = {}
    is_conservative_model = "conservative" in args.base_model

    if args.energy_loss_weight is not None:
        loss_weights["energy"] = args.energy_loss_weight

    if args.forces_loss_weight is not None:
        if is_conservative_model:
            loss_weights["grad_forces"] = args.forces_loss_weight
        else:  
            loss_weights["forces"] = args.forces_loss_weight

    if args.stress_loss_weight is not None:
        if is_conservative_model:
            loss_weights["grad_stress"] = args.stress_loss_weight
        else: 
            loss_weights["stress"] = args.stress_loss_weight

    if args.equigrad_loss_weight is not None and args.equigrad_loss_weight > 0.0:
        if not is_conservative_model:
            raise ValueError("Equigrad loss is only available for conservative models.")
        loss_weights["rotational_grad"] = args.equigrad_loss_weight

    if loss_weights:
        logging.info("=" * 60)
        logging.info("Custom loss weights specified:")
        for key, val in loss_weights.items():
            logging.info(f"  {key}: {val}")
        logging.info("=" * 60)

    # Instantiate model with configuration
    base_model = args.base_model
    model, atoms_adapter = getattr(pretrained, base_model)(
        device=device,
        precision=precision,
        train=True,
        train_reference_energies=args.trainable_reference_energies,
        loss_weights=loss_weights if loss_weights else None,
    )

    # Handle custom reference energies if provided
    if args.custom_reference_energies:
        logging.info("=" * 60)
        logging.info(f"Loading custom reference energies from: {args.custom_reference_energies}")
        custom_refs = load_custom_reference_energies(args.custom_reference_energies)
        custom_refs = custom_refs.to(device)

        model.heads["energy"].reference.linear.weight.data = custom_refs

        logging.info("Custom reference energies set:")
        for z in [1, 6, 7, 8]:  # H, C, N, O
            val = custom_refs[z].item()
            if val != 0:
                logging.info(f"  Element {z}: {val:.4f} eV")

        if args.trainable_reference_energies:
            logging.info("Custom reference energies will be trainable during finetuning")
        else:
            logging.info("Custom reference energies are FIXED (not trainable)")
        logging.info("=" * 60)

    if args.stress_loss_weight is not None:
        if args.stress_loss_weight > 0:
            model.enable_stress()
            logging.info("Stress training ENABLED (stress_loss_weight=%.4f)", args.stress_loss_weight)
        elif model.has_stress:
            model.disable_stress()
            logging.info("Stress training DISABLED (stress_loss_weight=0.0)")

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Base Model has {model_params:,} trainable parameters.")

    # Graph-level head: predicts 250 Kohn-Sham band energies from the
    # mean-pooled node embedding.  A single hidden layer is sufficient because
    # the eigenvalue spectrum is a smooth, ordered quantity.
    latent_dim = 256  # Node embedding dimensionality of orb_v3 omol models.

    eigenvalue_head = nn.Sequential(
        nn.LayerNorm(latent_dim),
        nn.Linear(latent_dim, 1024),
        nn.SiLU(),
        nn.LayerNorm(1024),
        nn.Linear(1024, 1024),
        nn.SiLU(),
        nn.Linear(1024, 250),   # Output: 250 band energies per structure
    ).to(device)

    # Node-level head: predicts 250 PDOS weights per atom.  An extra hidden
    # layer gives more capacity for the per-atom spectral decomposition.
    # Softplus ensures non-negative outputs (weights are physically positive).
    weight_head = nn.Sequential(
        nn.LayerNorm(latent_dim),
        nn.Linear(latent_dim, 1024),
        nn.SiLU(),
        nn.LayerNorm(1024),
        nn.Linear(1024, 1024),
        nn.SiLU(),
        nn.Linear(1024, 250),
        nn.Softplus(),          # Guarantees weights ≥ 0
    ).to(device)

    # Initialize the bias of the final linear layer in the weight head
    # to a negative value (-4.5). This shifts the initial outputs of the
    # Softplus function to match the typical scale of target weights (~0.01).
    # This prevents the initial loss from being huge and completely avoids
    # the vanishing gradient problem in the flat region of Softplus.
    nn.init.constant_(weight_head[-2].bias, -4.5)

    model.to(device=device)

    # Determine whether backbone parameters should be included in optimizer from epoch 0
    include_backbone_in_optimizer = (not args.freeze_backbone) or (args.unfreeze_epoch is not None)
    
    # Initially freeze GNN parameters if requested (even if unfreeze_epoch is set)
    if args.freeze_backbone:
        logging.info("Initially freezing GNN backbone parameters.")
        for param in model.parameters():
            param.requires_grad = False

    import re
    params = []
    # Exclude bias, LayerNorm, and BatchNorm parameters from weight decay.
    # Regularising these normalisation parameters can destabilise training.
    if include_backbone_in_optimizer:
        # Determine initial learning rate for GNN backbone.
        # If unfreeze_epoch is specified, we start with 0.0 learning rate.
        init_backbone_lr = 0.0 if (args.freeze_backbone and args.unfreeze_epoch is not None) else args.backbone_lr
        logging.info(f"Including GNN backbone in optimizer with initial LR: {init_backbone_lr}")
        
        for name, param in model.named_parameters():
            if re.search(r"(.*bias|.*layer_norm.*|.*batch_norm.*)", name):
                params.append({"params": param, "weight_decay": 0.0, "lr": init_backbone_lr})
            else:
                params.append({"params": param, "lr": init_backbone_lr})
    else:
        logging.info("Excluding GNN backbone parameters from optimizer parameter list (permanently frozen).")

    # The two custom heads are trained from scratch and do not need special
    # weight-decay treatment; they use the global default.
    params.append({"params": eigenvalue_head.parameters(), "lr": args.lr})
    params.append({"params": weight_head.parameters(), "lr": args.lr})

    optimizer = torch.optim.Adam(params)

    start_epoch = 0
    if args.resume_from_checkpoint:
        logging.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        eigenvalue_head.load_state_dict(checkpoint["eigenvalue_head_state"])
        weight_head.load_state_dict(checkpoint["weight_head_state"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as e:
            logging.warning(f"Could not load optimizer state (might be incompatible): {e}")
        start_epoch = checkpoint.get("epoch", -1) + 1
        logging.info(f"Resumed at epoch: {start_epoch}")
        
        # If resuming at or after unfreeze epoch, ensure GNN parameters have gradients enabled
        if args.unfreeze_epoch is not None and start_epoch >= args.unfreeze_epoch:
            logging.info("Resuming after unfreeze epoch. Ensuring GNN backbone parameters are unfrozen.")
            for param in model.parameters():
                param.requires_grad = True
    
    # Learning rate scheduler initialization (stepped once per epoch)
    cosine_start_epoch = None
    if args.scheduler == "cosine":
        logging.info("Initializing CosineAnnealingLR scheduler.")
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs, eta_min=args.min_lr
        )
        cosine_start_epoch = 0
    elif args.scheduler == "flat_cosine":
        logging.info("Initializing Flat-Cosine (SequentialLR) scheduler.")
        if args.unfreeze_epoch is not None:
            # If GNN unfreezes, scheduler resets at that epoch.
            # The final cosine start epoch is unfreeze_epoch + (max_epochs - unfreeze_epoch) // 2
            cosine_start_epoch = args.unfreeze_epoch + (args.max_epochs - args.unfreeze_epoch) // 2
        else:
            cosine_start_epoch = args.max_epochs // 2

        T_flat = args.max_epochs // 2
        if T_flat > 0:
            scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=T_flat)
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs - T_flat, eta_min=args.min_lr)
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[T_flat])
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.min_lr)
    elif args.scheduler == "plateau":
        logging.info("Initializing ReduceLROnPlateau scheduler (stepped per epoch).")
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=args.min_lr
        )
    else:
        logging.info("No learning rate scheduler specified (constant learning rate).")
        lr_scheduler = None

    if args.resume_from_checkpoint and "lr_scheduler" in checkpoint:
        try:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            logging.info("Loaded lr_scheduler state from checkpoint.")
        except Exception as e:
            logging.warning(f"Could not load lr_scheduler state: {e}")
    
    wandb_run = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb flag is set but wandb is not installed.")
        else:
            logging.info("Instantiating WandbLogger.")
            wandb_run = init_wandb_from_config(
                dataset=args.dataset, job_type="finetuning", entity=args.wandb_entity
            )
            wandb.define_metric("step")
            wandb.define_metric("finetune_step/*", step_metric="step")
    # Register CASTEP electronic-structure properties in the Orb property
    # registry.  This overwrites the placeholder entry made at import time with
    # fully picklable top-level functions, which is required for DataLoader
    # workers launched under the 'spawn' multiprocessing start method.
    PROPERTIES["eigenvalues"] = PropertyDefinition(
        name="eigenvalues",
        dim=250,           # 250 Kohn-Sham band energies per structure
        domain="graph",    # Graph-level: one vector per crystal structure
        row_to_property_fn=extract_eigenvalues,
    )
    PROPERTIES["weights"] = PropertyDefinition(
        name="weights",
        dim=250,           # 250 PDOS weights per atom
        domain="node",     # Node-level: one vector per atom
        row_to_property_fn=extract_weights,
    )

    # Build the target config: eigenvalues are a graph-level target;
    # forces and PDOS weights are node-level targets.
    graph_targets = ["energy", "stress"] if model.has_stress else ["energy"]
    graph_targets.append("eigenvalues")
    loader_args = dict(
        dataset_name=args.dataset,
        dataset_path=args.data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        target_config={"graph": graph_targets, "node": ["forces", "weights"]},
    )
    
    train_loader = build_train_loader(
        **loader_args,
        atoms_adapter=atoms_adapter,
        augmentation=True,
    )
    
    # Preprocess and cache validation database frames for training evaluation
    eval_frames = []
    if args.eval_every_x_epochs > 0:
        logging.info("Preprocessing and caching database frames for evaluation...")
        import ase.db
        db = ase.db.connect(args.data_path)
        for row in db.select():
            test_atoms = row.toatoms()
            single_graph = atoms_adapter.from_ase_atoms(test_atoms)
            gt = {
                "energy": row.energy if hasattr(row, "energy") else None,
                "forces": row.forces if hasattr(row, "forces") else None,
                "eigenvalues": row.data.get("eigenvalues") if "eigenvalues" in row.data else None,
                "weights": row.data.get("weights") if "weights" in row.data else None,
                "cell": test_atoms.get_cell().array
            }
            eval_frames.append((single_graph, gt))
        logging.info(f"Cached {len(eval_frames)} frames for evaluation.")

    best_monitored_val = None
    patience_counter = 0

    logging.info("Starting training!")

    num_steps = args.num_steps if args.num_steps > 0 else None

    for epoch in range(start_epoch, args.max_epochs):
        # Dynamic unfreezing check
        is_currently_frozen = args.freeze_backbone and (args.unfreeze_epoch is None or epoch < args.unfreeze_epoch)
        
        # Ensure requires_grad matches unfreezing state
        for param in model.parameters():
            param.requires_grad = not is_currently_frozen
            
        # Trigger unfreezing at target epoch
        if args.unfreeze_epoch is not None and epoch == args.unfreeze_epoch:
            logging.info(f"--- Unfreezing GNN backbone at epoch {epoch} ---")
            head_params_set = set(list(eigenvalue_head.parameters()) + list(weight_head.parameters()))
            for idx, group in enumerate(optimizer.param_groups):
                is_backbone = any(p not in head_params_set for p in group["params"])
                if is_backbone:
                    group["lr"] = args.backbone_lr
                    logging.info(f"  GNN Backbone group {idx} LR set to: {args.backbone_lr}")
                else:
                    group["lr"] = args.lr
                    logging.info(f"  Head group {idx} LR reset to: {args.lr}")
                # Remove cached initial_lr so the new scheduler picks up the updated learning rate
                group.pop("initial_lr", None)
            
            # Re-initialize the scheduler to start decay from this epoch
            if args.scheduler == "cosine":
                logging.info(f"Re-initializing CosineAnnealingLR scheduler with T_max={args.max_epochs - epoch}")
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.max_epochs - epoch, eta_min=args.min_lr
                )
                cosine_start_epoch = epoch
            elif args.scheduler == "flat_cosine":
                remaining_epochs = args.max_epochs - epoch
                logging.info(f"Re-initializing Flat-Cosine scheduler with remaining={remaining_epochs}")
                T_flat = remaining_epochs // 2
                if T_flat > 0:
                    scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=T_flat)
                    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs - T_flat, eta_min=args.min_lr)
                    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[T_flat])
                    cosine_start_epoch = epoch + T_flat
                else:
                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=args.min_lr)
                    cosine_start_epoch = epoch

        # Apply weight noise perturbation to break false minima
        # Only inject noise if we haven't reached the cosine scheduler phase
        is_cosine_phase = cosine_start_epoch is not None and epoch >= cosine_start_epoch
        if (
            args.weight_head_noise_std > 0 
            and epoch > 0 
            and epoch % args.weight_head_noise_interval == 0
            and not is_cosine_phase
        ):
            logging.info(f"Injecting random noise filter (std={args.weight_head_noise_std}) into weight_head parameters to break false minimum.")
            with torch.no_grad():
                for param in weight_head.parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * args.weight_head_noise_std
                        param.add_(noise)

        # Print learning rate at start of epoch
        #current_lrs = [group['lr'] for group in optimizer.param_groups]
        #logging.info(f"Start epoch {epoch} - Learning Rate: {current_lrs}")
        
        print(f"Start epoch: {epoch} training...")
        
        epoch_metrics = finetune(
            model=model,
            eigenvalue_head=eigenvalue_head,
            weight_head=weight_head,
            optimizer=optimizer,
            dataloader=train_loader,
            clip_grad=args.gradient_clip_val,
            device=device,
            num_steps=num_steps,
            epoch=epoch,
            accumulation_steps=args.accumulation_steps,
            freeze_backbone=is_currently_frozen,
            eigenvalue_loss_weight=args.eigenvalue_loss_weight,
            weight_loss_weight=args.weight_loss_weight,
            energy_loss_weight=args.energy_loss_weight,
            forces_loss_weight=args.forces_loss_weight,
            is_conservative_model=is_conservative_model,
        )

        # Step the learning rate scheduler once per epoch
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(epoch_metrics["loss/total"])
                logging.info(f"Stepped ReduceLROnPlateau with loss: {epoch_metrics['loss/total']:.4f}")
            else:
                lr_scheduler.step()
                logging.info("Stepped CosineAnnealingLR.")

        # Determine if we should save checkpoint at this epoch
        is_ckpt_epoch = (epoch % args.save_every_x_epochs == 0) or (epoch == args.max_epochs - 1)

        # Periodical evaluation and early stopping checks
        if args.eval_every_x_epochs > 0 and (epoch % args.eval_every_x_epochs == 0 or epoch == args.max_epochs - 1):
            plot_path = None
            if is_ckpt_epoch:
                plot_path = os.path.join(args.checkpoint_path, f"cellulose__epoch{epoch}.png")

            eval_metrics = evaluate_model(
                model=model,
                eigenvalue_head=eigenvalue_head,
                weight_head=weight_head,
                atoms_adapter=atoms_adapter,
                eval_frames=eval_frames,
                device=device,
                plot_path=plot_path
            )
            logging.info("=" * 60)
            logging.info(f"Epoch {epoch} Evaluation Metrics:")
            logging.info(f"  Eigenvalues RMSE: {eval_metrics['eigs_rmse']:.4f} eV")
            logging.info(f"  Weights RMSE:     {eval_metrics['weights_rmse']:.4f}")
            logging.info(f"  Forces RMSE:      {eval_metrics['forces_rmse']:.4f} eV/Å")
            logging.info("=" * 60)
            
            # Log to wandb if enabled
            if wandb_run is not None:
                wandb_run.log({
                    "eval/eigs_rmse": eval_metrics["eigs_rmse"],
                    "eval/weights_rmse": eval_metrics["weights_rmse"],
                    "eval/forces_rmse": eval_metrics["forces_rmse"],
                    "epoch": epoch
                })

            # Metrics explosion safety check (only active once GNN backbone is unfrozen)
            is_unfrozen = args.unfreeze_epoch is None or epoch >= args.unfreeze_epoch
            if is_unfrozen and (eval_metrics['eigs_rmse'] > 5.0 or eval_metrics['forces_rmse'] > 2.0):
                logging.warning("Exploding metrics detected! Terminating training run early.")
                break

            # Early stopping check
            if args.early_stopping_patience > 0:
                monitored_val = eval_metrics[args.early_stopping_metric]
                
                if best_monitored_val is None or monitored_val < best_monitored_val:
                    best_monitored_val = monitored_val
                    patience_counter = 0
                    logging.info(f"New best {args.early_stopping_metric}: {best_monitored_val:.4f}. Resetting patience counter.")
                else:
                    # Only increment patience counter if we are at or past the unfreeze_epoch
                    if args.unfreeze_epoch is None or epoch >= args.unfreeze_epoch:
                        patience_counter += 1
                        logging.info(f"{args.early_stopping_metric} did not improve. Patience: {patience_counter}/{args.early_stopping_patience}")
                        
                        if patience_counter >= args.early_stopping_patience:
                            logging.warning(f"Early stopping triggered! {args.early_stopping_metric} did not improve for {args.early_stopping_patience} evaluations.")
                            break
                    else:
                        logging.info(f"Patience counter not incremented because epoch {epoch} < unfreeze_epoch {args.unfreeze_epoch}.")

        # Save every X epochs and final epoch
        if (epoch % args.save_every_x_epochs == 0) or (epoch == args.max_epochs - 1):
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            
            checkpoint_data = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "eigenvalue_head_state": eigenvalue_head.state_dict(),
                "weight_head_state": weight_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None
            }
            
            torch.save(
                checkpoint_data,
                os.path.join(args.checkpoint_path, f"checkpoint_epoch{epoch}.ckpt"),
            )
            logging.info(f"Checkpoint saved to {args.checkpoint_path}")

    if wandb_run is not None:
        wandb_run.finish()


def main():
    """Main."""
    parser = argparse.ArgumentParser(
        description="Finetune orb model with custom loss weights and reference energy control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--random_seed", default=1234, type=int, help="Random seed for finetuning.")
    parser.add_argument("--device_id", default=0, type=int, help="GPU index to use if GPU is available.")
    parser.add_argument("--wandb", default=False, action="store_true", help="Log to Weights and Biases.")
    parser.add_argument("--wandb_entity", default="orbitalmaterials", type=str)
    parser.add_argument("--dataset", default="mp-traj", type=str)
    parser.add_argument("--data_path", default=os.path.join(os.getcwd(), "datasets/mptraj/finetune.db"), type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--gradient_clip_val", default=0.5, type=float)
    parser.add_argument("--max_epochs", default=50, type=int)
    parser.add_argument("--save_every_x_epochs", default=5, type=int)
    parser.add_argument("--num_steps", default=100, type=int)
    parser.add_argument("--checkpoint_path", default=os.path.join(os.getcwd(), "ckpts_electronic"), type=str)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str, help="Path to checkpoint to resume training.")
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--base_model", default="orb_v3_direct_inf_omat", type=str)
    parser.add_argument("--energy_loss_weight", default=0.0, type=float)
    parser.add_argument("--forces_loss_weight", default=0.0, type=float)
    parser.add_argument("--stress_loss_weight", default=0.0, type=float)
    parser.add_argument("--equigrad_loss_weight", default=0.0, type=float)
    parser.add_argument("--trainable_reference_energies", action="store_true")
    parser.add_argument("--custom_reference_energies", default=None, type=str)
    parser.add_argument("--accumulation_steps", default=4, type=int, help="Number of batches to accumulate gradients")
    parser.add_argument("--no_freeze_backbone", action="store_false", dest="freeze_backbone", help="Train the GNN backbone parameters (do not freeze).")
    parser.add_argument("--backbone_lr", default=1e-5, type=float, help="Learning rate for GNN backbone.")
    parser.add_argument("--unfreeze_epoch", default=None, type=int, help="Epoch at which to unfreeze the GNN backbone.")
    parser.add_argument("--scheduler", default="cosine", choices=["none", "cosine", "flat_cosine", "plateau"], help="Learning rate scheduler to use (stepped once per epoch).")
    parser.add_argument("--min_lr", default=1e-6, type=float, help="Minimum learning rate for the scheduler.")
    parser.add_argument("--weight_head_noise_std", default=0.0, type=float, help="Standard deviation of noise to inject into weight_head parameters to break false minima.")
    parser.add_argument("--weight_head_noise_interval", default=5, type=int, help="Epoch interval at which noise is injected into weight_head.")
    parser.add_argument("--eigenvalue_loss_weight", default=0.02, type=float, help="Loss weight scaling factor for eigenvalues.")
    parser.add_argument("--weight_loss_weight", default=1.0, type=float, help="Loss weight scaling factor for PDOS weights.")
    parser.add_argument("--eval_every_x_epochs", default=1, type=int, help="Frequency of running evaluation on cached database frames. Set to 0 to disable.")
    parser.add_argument("--early_stopping_patience", default=5, type=int, help="Patience (in evaluations) for early stopping. Set to 0 to disable.")
    parser.add_argument("--early_stopping_metric", default="forces_rmse", choices=["forces_rmse", "eigs_rmse", "weights_rmse"], help="Metric to monitor for early stopping.")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()