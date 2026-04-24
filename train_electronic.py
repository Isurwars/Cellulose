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
    lr_scheduler: _LRScheduler | None = None,
    num_steps: int | None = None,
    clip_grad: float | None = None,
    log_freq: float = 10,
    device: torch.device = torch.device("cpu"),
    epoch: int = 0,
    accumulation_steps: int = 4,
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

    # Set the model to "train" mode.
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

        batch = next(batch_iterator)
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
            # --- 1. Backbone forward pass ---
            # Call the raw GNN (model.model) directly to obtain node embeddings
            # without triggering the built-in physics-loss heads.  The physics
            # loss is not needed here; we only want the latent features.
            gnn_out = model.model(batch)
            node_features = gnn_out["node_features"]          # [N_nodes, latent_dim]
            graph_features = node_features.mean(dim=0, keepdim=True)  # [1, latent_dim]

            # --- 2. Electronic structure predictions ---
            pred_weights = weight_head(node_features)          # [N_nodes, 250] PDOS weights
            pred_eigenvalues = eigenvalue_head(graph_features) # [1, 250] band energies

            true_eigenvalues = batch.system_targets['eigenvalues']  # [1, 250]
            true_weights = batch.node_targets['weights']            # [N_nodes, 250]

            # --- 3. Loss calculations ---
            # Band energy MSE — straightforward regression on the 250 eigenvalues.
            eig_loss = torch.nn.functional.mse_loss(pred_eigenvalues, true_eigenvalues)

            # Peak-weighted MSE for PDOS weights.  The PDOS is sparse: most
            # weights are near zero and only a handful of peaks carry signal.
            # Multiplying squared errors by (1 + 20 * true_weight) up-weights
            # the loss on those peaks so they dominate the gradient signal.
            squared_errors = (pred_weights - true_weights) ** 2
            peak_multiplier = 1.0 + (true_weights * 20.0)
            magnitude_loss = torch.mean(squared_errors * peak_multiplier)

            # Cosine-similarity shape loss encourages the predicted PDOS profile
            # to match the overall spectral shape, independent of scale.
            cos_sim = torch.nn.functional.cosine_similarity(pred_weights, true_weights, dim=-1)
            shape_loss = torch.mean(1.0 - cos_sim)

            # Combined weight loss: magnitude fidelity weighted 3× more than
            # shape fidelity, empirically chosen to balance the two objectives.
            weight_loss = (3.0 * magnitude_loss) + (0.5 * shape_loss)

            # --- 4. Total loss — electronic structure only ---
            # The backbone physics heads (energy, forces, stress) are not used;
            # their loss weights are set to 0.0 via CLI arguments.
            total_loss = eig_loss + weight_loss
            scaled_loss = total_loss / accumulation_steps  # scale for gradient accumulation

            # --- 5. Logging ---
            batch_outputs = {}
            batch_outputs["loss/eigenvalues"] = eig_loss.detach()
            batch_outputs["loss/weights"] = weight_loss.detach()
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

            if lr_scheduler is not None:
                lr_scheduler.step(scaled_loss.detach())

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

    if args.equigrad_loss_weight is not None:
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
        nn.Linear(latent_dim, 1024),
        nn.SiLU(),
        nn.Linear(1024, 250),   # Output: 250 band energies per structure
    ).to(device)

    # Node-level head: predicts 250 PDOS weights per atom.  An extra hidden
    # layer gives more capacity for the per-atom spectral decomposition.
    # Softplus ensures non-negative outputs (weights are physically positive).
    weight_head = nn.Sequential(
        nn.Linear(latent_dim, 1024),
        nn.SiLU(),
        nn.Linear(1024, 1024),
        nn.SiLU(),
        nn.Linear(1024, 250),
        nn.Softplus(),          # Guarantees weights ≥ 0
    ).to(device)

    model.to(device=device)

    import re
    params = []
    # Exclude bias, LayerNorm, and BatchNorm parameters from weight decay.
    # Regularising these normalisation parameters can destabilise training.
    for name, param in model.named_parameters():
        if re.search(r"(.*bias|.*layer_norm.*|.*batch_norm.*)", name):
            params.append({"params": param, "weight_decay": 0.0})
        else:
            params.append({"params": param})
    # The two custom heads are trained from scratch and do not need special
    # weight-decay treatment; they use the global default.
    params.append({"params": eigenvalue_head.parameters()})
    params.append({"params": weight_head.parameters()})

    optimizer = torch.optim.Adam(params, lr=args.lr)

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
    
    # ReduceLROnPlateau was found to cause training instability: the loss
    # diverged after ~25 epochs because the scheduler reduced the LR too
    # aggressively, preventing the optimizer from escaping sharp minima.
    # A constant LR (scheduler=None) gives a more stable learning curve
    # for this electronic-structure fine-tuning task.
    #
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,   # Halve LR on plateau
    #     patience=5,   # Wait 5 epochs before reducing
    #     min_lr=1e-7,  # Floor to avoid effectively freezing the model
    # )
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
    logging.info("Starting training!")

    num_steps = args.num_steps

    for epoch in range(start_epoch, args.max_epochs):
        print(f"Start epoch: {epoch} training...")
        finetune(
            model=model,
            eigenvalue_head=eigenvalue_head,
            weight_head=weight_head,
            optimizer=optimizer,
            dataloader=train_loader,
            lr_scheduler=lr_scheduler,
            clip_grad=args.gradient_clip_val,
            device=device,
            num_steps=num_steps,
            epoch=epoch,
            accumulation_steps=args.accumulation_steps,
        )

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
                #"lr_scheduler": lr_scheduler.state_dict()
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
    parser.add_argument("--base_model", default="orb_v3_conservative_inf_omat", type=str)
    parser.add_argument("--energy_loss_weight", default=0.0, type=float)
    parser.add_argument("--forces_loss_weight", default=0.0, type=float)
    parser.add_argument("--stress_loss_weight", default=0.0, type=float)
    parser.add_argument("--equigrad_loss_weight", default=0.0, type=float)
    parser.add_argument("--trainable_reference_energies", action="store_true")
    parser.add_argument("--custom_reference_energies", default=None, type=str)
    parser.add_argument("--accumulation_steps", default=4, type=int, help="Number of batches to accumulate gradients")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()