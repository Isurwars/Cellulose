"""Finetuning loop with custom loss weights, reference energy control, and custom spectral head."""

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
from orb_models.common.models.base import ModelMixin
from orb_models.common.training.metrics import ScalarMetricTracker
from orb_models.common.training.util import get_optim, init_device
from orb_models.common.utils import seed_everything
from orb_models.forcefield import pretrained
from orb_models.common.dataset.property_definitions import PropertyDefinition
import numpy as np

def eigenvalues_row_fn(row, dataset: str):
    import torch
    val = row.data.get("eigenvalues")
    if val is None:
        raise ValueError(f"No eigenvalues in row {row.id}")
    return torch.from_numpy(np.array(val, dtype=np.float64))

property_definitions.PROPERTIES["eigenvalues"] = PropertyDefinition(
    name="eigenvalues",
    dim=250,
    domain="real",
    row_to_property_fn=eigenvalues_row_fn,
)

# Define custom scatter_mean for pooling node features to graph features
def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
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
    spectral_head: nn.Module, # ADDED: Pass the new head to the training loop
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    lr_scheduler: _LRScheduler | None = None,
    num_steps: int | None = None,
    clip_grad: float | None = None,
    log_freq: float = 10,
    device: torch.device = torch.device("cpu"),
    epoch: int = 0,
):
    """Train for a fixed number of steps.

    Args:
        model: The model to optimize.
        spectral_head: The custom MLP for predicting eigenvalues.
        optimizer: The optimizer for the model.
        dataloader: A Pytorch Dataloader, which may be infinite if num_steps is passed.
        lr_scheduler: Optional, a Learning rate scheduler for modifying the learning rate.
        num_steps: The number of training steps to take.
        clip_grad: Optional, the gradient clipping threshold.
        log_freq: The logging frequency for step metrics.
        device: The device to use for training.
        epoch: The number of epochs the model has been fintuned.

    Returns
        A dictionary of metrics.
    """
    run: Any | None = wandb.run if WANDB_AVAILABLE else None

    metrics = ScalarMetricTracker()

    # Set the model to "train" mode.
    model.train()
    spectral_head.train() # ADDED: Set spectral head to train mode

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

        optimizer.zero_grad(set_to_none=True)

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

        with torch.autocast("cuda", enabled=False):
            # 1. Base Model Forward Pass (Energy & Forces)
            batch_outputs = model.loss(batch)
            base_loss = batch_outputs.loss
            
            # 2. Spectral Forward Pass
            # Extract internal node features from the base GNN
            gnn_out = model.model(batch)
            node_features = gnn_out["node_features"]
            
            # Pool node features into graph features to predict global eigenvalues
            batch_idx = torch.arange(len(batch.n_node), device=device).repeat_interleave(batch.n_node)
            graph_features = scatter_mean(node_features, batch_idx, dim=0)
            
            pred_eigenvalues = spectral_head(graph_features)
            
            if 'eigenvalues' in batch.system_targets:
                true_eigenvalues = batch.system_targets['eigenvalues']
                spectral_loss = nn.functional.mse_loss(pred_eigenvalues, true_eigenvalues)
            else:
                raise ValueError("Target 'eigenvalues' not found in batch. Ensure target_config includes it.")
            
            # 4. Total Loss
            # Weigh the spectral loss so it doesn't overpower the force learning
            total_loss = base_loss + (0.1 * spectral_loss)
            
            # Add spectral metrics to the log
            batch_outputs.log["loss/spectral"] = spectral_loss.detach()
            batch_outputs.log["loss/total"] = total_loss.detach()
            metrics.update(batch_outputs.log)
            
        if torch.isnan(total_loss):
            raise ValueError("nan loss encountered")
            
        total_loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            torch.nn.utils.clip_grad_norm_(spectral_head.parameters(), clip_grad) # ADDED: Clip spectral head

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metrics.update(step_metrics)

        if i != 0 and i % log_freq == 0:
            metrics_dict = metrics.get_metrics()
            if run is not None:
                step = (epoch * num_training_batches) + i
                if run.sweep_id is not None:
                    run.log(
                        {"loss": metrics_dict["loss/total"]}, # ADDED: Log total loss
                        commit=False,
                    )
                run.log(
                    {"step": step},
                    commit=False,
                )
                run.log(prefix_keys(metrics_dict, "finetune_step"), commit=True)

        # Finished a single full step!
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

    # ---------------------------------------------------------
    # ADDED: Define a standard PyTorch Sequential Head for Eigenvalues
    latent_dim = model.model.node_embed_size
    
    spectral_head = nn.Sequential(
        nn.Linear(latent_dim, 1024),
        nn.SiLU(),
        nn.Linear(1024, 250)  # Outputting the 250 CASTEP eigenvalues
    ).to(device)
    
    spectral_params = sum(p.numel() for p in spectral_head.parameters() if p.requires_grad)
    logging.info(f"Spectral Head has {spectral_params:,} trainable parameters.")
    # ---------------------------------------------------------

    model.to(device=device)
    total_steps = args.max_epochs * args.num_steps
    
    import re
    params = []
    # Split parameters based on the regex
    for name, param in model.named_parameters():
        if re.search(r"(.*bias|.*layer_norm.*|.*batch_norm.*)", name):
            params.append({"params": param, "weight_decay": 0.0})
        else:
            params.append({"params": param})
    params.append({"params": spectral_head.parameters()})

    optimizer = torch.optim.Adam(params, lr=args.lr)

    div_factor = 10  
    final_div_factor = 10  
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * div_factor,
        total_steps=total_steps,
        pct_start=0.05,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )
    
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

    # ADDED: Include 'eigenvalues' in the graph target config so the dataloader fetches it from DB
    graph_targets = ["energy", "stress"] if model.has_stress else ["energy"]
    graph_targets.append("eigenvalues")
    loader_args = dict(
        dataset_name=args.dataset,
        dataset_path=args.data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        target_config={"graph": graph_targets, "node": ["forces"]},
    )
    
    train_loader = build_train_loader(
        **loader_args,
        atoms_adapter=atoms_adapter,
        augmentation=True,
    )
    logging.info("Starting training!")

    num_steps = args.num_steps
    start_epoch = 0

    for epoch in range(start_epoch, args.max_epochs):
        print(f"Start epoch: {epoch} training...")
        finetune(
            model=model,
            spectral_head=spectral_head, # ADDED: Pass spectral head
            optimizer=optimizer,
            dataloader=train_loader,
            lr_scheduler=lr_scheduler,
            clip_grad=args.gradient_clip_val,
            device=device,
            num_steps=num_steps,
            epoch=epoch,
        )

        # Save every X epochs and final epoch
        if (epoch % args.save_every_x_epochs == 0) or (epoch == args.max_epochs - 1):
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            
            # ADDED: Save both the base model and the spectral head state
            checkpoint_data = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "spectral_head_state": spectral_head.state_dict(),
                "optimizer": optimizer.state_dict()
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
    parser.add_argument("--checkpoint_path", default=os.path.join(os.getcwd(), "ckpts"), type=str)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--base_model", default="orb_v3_conservative_inf_omat", type=str)
    parser.add_argument("--energy_loss_weight", default=None, type=float)
    parser.add_argument("--forces_loss_weight", default=None, type=float)
    parser.add_argument("--stress_loss_weight", default=None, type=float)
    parser.add_argument("--equigrad_loss_weight", default=None, type=float)
    parser.add_argument("--trainable_reference_energies", action="store_true")
    parser.add_argument("--custom_reference_energies", default=None, type=str)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()