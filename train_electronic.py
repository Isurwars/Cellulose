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

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any

import ase
import ase.db
import ase.data
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SubsetRandomSampler

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_BANDS: int = 250
"""Number of Kohn-Sham band energies / PDOS weight channels."""

LATENT_DIM: int = 256
"""Node embedding dimensionality of orb_v3 omol models."""


# ---------------------------------------------------------------------------
# Utilities  (future: utils.py)
# ---------------------------------------------------------------------------


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


def prefix_keys(dict_to_prefix: dict[str, Any], prefix: str, sep: str = "/") -> dict[str, Any]:
    """Add a prefix to dictionary keys with a separator."""
    return {f"{prefix}{sep}{k}": v for k, v in dict_to_prefix.items()}


def build_graph_index(n_node: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Build a per-node graph membership index from a per-graph node-count tensor.

    Args:
        n_node: 1-D tensor of shape [num_graphs] giving the number of nodes in
            each graph in the batch.
        device: Device to place the result on.

    Returns:
        1-D integer tensor of shape [total_nodes] mapping each node to its graph.
    """
    return torch.repeat_interleave(
        torch.arange(len(n_node), device=device), n_node
    )


def split_train_val(
    total_size: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split dataset indices into train and validation sets.

    Uses a seeded permutation so the split is reproducible across runs.

    Args:
        total_size: Total number of samples in the dataset.
        val_fraction: Fraction of data to use for validation (0.0–1.0).
        seed: Random seed for reproducibility.

    Returns:
        ``(train_indices, val_indices)`` — disjoint lists of integer indices.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(total_size).tolist()
    n_val = max(1, int(total_size * val_fraction))
    val_indices = sorted(indices[:n_val])
    train_indices = sorted(indices[n_val:])
    logging.info(
        f"Train/val split: {len(train_indices)} train, {len(val_indices)} val "
        f"({val_fraction:.0%} val, seed={seed})"
    )
    return train_indices, val_indices


# ---------------------------------------------------------------------------
# Property extraction  (future: data.py)
# ---------------------------------------------------------------------------

# These extraction functions must be defined at module top-level (not as
# lambdas or nested functions) so that Python's multiprocessing 'spawn' mode
# can pickle them for DataLoader worker processes.


def extract_eigenvalues(row: Any, dataset: str | None = None) -> torch.Tensor:
    """Return the 250 CASTEP Kohn-Sham eigenvalues for a structure as float32."""
    return torch.tensor(row.data["eigenvalues"], dtype=torch.float32)


def extract_weights(row: Any, dataset: str | None = None) -> torch.Tensor:
    """Return the per-atom 250-band PDOS weight vector for a structure as float32."""
    return torch.tensor(row.data["weights"], dtype=torch.float32)


# Register CASTEP electronic-structure properties in the Orb property
# registry once at import time.  Uses the top-level picklable functions above
# so DataLoader workers launched under 'spawn' can serialise them.
PROPERTIES["eigenvalues"] = PropertyDefinition(
    name="eigenvalues",
    dim=NUM_BANDS,
    domain="graph",
    row_to_property_fn=extract_eigenvalues,
)
PROPERTIES["weights"] = PropertyDefinition(
    name="weights",
    dim=NUM_BANDS,
    domain="node",
    row_to_property_fn=extract_weights,
)


# ---------------------------------------------------------------------------
# Reference energies  (future: data.py)
# ---------------------------------------------------------------------------


def load_custom_reference_energies(filepath: str) -> torch.Tensor:
    """Load custom reference energies from a file.

    Supports two formats:
      1. JSON: ``{"1": -13.6, "6": -1030.5, ...}`` or
               ``{"H": -13.6, "C": -1030.5, ...}``
      2. Text: One line per element — ``element_number energy`` or
               ``element_symbol energy``

    Args:
        filepath: Path to the reference energies file.

    Returns:
        Tensor of shape [118] with reference energies.
    """
    # Use ASE's built-in element-symbol → atomic-number mapping instead of a
    # hand-rolled dictionary.  This covers all 118 elements automatically.
    atomic_numbers: dict[str, int] = ase.data.atomic_numbers  # type: ignore[assignment]

    ref_energies = torch.zeros(118)

    def _set_ref(key: str, value: float) -> None:
        """Resolve *key* (atomic number or element symbol) and store *value*."""
        try:
            z = int(key)
            if 1 <= z <= 118:
                ref_energies[z] = value
            else:
                logging.warning(f"Atomic number out of range: {key}")
        except ValueError:
            z = atomic_numbers.get(key, 0)
            if z:
                ref_energies[z] = value
            else:
                logging.warning(f"Unknown element symbol or invalid atomic number: {key}")

    # Try to load as JSON first
    try:
        with open(filepath) as f:
            data = json.load(f)

        for key, value in data.items():
            _set_ref(key, float(value))
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

                _set_ref(parts[0], float(parts[1]))
        logging.info(f"Loaded reference energies from text file: {filepath}")

    return ref_energies


# ---------------------------------------------------------------------------
# Loss computation  (future: losses.py)
# ---------------------------------------------------------------------------


def compute_electronic_losses(
    pred_eigenvalues: torch.Tensor,
    true_eigenvalues: torch.Tensor,
    pred_weights: torch.Tensor,
    true_weights: torch.Tensor,
    device: torch.device,
    *,
    eig_loss_type: str = "mse",
    huber_delta: float = 1.0,
    peak_boost: float = 20.0,
    active_threshold: float = 0.1,
    magnitude_weight: float = 3.0,
    cramer_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute eigenvalue loss and peak-weighted PDOS weight loss.

    The weight loss is a combination of:
      * **Magnitude loss** — peak-weighted MSE that emphasises spectral peaks.
      * **Cramér (L2-Wasserstein) shape loss** — penalises CDF differences for
        atoms with non-negligible PDOS signal, encouraging correct spectral shape.

    Args:
        pred_eigenvalues: Predicted band energies, shape [num_graphs, 250].
        true_eigenvalues: Ground-truth band energies, shape [num_graphs, 250].
        pred_weights: Predicted PDOS weights, shape [N_nodes, 250].
        true_weights: Ground-truth PDOS weights, shape [N_nodes, 250].
        device: Device for zero-tensors when no active atoms exist.
        eig_loss_type: Loss function for eigenvalues — ``"mse"`` or ``"huber"``.
            Huber (smooth L1) is more robust to outlier eigenvalues.
        huber_delta: Transition point for Huber loss (only used when
            ``eig_loss_type="huber"``).
        peak_boost: Multiplier for peak-weighting in PDOS magnitude loss.
            Higher values penalise errors at spectral peaks more heavily.
        active_threshold: Minimum sum of true weights for an atom to be
            included in the Cramér shape loss.
        magnitude_weight: Blend coefficient for the magnitude component.
        cramer_weight: Blend coefficient for the Cramér shape component.

    Returns:
        (eig_loss, weight_loss) — both scalar tensors on *device*.
    """
    # Eigenvalue loss
    if eig_loss_type == "huber":
        eig_loss = torch.nn.functional.huber_loss(
            pred_eigenvalues, true_eigenvalues, delta=huber_delta
        )
    else:
        eig_loss = torch.nn.functional.mse_loss(pred_eigenvalues, true_eigenvalues)

    # Peak-weighted MSE for PDOS weights
    squared_errors = (pred_weights - true_weights) ** 2
    peak_multiplier = 1.0 + (true_weights * peak_boost)
    magnitude_loss = torch.mean(squared_errors * peak_multiplier)

    # Masked Cramér (L2 Wasserstein) shape loss
    true_sums = true_weights.sum(dim=-1, keepdim=True)
    active_mask = (true_sums > active_threshold).squeeze(-1)

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

    weight_loss = (magnitude_weight * magnitude_loss) + (cramer_weight * cramer_loss)

    return eig_loss, weight_loss


# ---------------------------------------------------------------------------
# Model heads  (future: heads.py)
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block: ``x + Dropout(SiLU(Linear(LayerNorm(x))))``.

    Adds a skip connection around a single hidden layer, improving gradient
    flow through deeper head networks at zero inference cost.
    """

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AttentionPool(nn.Module):
    """Learned attention-weighted graph pooling.

    Replaces uniform ``scatter_mean`` with a per-node importance gate so the
    model can learn which atoms contribute most to graph-level properties
    (e.g. H atoms in cellulose likely matter less for band structure than C/O).

    The gate is a sigmoid-activated linear projection that produces a scalar
    weight per node, which is multiplied element-wise with the node features
    before segment-mean pooling.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, graph_idx: torch.Tensor) -> torch.Tensor:
        """Attention-pool node features into graph features.

        Args:
            x: Node feature matrix, shape ``[N_nodes, dim]``.
            graph_idx: Per-node graph membership, shape ``[N_nodes]``.

        Returns:
            Graph features, shape ``[num_graphs, dim]``.
        """
        weights = self.gate(x)  # [N_nodes, 1]
        return scatter_mean(x * weights, graph_idx, dim=0)


def build_heads(
    latent_dim: int,
    device: torch.device,
    dropout: float = 0.0,
    couple_heads: bool = False,
) -> tuple[nn.Module, nn.Module, AttentionPool]:
    """Construct the eigenvalue head, PDOS-weight head, and attention pooling.

    Args:
        latent_dim: Dimensionality of the GNN node embeddings.
        device: Device to place the modules on.
        dropout: Dropout probability applied after each SiLU activation
            in the heads (0.0 = no dropout).
        couple_heads: Whether to feed eigenvalues as additional features
            to the weight head.

    Returns:
        ``(eigenvalue_head, weight_head, attention_pool)`` — all on *device*.
    """
    hidden_dim = 1024
    weight_in_dim = latent_dim + NUM_BANDS if couple_heads else latent_dim

    # Graph-level head: predicts 250 Kohn-Sham band energies from the
    # attention-pooled node embedding.  A residual block in the hidden layer
    # improves gradient flow without adding inference cost.
    eigenvalue_head = nn.Sequential(
        nn.LayerNorm(latent_dim),
        nn.Linear(latent_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        ResidualBlock(hidden_dim, dropout),
        nn.Linear(hidden_dim, NUM_BANDS),
    ).to(device)

    # Node-level head: predicts 250 PDOS weights per atom.  Sigmoid ensures
    # outputs in [0, 1].  A residual block gives capacity for the per-atom
    # spectral decomposition.
    weight_head = nn.Sequential(
        nn.LayerNorm(weight_in_dim),
        nn.Linear(weight_in_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        ResidualBlock(hidden_dim, dropout),
        nn.Linear(hidden_dim, NUM_BANDS),
        nn.Sigmoid(),
    ).to(device)

    # Initialize the bias of the final linear layer in the weight head
    # to a negative value (-4.5).  Sigmoid(-4.5) ≈ 0.011, matching the
    # typical scale of target weights and avoiding initial loss explosions.
    nn.init.constant_(weight_head[-2].bias, -4.5)

    # Learned attention pooling for graph-level predictions
    attention_pool = AttentionPool(latent_dim).to(device)

    return eigenvalue_head, weight_head, attention_pool


# ---------------------------------------------------------------------------
# Wandb integration
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Data loading  (future: data.py)
# ---------------------------------------------------------------------------


def build_train_loader(
    dataset_name: str,
    dataset_path: str,
    num_workers: int,
    batch_size: int,
    atoms_adapter: AbstractAtomsAdapter,
    augmentation: bool | None = True,
    target_config: dict[str, list[str]] | None = None,
    train_indices: list[int] | None = None,
    **kwargs: Any,
) -> DataLoader:
    """Build the training DataLoader from an ASE SQLite database.

    Args:
        dataset_name: The name of the dataset.
        dataset_path: Dataset path.
        num_workers: The number of workers for each dataset.
        batch_size: The batch_size config for each dataset.
        atoms_adapter: The atoms adapter for converting ase.Atoms to
            model-specific AbstractAtomBatch instances.
        augmentation: If rotation augmentation is used.
        target_config: The target config.

    Returns:
        The train DataLoader.
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

    if train_indices is not None:
        sampler = SubsetRandomSampler(train_indices)
    else:
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


def cache_eval_frames(
    data_path: str,
    atoms_adapter: AbstractAtomsAdapter,
    val_indices: set[int] | None = None,
) -> list[tuple[Any, dict[str, Any]]]:
    """Preprocess and cache all database frames for evaluation.

    Each entry is a ``(single_graph, ground_truth_dict)`` pair suitable for
    direct consumption by :func:`evaluate_model`.

    Args:
        data_path: Path to the ASE SQLite database.
        atoms_adapter: Adapter for converting ASE Atoms to model graphs.

    Returns:
        List of ``(graph, gt)`` tuples.
    """
    logging.info("Preprocessing and caching database frames for evaluation...")
    db = ase.db.connect(data_path)
    eval_frames: list[tuple[Any, dict[str, Any]]] = []

    for idx, row in enumerate(db.select()):
        if val_indices is not None and idx not in val_indices:
            continue
        test_atoms = row.toatoms()
        single_graph = atoms_adapter.from_ase_atoms(test_atoms)
        gt: dict[str, Any] = {
            "energy": row.energy if hasattr(row, "energy") else None,
            "forces": row.forces if hasattr(row, "forces") else None,
            "eigenvalues": row.data.get("eigenvalues") if "eigenvalues" in row.data else None,
            "weights": row.data.get("weights") if "weights" in row.data else None,
            "cell": test_atoms.get_cell().array,
        }
        eval_frames.append((single_graph, gt))

    logging.info(f"Cached {len(eval_frames)} frames for evaluation.")
    return eval_frames


# ---------------------------------------------------------------------------
# Optimizer & scheduler construction  (future: optim.py)
# ---------------------------------------------------------------------------


def build_loss_weights(args: argparse.Namespace) -> dict[str, float]:
    """Convert CLI loss-weight arguments into the dict expected by Orb models.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Dictionary mapping loss names to their weights (only non-None entries).
    """
    is_conservative = "conservative" in args.base_model
    loss_weights: dict[str, float] = {}

    if args.energy_loss_weight is not None:
        loss_weights["energy"] = args.energy_loss_weight

    if args.forces_loss_weight is not None:
        key = "grad_forces" if is_conservative else "forces"
        loss_weights[key] = args.forces_loss_weight

    if args.stress_loss_weight is not None:
        key = "grad_stress" if is_conservative else "stress"
        loss_weights[key] = args.stress_loss_weight

    if args.equigrad_loss_weight is not None and args.equigrad_loss_weight > 0.0:
        if not is_conservative:
            raise ValueError("Equigrad loss is only available for conservative models.")
        loss_weights["rotational_grad"] = args.equigrad_loss_weight

    if loss_weights:
        logging.info("=" * 60)
        logging.info("Custom loss weights specified:")
        for key, val in loss_weights.items():
            logging.info(f"  {key}: {val}")
        logging.info("=" * 60)

    return loss_weights


def build_optimizer(
    model: ModelMixin,
    eigenvalue_head: nn.Module,
    weight_head: nn.Module,
    attention_pool: AttentionPool,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    """Build the Adam optimizer with per-group learning rates.

    Backbone parameters get special weight-decay treatment (bias and
    normalisation layers are exempt).  If the backbone will be unfrozen later,
    its initial LR is set to 0.0 so it contributes no gradient signal until
    the target epoch.

    Args:
        model: The pretrained Orb GNN backbone.
        eigenvalue_head: Eigenvalue prediction head.
        weight_head: PDOS weight prediction head.
        attention_pool: Learned attention pooling module.
        args: Parsed command-line arguments.

    Returns:
        Configured Adam optimizer.
    """
    include_backbone = (not args.freeze_backbone) or (args.unfreeze_epoch is not None)

    params: list[dict[str, Any]] = []

    if include_backbone:
        init_backbone_lr = (
            0.0 if (args.freeze_backbone and args.unfreeze_epoch is not None) else args.backbone_lr
        )
        logging.info(f"Including GNN backbone in optimizer with initial LR: {init_backbone_lr}")

        # Exclude bias, LayerNorm, and BatchNorm parameters from weight decay.
        # Regularising these normalisation parameters can destabilise training.
        for name, param in model.named_parameters():
            if re.search(r"(.*bias|.*layer_norm.*|.*batch_norm.*)", name):
                params.append({"params": param, "weight_decay": 0.0, "lr": init_backbone_lr})
            else:
                params.append({"params": param, "lr": init_backbone_lr})
    else:
        logging.info("Excluding GNN backbone parameters from optimizer (permanently frozen).")

    # The custom heads and attention pool are trained from scratch and do not
    # need special weight-decay treatment; they use the global default.
    params.append({"params": eigenvalue_head.parameters(), "lr": args.lr})
    params.append({"params": weight_head.parameters(), "lr": args.lr})
    params.append({"params": attention_pool.parameters(), "lr": args.lr})

    return torch.optim.Adam(params)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    total_epochs: int,
    unfreeze_offset: int | None = None,
) -> tuple[torch.optim.lr_scheduler._LRScheduler | None, int | None]:
    """Build the learning-rate scheduler.

    This function is called both at init and again when the backbone unfreezes,
    avoiding duplication of the scheduler construction logic.

    Args:
        optimizer: The optimizer whose LR groups are managed.
        args: Parsed command-line arguments (uses ``scheduler``, ``min_lr``).
        total_epochs: Number of epochs the scheduler should span.
        unfreeze_offset: If set, the epoch at which the backbone unfreezes
            (relative to the start of this scheduler's lifetime).

    Returns:
        ``(lr_scheduler, cosine_start_epoch)`` — the scheduler (or ``None``)
        and the absolute epoch at which the cosine phase begins (or ``None``).
    """
    cosine_start_epoch: int | None = None

    if args.scheduler == "cosine":
        logging.info("Initializing CosineAnnealingLR scheduler.")
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=args.min_lr
        )
        cosine_start_epoch = 0

    elif args.scheduler == "flat_cosine":
        logging.info("Initializing Flat-Cosine (SequentialLR) scheduler.")
        T_flat = total_epochs // 2

        if unfreeze_offset is not None:
            cosine_start_epoch = unfreeze_offset + (total_epochs - unfreeze_offset) // 2
        else:
            cosine_start_epoch = total_epochs // 2

        if T_flat > 0:
            scheduler1 = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=T_flat
            )
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - T_flat, eta_min=args.min_lr
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[scheduler1, scheduler2], milestones=[T_flat]
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=args.min_lr
            )

    elif args.scheduler == "plateau":
        logging.info("Initializing ReduceLROnPlateau scheduler (stepped per epoch).")
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=args.min_lr
        )
        cosine_start_epoch = None

    else:
        logging.info("No learning rate scheduler specified (constant learning rate).")
        lr_scheduler = None
        cosine_start_epoch = None

    return lr_scheduler, cosine_start_epoch


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    epoch: int,
    model: ModelMixin,
    eigenvalue_head: nn.Module,
    weight_head: nn.Module,
    attention_pool: AttentionPool,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any | None,
    config: dict[str, Any] | None = None,
    filename: str | None = None,
) -> None:
    """Save a training checkpoint to disk.

    Args:
        path: Directory in which to save the checkpoint file.
        epoch: Current epoch number (embedded in the filename).
        model: The GNN backbone.
        eigenvalue_head: Eigenvalue prediction head.
        weight_head: PDOS weight prediction head.
        attention_pool: Learned attention pooling module.
        optimizer: Optimizer (state is saved for resumption).
        lr_scheduler: LR scheduler (state saved if not ``None``).
        config: Training configuration dict (saved for reproducibility).
        filename: Override the default checkpoint filename.
    """
    os.makedirs(path, exist_ok=True)
    checkpoint_data = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "eigenvalue_head_state": eigenvalue_head.state_dict(),
        "weight_head_state": weight_head.state_dict(),
        "attention_pool_state": attention_pool.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "config": config,
    }
    filepath = os.path.join(path, filename or f"checkpoint_epoch{epoch}.ckpt")
    torch.save(checkpoint_data, filepath)
    logging.info(f"Checkpoint saved to {path}")


def resume_checkpoint(
    checkpoint_path: str,
    model: ModelMixin,
    eigenvalue_head: nn.Module,
    weight_head: nn.Module,
    attention_pool: AttentionPool,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any | None,
    device: torch.device,
    unfreeze_epoch: int | None,
) -> int:
    """Restore model, heads, attention pool, optimizer, and scheduler from a checkpoint.

    Args:
        checkpoint_path: Path to the ``.ckpt`` file.
        model: The GNN backbone (modified in-place).
        eigenvalue_head: Eigenvalue head (modified in-place).
        weight_head: Weight head (modified in-place).
        attention_pool: Attention pooling module (modified in-place).
        optimizer: Optimizer (state loaded in-place).
        lr_scheduler: LR scheduler (state loaded if present in checkpoint).
        device: Device to map tensors onto.
        unfreeze_epoch: If set, the epoch at which the backbone should be
            unfrozen.  Used to enable gradients when resuming past that point.

    Returns:
        The epoch to resume from (i.e. the next epoch after the saved one).
    """
    logging.info(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    eigenvalue_head.load_state_dict(checkpoint["eigenvalue_head_state"])
    weight_head.load_state_dict(checkpoint["weight_head_state"])

    # Attention pool state may be absent in older checkpoints
    if "attention_pool_state" in checkpoint:
        attention_pool.load_state_dict(checkpoint["attention_pool_state"])
    else:
        logging.warning("No attention_pool_state in checkpoint (old format); using fresh init.")

    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except ValueError as e:
        logging.warning(f"Could not load optimizer state (might be incompatible): {e}")

    start_epoch: int = checkpoint.get("epoch", -1) + 1
    logging.info(f"Resumed at epoch: {start_epoch}")

    # If resuming at or after unfreeze epoch, ensure GNN parameters have gradients enabled
    if unfreeze_epoch is not None and start_epoch >= unfreeze_epoch:
        logging.info("Resuming after unfreeze epoch. Ensuring GNN backbone parameters are unfrozen.")
        for param in model.parameters():
            param.requires_grad = True

    # Restore scheduler state if available
    if lr_scheduler is not None and "lr_scheduler" in checkpoint and checkpoint["lr_scheduler"] is not None:
        try:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            logging.info("Loaded lr_scheduler state from checkpoint.")
        except Exception as e:
            logging.warning(f"Could not load lr_scheduler state: {e}")

    return start_epoch


# ---------------------------------------------------------------------------
# Training loop  (future: trainer.py)
# ---------------------------------------------------------------------------


def finetune(
    model: ModelMixin,
    eigenvalue_head: nn.Module,
    weight_head: nn.Module,
    attention_pool: AttentionPool,
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
    *,
    eig_loss_type: str = "mse",
    huber_delta: float = 1.0,
    pdos_peak_boost: float = 20.0,
    pdos_active_threshold: float = 0.1,
    pdos_magnitude_weight: float = 3.0,
    pdos_cramer_weight: float = 0.5,
    couple_heads: bool = False,
    detach_coupling: bool = False,
) -> dict[str, float]:
    """Run one epoch of electronic-structure finetuning.

    Performs gradient-accumulation training over the dataloader.  The physics
    loss (energy/forces) is deliberately ignored unless explicitly weighted;
    only the eigenvalue loss and the peak-weighted PDOS weight loss are
    back-propagated by default.

    Args:
        model: The pretrained Orb GNN backbone (run in train mode).
        eigenvalue_head: MLP that maps pooled node features to 250 band
            energies (graph-level target).
        weight_head: MLP that maps per-node features to 250 PDOS weights
            (node-level target, Sigmoid output).
        attention_pool: Learned attention pooling module.
        optimizer: Shared optimizer covering backbone + heads + attention pool.
        dataloader: PyTorch DataLoader; may be finite or stepped via num_steps.
        num_steps: Hard cap on batches consumed per epoch.
        clip_grad: If set, gradient norms for all parameter groups are clipped
            to this value before each optimizer step.
        log_freq: Log aggregated metrics every this many steps.
        device: Device on which tensors live.
        epoch: Current epoch index (used to compute the global step for wandb).
        accumulation_steps: Number of forward passes whose gradients are summed
            before a single optimizer step.
        freeze_backbone: Whether the GNN backbone is frozen this epoch.
        eigenvalue_loss_weight: Scaling factor for eigenvalue loss.
        weight_loss_weight: Scaling factor for PDOS weight loss.
        energy_loss_weight: Scaling factor for energy MSE loss.
        forces_loss_weight: Scaling factor for forces MSE loss.
        is_conservative_model: Whether the model uses conservative force computation.
        eig_loss_type: Loss type for eigenvalues ('mse' or 'huber').
        huber_delta: Transition point for Huber loss.
        pdos_peak_boost: PDOS peak boost factor.
        pdos_active_threshold: Minimum active sum for Cramér shape loss.
        pdos_magnitude_weight: Weight of magnitude loss.
        pdos_cramer_weight: Weight of Cramér shape loss.

    Returns:
        A dictionary of scalar metrics averaged over the epoch.
    """
    run_handle: Any | None = wandb.run if WANDB_AVAILABLE else None
    metrics = ScalarMetricTracker()

    # Set the model to "train" mode (or "eval" if frozen).
    if freeze_backbone:
        model.eval()
    else:
        model.train()
    eigenvalue_head.train()
    weight_head.train()
    attention_pool.train()

    # Resolve total number of training batches
    num_training_batches: int | float
    if num_steps is not None:
        num_training_batches = num_steps
    else:
        try:
            num_training_batches = len(dataloader)
        except TypeError:
            raise ValueError("Dataloader has no length, you must specify num_steps.")

    batch_generator_tqdm = tqdm.tqdm(iter(dataloader), total=num_training_batches)

    for i, batch in enumerate(batch_generator_tqdm):
        if num_steps and i == num_steps:
            break

        # Reset metrics at log boundaries for per-window reporting
        if i % log_freq == 0:
            metrics.reset()

        batch = batch.to(device)

        step_metrics: dict[str, float] = {
            "batch_size": float(len(batch.n_node)),
            "batch_num_edges": float(batch.n_edge.sum()),
            "batch_num_nodes": float(batch.n_node.sum()),
        }

        # Autocast is disabled (enabled=False) rather than using bfloat16.
        # The Orb model has float32 LayerNorm weights; enabling float16 autocast
        # feeds fp16 activations into those layers and triggers a kernel dispatch
        # mismatch.  bfloat16 would be safe, but keeping full float32 avoids any
        # precision loss when fine-tuning the sensitive electronic-structure heads.
        with torch.autocast("cuda", enabled=False):
            # --- GNN Backbone forward pass ---
            gnn_out = model.model(batch)
            node_features = gnn_out["node_features"]  # [N_nodes, latent_dim]

            graph_idx = build_graph_index(batch.n_node, node_features.device)
            # Use AttentionPool for pooling node features to graph features
            graph_features = attention_pool(node_features, graph_idx)

            # --- Electronic structure predictions & loss ---
            pred_eigenvalues = eigenvalue_head(graph_features)  # [num_graphs, 250]

            if couple_heads:
                node_eigenvalues = pred_eigenvalues[graph_idx]  # [N_nodes, 250]
                if detach_coupling:
                    node_eigenvalues = node_eigenvalues.detach()
                weight_head_input = torch.cat([node_features, node_eigenvalues], dim=-1)
            else:
                weight_head_input = node_features

            pred_weights = weight_head(weight_head_input)          # [N_nodes, 250]

            true_eigenvalues = batch.system_targets["eigenvalues"]
            true_weights = batch.node_targets["weights"]

            eig_loss, weight_loss = compute_electronic_losses(
                pred_eigenvalues, true_eigenvalues,
                pred_weights, true_weights,
                device,
                eig_loss_type=eig_loss_type,
                huber_delta=huber_delta,
                peak_boost=pdos_peak_boost,
                active_threshold=pdos_active_threshold,
                magnitude_weight=pdos_magnitude_weight,
                cramer_weight=pdos_cramer_weight,
            )

            # --- Physics predictions & loss (only if GNN is unfrozen) ---
            is_physics_active = (not freeze_backbone) and (
                energy_loss_weight > 0.0 or forces_loss_weight > 0.0
            )

            if is_physics_active:
                if is_conservative_model:
                    with torch.set_grad_enabled(True):
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
                forces_loss = torch.nn.functional.mse_loss(pred_forces, true_forces)
            else:
                energy_loss = torch.tensor(0.0, device=device)
                forces_loss = torch.tensor(0.0, device=device)

            # --- Total loss ---
            total_loss = (
                (energy_loss_weight * energy_loss)
                + (forces_loss_weight * forces_loss)
                + (eigenvalue_loss_weight * eig_loss)
                + (weight_loss_weight * weight_loss)
            )
            scaled_loss = total_loss / accumulation_steps

            # --- Logging ---
            batch_outputs: dict[str, torch.Tensor] = {
                "loss/eigenvalues": eig_loss.detach(),
                "loss/weights": weight_loss.detach(),
                "loss/total": total_loss.detach(),
            }
            if is_physics_active:
                batch_outputs["loss/energy"] = energy_loss.detach()
                batch_outputs["loss/forces"] = forces_loss.detach()
            metrics.update(batch_outputs)

        if torch.isnan(scaled_loss):
            logging.warning(f"NaN scaled_loss at step {i}. Skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaled_loss.backward()

        # Gradient accumulation: only update weights every `accumulation_steps`
        # batches (or on the final batch).  This simulates a larger effective
        # batch size without the memory overhead of a physically larger batch.
        if (i + 1) % accumulation_steps == 0 or (i + 1) == num_training_batches:
            if clip_grad is not None:
                # Clip gradients for all parameter groups independently
                # to prevent instability when the heads first start training.
                # clip_grad_norm_ returns the pre-clip total norm for diagnostics.
                backbone_gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                eig_head_gnorm = torch.nn.utils.clip_grad_norm_(eigenvalue_head.parameters(), clip_grad)
                w_head_gnorm = torch.nn.utils.clip_grad_norm_(weight_head.parameters(), clip_grad)
                attn_pool_gnorm = torch.nn.utils.clip_grad_norm_(attention_pool.parameters(), clip_grad)
                metrics.update({
                    "grad_norm/backbone": backbone_gnorm.detach(),
                    "grad_norm/eigenvalue_head": eig_head_gnorm.detach(),
                    "grad_norm/weight_head": w_head_gnorm.detach(),
                    "grad_norm/attention_pool": attn_pool_gnorm.detach(),
                })

            optimizer.step()

            # Release gradient tensors immediately to free memory for the next
            # accumulation window (set_to_none is more efficient than zeroing).
            optimizer.zero_grad(set_to_none=True)

        metrics.update(step_metrics)

        if i != 0 and i % log_freq == 0:
            metrics_dict = metrics.get_metrics()
            if run_handle is not None:
                step = (epoch * num_training_batches) + i
                if run_handle.sweep_id is not None:
                    run_handle.log({"loss": metrics_dict["loss/total"]}, commit=False)
                run_handle.log({"step": step}, commit=False)
                run_handle.log(prefix_keys(metrics_dict, "finetune_step"), commit=True)

    return metrics.get_metrics()


# ---------------------------------------------------------------------------
# Evaluation  (future: evaluation.py)
# ---------------------------------------------------------------------------


def evaluate_model(
    model: ModelMixin,
    eigenvalue_head: nn.Module,
    weight_head: nn.Module,
    attention_pool: AttentionPool,
    atoms_adapter: AbstractAtomsAdapter,
    eval_frames: list[tuple[Any, dict[str, Any]]],
    device: torch.device,
    is_conservative_model: bool = False,
    plot_path: str | None = None,
    fast_eval: bool = False,
    couple_heads: bool = False,
) -> dict[str, float]:
    """Evaluate current model checkpoint on cached validation frames.

    Args:
        model: The GNN backbone (set to eval mode internally).
        eigenvalue_head: Eigenvalue prediction head.
        weight_head: PDOS weight prediction head.
        attention_pool: Attention pooling module.
        atoms_adapter: Adapter for batching single graphs.
        eval_frames: Pre-cached ``(graph, ground_truth)`` pairs from
            :func:`cache_eval_frames`.
        device: Device for computation.
        is_conservative_model: Whether the model uses conservative force
            computation (forces derived via autograd).
        plot_path: If set, save a 3-panel parity plot to this path.
        fast_eval: If ``True``, evaluate only the first 100 frames and skip
            force evaluation for speed.

    Returns:
        Dictionary with keys ``forces_rmse``, ``eigs_rmse``, ``weights_rmse``.
    """
    model.eval()
    eigenvalue_head.eval()
    weight_head.eval()
    attention_pool.eval()

    results: dict[str, list[Any]] = {
        "forces_true": [], "forces_pred": [],
        "eigs_true": [], "eigs_pred": [],
        "weights_true": [], "weights_pred": [],
    }

    frames_to_eval = eval_frames[:100] if fast_eval else eval_frames

    for single_graph, gt in frames_to_eval:
        inputs = atoms_adapter.batch([single_graph]).to(device)
        inputs.system_features = {
            "total_charge": torch.tensor([0.0], dtype=torch.float32, device=device),
            "spin_multiplicity": torch.tensor([1.0], dtype=torch.float32, device=device),
            "cell": torch.tensor(gt["cell"], dtype=torch.float32, device=device).unsqueeze(0),
        }

        # 1. Physics Evaluation (Forces)
        if not fast_eval:
            if is_conservative_model:
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

            # Use AttentionPool for consistency with training
            graph_idx = build_graph_index(inputs.n_node, node_feats.device)
            graph_feats = attention_pool(node_feats, graph_idx)

            pred_eigs_tensor = eigenvalue_head(graph_feats)
            if couple_heads:
                node_eigenvalues = pred_eigs_tensor[graph_idx]  # [N_nodes, 250]
                weight_head_input = torch.cat([node_feats, node_eigenvalues], dim=-1)
            else:
                weight_head_input = node_feats

            pred_eigs = pred_eigs_tensor.cpu().numpy().flatten()
            pred_weights = weight_head(weight_head_input).cpu().numpy().flatten()

            results["eigs_true"].append(gt["eigenvalues"])
            results["eigs_pred"].append(pred_eigs)
            results["weights_true"].append(np.array(gt["weights"]).flatten())
            results["weights_pred"].append(pred_weights)

    # Calculate RMSE
    if not fast_eval:
        f_true = np.concatenate(results["forces_true"]).flatten()
        f_pred = np.concatenate(results["forces_pred"]).flatten()
        forces_rmse = float(np.sqrt(np.mean((f_true - f_pred) ** 2)))
    else:
        forces_rmse = float("nan")

    eig_true = np.array(results["eigs_true"]).flatten()
    eig_pred = np.array(results["eigs_pred"]).flatten()
    eigs_rmse = float(np.sqrt(np.mean((eig_true - eig_pred) ** 2)))

    w_true = np.concatenate(results["weights_true"])
    w_pred = np.concatenate(results["weights_pred"])
    weights_rmse = float(np.sqrt(np.mean((w_true - w_pred) ** 2)))

    if plot_path is not None:
        _save_parity_plots(
            eig_true, eig_pred, eigs_rmse,
            w_true, w_pred, weights_rmse,
            f_true if not fast_eval else None,
            f_pred if not fast_eval else None,
            forces_rmse,
            plot_path,
        )

    return {
        "forces_rmse": forces_rmse,
        "eigs_rmse": eigs_rmse,
        "weights_rmse": weights_rmse,
    }


def _save_parity_plots(
    eig_true: np.ndarray,
    eig_pred: np.ndarray,
    eigs_rmse: float,
    w_true: np.ndarray,
    w_pred: np.ndarray,
    weights_rmse: float,
    f_true: np.ndarray | None,
    f_pred: np.ndarray | None,
    forces_rmse: float,
    plot_path: str,
) -> None:
    """Save a 3-panel parity plot (eigenvalues, weights, forces) to disk."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Eigenvalues Parity Plot
    ax[0].scatter(eig_true, eig_pred, alpha=0.1, s=0.5)
    ax[0].plot([eig_true.min(), eig_true.max()], [eig_true.min(), eig_true.max()], "r--")
    ax[0].set_title(f"Eigenvalues (RMSE: {eigs_rmse:.3f} eV)")
    ax[0].set_xlabel("DFT Eigenvalues (eV)")
    ax[0].set_ylabel("ML Predicted (eV)")

    # 2. PDOS Weights Parity Plot
    ax[1].scatter(w_true, w_pred, alpha=0.1, s=0.5)
    ax[1].plot([w_true.min(), w_true.max()], [w_true.min(), w_true.max()], "r--")
    ax[1].set_title(f"PDOS Weights (RMSE: {weights_rmse:.3f})")
    ax[1].set_xlabel("DFT PDOS Weights")
    ax[1].set_ylabel("ML Predicted")

    # 3. Forces Parity Plot
    if f_true is not None and f_pred is not None:
        ax[2].scatter(f_true, f_pred, alpha=0.3, s=1)
        ax[2].plot([f_true.min(), f_true.max()], [f_true.min(), f_true.max()], "r--")
        ax[2].set_title(f"Forces (RMSE: {forces_rmse:.3f} eV/Å)")
    else:
        ax[2].set_title("Forces (skipped — fast eval)")
    ax[2].set_xlabel("DFT Forces (eV/Å)")
    ax[2].set_ylabel("ML Predicted (eV/Å)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved parity plot to {plot_path}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """Top-level training orchestrator.

    Sets up the model, heads, optimizer, scheduler, data, and runs the
    epoch loop with evaluation and checkpointing.

    Args:
        args: Parsed command-line arguments from :func:`main`.
    """
    device = init_device(device_id=args.device_id)
    seed_everything(args.random_seed)

    precision = "float32-high"
    is_conservative_model = "conservative" in args.base_model

    # --- Model ---
    loss_weights = build_loss_weights(args)

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
        custom_refs = load_custom_reference_energies(args.custom_reference_energies).to(device)

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

    # --- Heads ---
    eigenvalue_head, weight_head, attention_pool = build_heads(
        LATENT_DIM, device, dropout=args.dropout, couple_heads=args.couple_heads
    )
    model.to(device=device)

    # --- Freeze backbone if requested ---
    if args.freeze_backbone:
        logging.info("Initially freezing GNN backbone parameters.")
        for param in model.parameters():
            param.requires_grad = False

    # --- Optimizer ---
    optimizer = build_optimizer(model, eigenvalue_head, weight_head, attention_pool, args)

    # --- Scheduler ---
    lr_scheduler, cosine_start_epoch = build_scheduler(
        optimizer, args,
        total_epochs=args.max_epochs,
        unfreeze_offset=args.unfreeze_epoch,
    )

    # --- Resume from checkpoint ---
    start_epoch = 0
    if args.resume_from_checkpoint:
        start_epoch = resume_checkpoint(
            args.resume_from_checkpoint,
            model, eigenvalue_head, weight_head, attention_pool,
            optimizer, lr_scheduler, device,
            unfreeze_epoch=args.unfreeze_epoch,
        )

    # --- Wandb ---
    wandb_run = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb flag is set but wandb is not installed.")
        logging.info("Instantiating WandbLogger.")
        wandb_run = init_wandb_from_config(
            dataset=args.dataset, job_type="finetuning", entity=args.wandb_entity
        )
        wandb.define_metric("step")
        wandb.define_metric("finetune_step/*", step_metric="step")

    # --- Data (train/val split) ---
    graph_targets = ["energy", "stress"] if model.has_stress else ["energy"]
    graph_targets.append("eigenvalues")

    # Compute train/val split before building the DataLoader
    train_indices: list[int] | None = None
    val_indices: set[int] | None = None
    if args.val_fraction > 0.0:
        db = ase.db.connect(args.data_path)
        total_size = db.count()
        train_idx_list, val_idx_list = split_train_val(
            total_size, args.val_fraction, args.random_seed
        )
        train_indices = train_idx_list
        val_indices = set(val_idx_list)

    train_loader = build_train_loader(
        dataset_name=args.dataset,
        dataset_path=args.data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        target_config={"graph": graph_targets, "node": ["forces", "weights"]},
        atoms_adapter=atoms_adapter,
        augmentation=True,
        train_indices=train_indices,
    )

    eval_frames: list[tuple[Any, dict[str, Any]]] = []
    if args.eval_every_x_epochs > 0:
        eval_frames = cache_eval_frames(args.data_path, atoms_adapter, val_indices=val_indices)

    # --- Training loop ---
    logging.info("Starting training!")
    num_steps = args.num_steps if args.num_steps > 0 else None
    best_composite_metric = float("inf")
    config_dict = vars(args)

    for epoch in range(start_epoch, args.max_epochs):
        # Dynamic unfreezing check
        is_currently_frozen = args.freeze_backbone and (
            args.unfreeze_epoch is None or epoch < args.unfreeze_epoch
        )

        # Ensure requires_grad matches unfreezing state
        for param in model.parameters():
            param.requires_grad = not is_currently_frozen

        # Trigger unfreezing at target epoch
        if args.unfreeze_epoch is not None and epoch == args.unfreeze_epoch:
            logging.info(f"--- Unfreezing GNN backbone at epoch {epoch} ---")
            head_params_set = set(
                list(eigenvalue_head.parameters()) + list(weight_head.parameters()) + list(attention_pool.parameters())
            )
            for idx, group in enumerate(optimizer.param_groups):
                is_backbone = any(p not in head_params_set for p in group["params"])
                if is_backbone:
                    group["lr"] = args.backbone_lr
                    logging.info(f"  GNN Backbone group {idx} LR set to: {args.backbone_lr}")
                else:
                    group["lr"] = args.lr
                    logging.info(f"  Head group {idx} LR reset to: {args.lr}")
                # Remove cached initial_lr so the new scheduler picks up the updated LR
                group.pop("initial_lr", None)

            # Re-initialize the scheduler to start decay from this epoch
            remaining_epochs = args.max_epochs - epoch
            lr_scheduler, cosine_start_epoch = build_scheduler(
                optimizer, args,
                total_epochs=remaining_epochs,
            )
            # Offset cosine_start_epoch to absolute epoch numbering
            if cosine_start_epoch is not None:
                cosine_start_epoch += epoch

        # Apply weight noise perturbation to break false minima.
        # Only inject noise before the cosine decay phase.
        is_cosine_phase = cosine_start_epoch is not None and epoch >= cosine_start_epoch
        if (
            args.weight_head_noise_std > 0
            and epoch > 0
            and epoch % args.weight_head_noise_interval == 0
            and not is_cosine_phase
        ):
            logging.info(
                f"Injecting random noise filter (std={args.weight_head_noise_std}) "
                f"into weight_head parameters to break false minimum."
            )
            with torch.no_grad():
                for param in weight_head.parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * args.weight_head_noise_std
                        param.add_(noise)

        current_lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
        logging.info(f"Epoch {epoch} — LRs: {current_lrs}")

        epoch_metrics = finetune(
            model=model,
            eigenvalue_head=eigenvalue_head,
            weight_head=weight_head,
            attention_pool=attention_pool,
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
            eig_loss_type=args.eig_loss_type,
            huber_delta=args.huber_delta,
            pdos_peak_boost=args.pdos_peak_boost,
            pdos_active_threshold=args.pdos_active_threshold,
            pdos_magnitude_weight=args.pdos_magnitude_weight,
            pdos_cramer_weight=args.pdos_cramer_weight,
            couple_heads=args.couple_heads,
            detach_coupling=args.detach_coupling,
        )

        # Step the learning rate scheduler once per epoch
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(epoch_metrics["loss/total"])
                logging.info(f"Stepped ReduceLROnPlateau with loss: {epoch_metrics['loss/total']:.4f}")
            else:
                lr_scheduler.step()
                logging.info("Stepped LR scheduler.")

        # Determine if we should save checkpoint at this epoch
        is_ckpt_epoch = (epoch % args.save_every_x_epochs == 0) or (epoch == args.max_epochs - 1)

        # Periodical evaluation and validation checks
        if args.eval_every_x_epochs > 0 and (
            epoch % args.eval_every_x_epochs == 0 or epoch == args.max_epochs - 1
        ):
            plot_path = None
            if is_ckpt_epoch:
                plot_path = os.path.join(args.checkpoint_path, f"cellulose__epoch{epoch}.png")

            eval_metrics = evaluate_model(
                model=model,
                eigenvalue_head=eigenvalue_head,
                weight_head=weight_head,
                attention_pool=attention_pool,
                atoms_adapter=atoms_adapter,
                eval_frames=eval_frames,
                device=device,
                is_conservative_model=is_conservative_model,
                plot_path=plot_path,
                fast_eval=not is_ckpt_epoch,
                couple_heads=args.couple_heads,
            )
            logging.info("=" * 60)
            logging.info(f"Epoch {epoch} Evaluation Metrics:")
            logging.info(f"  Eigenvalues RMSE: {eval_metrics['eigs_rmse']:.4f} eV")
            logging.info(f"  Weights RMSE:     {eval_metrics['weights_rmse']:.4f}")
            forces_rmse_str = (
                f"{eval_metrics['forces_rmse']:.4f} eV/Å"
                if not np.isnan(eval_metrics["forces_rmse"])
                else "N/A (fast eval)"
            )
            logging.info(f"  Forces RMSE:      {forces_rmse_str}")
            logging.info("=" * 60)

            # Log to wandb if enabled
            if wandb_run is not None:
                wandb_run.log({
                    "eval/eigs_rmse": eval_metrics["eigs_rmse"],
                    "eval/weights_rmse": eval_metrics["weights_rmse"],
                    "eval/forces_rmse": eval_metrics["forces_rmse"],
                    "epoch": epoch,
                })

            # Best-model checkpointing
            composite_metric = eval_metrics["eigs_rmse"] + eval_metrics["weights_rmse"]
            if composite_metric < best_composite_metric:
                logging.info(
                    f"  ★ New best model (composite={composite_metric:.4f}, "
                    f"prev best={best_composite_metric:.4f})"
                )
                best_composite_metric = composite_metric
                save_checkpoint(
                    args.checkpoint_path, epoch,
                    model, eigenvalue_head, weight_head, attention_pool,
                    optimizer, lr_scheduler,
                    config=config_dict, filename="best_model.ckpt",
                )

            # Metrics explosion safety check (only active once GNN backbone is unfrozen)
            is_unfrozen = args.unfreeze_epoch is None or epoch >= args.unfreeze_epoch
            exploding_eigs = eval_metrics["eigs_rmse"] > 5.0
            exploding_forces = (
                not np.isnan(eval_metrics["forces_rmse"]) and eval_metrics["forces_rmse"] > 2.0
            )
            if is_unfrozen and (exploding_eigs or exploding_forces):
                logging.warning("Exploding metrics detected! Terminating training run early.")
                break

        # Save checkpoint
        if is_ckpt_epoch:
            save_checkpoint(
                args.checkpoint_path, epoch,
                model, eigenvalue_head, weight_head, attention_pool,
                optimizer, lr_scheduler,
                config=config_dict,
            )

    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse command-line arguments and launch training."""
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
    parser.add_argument("--val_fraction", default=0.1, type=float, help="Fraction of data to hold out for validation (0.0 = train on all, evaluate on all).")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate in the heads (0.0 to disable).")
    parser.add_argument("--eig_loss_type", default="mse", choices=["mse", "huber"], help="Loss type for eigenvalues ('mse' or 'huber').")
    parser.add_argument("--huber_delta", default=1.0, type=float, help="Delta threshold for Huber loss.")
    parser.add_argument("--pdos_peak_boost", default=20.0, type=float, help="PDOS peak boost factor inside loss.")
    parser.add_argument("--pdos_active_threshold", default=0.1, type=float, help="Threshold for active nodes in Cramér shape loss.")
    parser.add_argument("--pdos_magnitude_weight", default=3.0, type=float, help="Magnitude component weight in PDOS loss.")
    parser.add_argument("--pdos_cramer_weight", default=0.5, type=float, help="Cramér shape component weight in PDOS loss.")
    parser.add_argument("--no_couple_heads", action="store_false", dest="couple_heads", help="Do not couple the eigenvalue head to the weight head.")
    parser.add_argument("--detach_coupling", action="store_true", help="Detach the predicted eigenvalues before feeding them to the weight head to prevent weight gradients from flowing back to the eigenvalue head.")

    args, _ = parser.parse_known_args()
    run(args)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    main()