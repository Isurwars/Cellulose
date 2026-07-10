import argparse
import logging
import os
import re
from typing import Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from orb_models.common.models.base import ModelMixin
from orb_models.common.training.metrics import ScalarMetricTracker
from orb_models.common.atoms.abstract_atoms_adapter import AbstractAtomsAdapter

from utils import build_graph_index, prefix_keys
from losses import compute_electronic_losses, compute_force_loss, compute_energy_loss
from models import AttentionPool, ForceResidualHead, WeightHead


class UncertaintyLossWeighting(nn.Module):
    """Learnable homoscedastic uncertainty loss weighting (Kendall et al.)."""
    def __init__(self, tasks: list[str], initial_weights: dict[str, float]) -> None:
        super().__init__()
        self.tasks = tasks
        log_vars = {}
        for task in tasks:
            w = initial_weights.get(task, 1.0)
            w = max(w, 1e-4)
            log_vars[task] = nn.Parameter(torch.tensor(-np.log(w), dtype=torch.float32))
        self.log_vars = nn.ParameterDict(log_vars)

    def forward(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = 0.0
        weighted_losses = {}
        for task, loss in losses.items():
            if task in self.log_vars:
                log_var = self.log_vars[task]
                precision = torch.exp(-log_var)
                total_loss += precision * loss + 0.5 * log_var
                weighted_losses[f"uncertainty_weight/{task}"] = precision.item()
            else:
                total_loss += loss
        return total_loss, weighted_losses


def build_loss_weights(args: argparse.Namespace) -> dict[str, float]:
    """Convert CLI loss-weight arguments into the dict expected by Orb models."""
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

    # Log eigenvalues and weights weights if they are present in args
    log_weights = dict(loss_weights)
    if getattr(args, "eigenvalue_loss_weight", None) is not None:
        log_weights["eigenvalues"] = args.eigenvalue_loss_weight
    if getattr(args, "weight_loss_weight", None) is not None:
        log_weights["weights"] = args.weight_loss_weight

    if log_weights:
        logging.info("=" * 60)
        logging.info("Custom loss weights specified:")
        for key, val in log_weights.items():
            logging.info(f"  {key}: {val}")
        logging.info("=" * 60)

    return loss_weights


def build_optimizer(
    model: ModelMixin,
    eigenvalue_head: nn.Module,
    weight_head: nn.Module,
    attention_pool: AttentionPool,
    args: argparse.Namespace,
    force_residual_head: ForceResidualHead | None = None,
    uncertainty_weighting: nn.Module | None = None,
) -> torch.optim.Optimizer:
    """Build the Adam optimizer with two parameter groups: backbone and heads."""
    params: list[dict[str, Any]] = []

    # 1. GNN Backbone group
    include_backbone = (not args.freeze_backbone) or (args.unfreeze_epoch is not None)
    if include_backbone:
        init_backbone_lr = (
            0.0 if (args.freeze_backbone and args.unfreeze_epoch is not None and args.unfreeze_epoch > 0) else args.backbone_lr
        )
        logging.info(f"Including GNN backbone in optimizer with initial LR: {init_backbone_lr}")
        params.append({"params": list(model.parameters()), "lr": init_backbone_lr})
    else:
        logging.info("Excluding GNN backbone parameters from optimizer (permanently frozen).")

    # 2. Heads group (all heads, pool, and uncertainty weighting)
    head_params = (
        list(eigenvalue_head.parameters()) +
        list(weight_head.parameters()) +
        list(attention_pool.parameters())
    )
    if force_residual_head is not None:
        head_params += list(force_residual_head.parameters())
    if uncertainty_weighting is not None:
        head_params += list(uncertainty_weighting.parameters())

    params.append({"params": head_params, "lr": args.lr})

    return torch.optim.Adam(params)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    total_epochs: int,
    unfreeze_offset: int | None = None,
) -> tuple[torch.optim.lr_scheduler._LRScheduler | None, int | None]:
    """Build the learning-rate scheduler.

    When ``args.warmup_epochs`` > 0, prepends a linear warmup ramp to
    whatever base schedule is selected. Employs a flat SequentialLR
    structure to avoid nested SequentialLR issues.
    """
    cosine_start_epoch: int | None = None
    warmup_epochs: int = getattr(args, "warmup_epochs", 0)

    schedulers: list[torch.optim.lr_scheduler._LRScheduler] = []

    # 1. Warmup Phase
    if warmup_epochs > 0:
        if args.scheduler == "plateau":
            logging.warning("Warmup is not supported with ReduceLROnPlateau. Skipping warmup.")
        else:
            logging.info(f"Adding {warmup_epochs}-epoch linear warmup to scheduler.")
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs
            )
            schedulers.append(warmup_scheduler)

    # 2. Base Scheduler Phases
    if args.scheduler == "cosine":
        logging.info("Initializing CosineAnnealingLR scheduler.")
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=args.min_lr
        )
        schedulers.append(cosine_scheduler)
        cosine_start_epoch = warmup_epochs

    elif args.scheduler == "flat_cosine":
        logging.info("Initializing Flat-Cosine scheduler.")
        T_flat = (total_epochs - warmup_epochs) // 2

        if unfreeze_offset is not None:
            cosine_start_epoch = warmup_epochs + unfreeze_offset + (total_epochs - warmup_epochs - unfreeze_offset) // 2
        else:
            cosine_start_epoch = warmup_epochs + (total_epochs - warmup_epochs) // 2

        if T_flat > 0:
            scheduler1 = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=T_flat
            )
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_epochs - warmup_epochs - T_flat), eta_min=args.min_lr
            )
            schedulers.append(scheduler1)
            schedulers.append(scheduler2)
        else:
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=args.min_lr
            )
            schedulers.append(cosine_scheduler)

    elif args.scheduler == "plateau":
        logging.info("Initializing ReduceLROnPlateau scheduler (stepped per epoch).")
        # Plateau cannot be used in SequentialLR, so we return it directly
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=args.min_lr
        )
        return plateau_scheduler, None

    else:
        logging.info("No learning rate scheduler specified (constant learning rate).")
        if len(schedulers) == 1:
            # Only warmup scheduler exists; append constant scheduler for remaining epochs
            remaining_epochs = max(1, total_epochs - warmup_epochs)
            constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=remaining_epochs
            )
            schedulers.append(constant_scheduler)
        elif len(schedulers) == 0:
            return None, None

    # Construct the flat SequentialLR if we have multiple schedulers
    if len(schedulers) == 1:
        return schedulers[0], cosine_start_epoch
    else:
        milestones = []
        accumulated = 0
        if warmup_epochs > 0:
            accumulated += warmup_epochs
            milestones.append(accumulated)
            if args.scheduler == "flat_cosine" and T_flat > 0:
                accumulated += T_flat
                milestones.append(accumulated)
        else:
            if args.scheduler == "flat_cosine" and T_flat > 0:
                accumulated += T_flat
                milestones.append(accumulated)

        logging.info(f"Constructing SequentialLR with milestones: {milestones}")
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=schedulers, milestones=milestones
        ), cosine_start_epoch


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
    *,
    mean_eigenvalues: torch.Tensor | None = None,
    std_eigenvalues: torch.Tensor | None = None,
    std_forces: torch.Tensor | None = None,
    force_residual_head: ForceResidualHead | None = None,
    uncertainty_weighting: nn.Module | None = None,
) -> None:
    """Save a training checkpoint to disk."""
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
        "mean_eigenvalues": mean_eigenvalues,
        "std_eigenvalues": std_eigenvalues,
        "std_forces": std_forces,
        "force_residual_head_state": force_residual_head.state_dict() if force_residual_head is not None else None,
        "uncertainty_weighting_state": uncertainty_weighting.state_dict() if uncertainty_weighting is not None else None,
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
    force_residual_head: ForceResidualHead | None = None,
    uncertainty_weighting: nn.Module | None = None,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Restore model, heads, attention pool, optimizer, and scheduler from a checkpoint."""
    logging.info(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    eigenvalue_head.load_state_dict(checkpoint["eigenvalue_head_state"])
    weight_head.load_state_dict(checkpoint["weight_head_state"])

    if "attention_pool_state" in checkpoint:
        attention_pool.load_state_dict(checkpoint["attention_pool_state"])
    else:
        logging.warning("No attention_pool_state in checkpoint (old format); using fresh init.")

    if force_residual_head is not None and "force_residual_head_state" in checkpoint and checkpoint["force_residual_head_state"] is not None:
        force_residual_head.load_state_dict(checkpoint["force_residual_head_state"])
        logging.info("Loaded force_residual_head state from checkpoint.")
    elif force_residual_head is not None:
        logging.warning("No force_residual_head_state in checkpoint; using fresh init.")

    if uncertainty_weighting is not None and "uncertainty_weighting_state" in checkpoint and checkpoint["uncertainty_weighting_state"] is not None:
        uncertainty_weighting.load_state_dict(checkpoint["uncertainty_weighting_state"])
        logging.info("Loaded uncertainty_weighting state from checkpoint.")

    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except ValueError as e:
        logging.warning(f"Could not load optimizer state (might be incompatible): {e}")

    start_epoch: int = checkpoint.get("epoch", -1) + 1
    logging.info(f"Resumed at epoch: {start_epoch}")

    if unfreeze_epoch is not None and start_epoch >= unfreeze_epoch:
        logging.info("Resuming after unfreeze epoch. Ensuring GNN backbone parameters are unfrozen.")
        for param in model.parameters():
            param.requires_grad = True

    if lr_scheduler is not None and "lr_scheduler" in checkpoint and checkpoint["lr_scheduler"] is not None:
        try:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            logging.info("Loaded lr_scheduler state from checkpoint.")
        except Exception as e:
            logging.warning(f"Could not load lr_scheduler state: {e}")

    mean_eigenvalues = checkpoint.get("mean_eigenvalues", None)
    std_eigenvalues = checkpoint.get("std_eigenvalues", None)
    std_forces = checkpoint.get("std_forces", None)

    return start_epoch, mean_eigenvalues, std_eigenvalues, std_forces


def finetune(
    model: ModelMixin,
    eigenvalue_head: nn.Module,
    weight_head: nn.Module,
    attention_pool: AttentionPool,
    optimizer: torch.optim.Optimizer,
    dataloader: Any,
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
    couple_heads: bool = False,
    detach_coupling: bool = False,
    mean_eigenvalues: torch.Tensor | None = None,
    std_eigenvalues: torch.Tensor | None = None,
    std_forces: torch.Tensor | None = None,
    force_residual_head: ForceResidualHead | None = None,
    uncertainty_weighting: nn.Module | None = None,
) -> dict[str, float]:
    """Run one epoch of electronic-structure finetuning."""
    run_handle: Any | None = wandb.run if WANDB_AVAILABLE else None
    # Two trackers: window_metrics for periodic logging, epoch_metrics for
    # the full-epoch average returned at the end.
    window_metrics = ScalarMetricTracker()
    epoch_metrics = ScalarMetricTracker()

    if freeze_backbone:
        model.eval()
    else:
        model.train()
    eigenvalue_head.train()
    weight_head.train()
    attention_pool.train()
    if force_residual_head is not None:
        force_residual_head.train()

    num_training_batches: int | float
    if num_steps is not None:
        num_training_batches = num_steps
    else:
        try:
            num_training_batches = len(dataloader)
        except TypeError:
            raise ValueError("Dataloader has no length, you must specify num_steps.")

    batch_generator_tqdm = tqdm.tqdm(iter(dataloader), total=num_training_batches)

    # Track how many valid backward passes have been accumulated since the
    # last optimizer step so NaN-skipped batches don't trigger premature steps.
    valid_accum_count = 0

    for i, batch in enumerate(batch_generator_tqdm):
        if num_steps and i == num_steps:
            break

        if i % log_freq == 0:
            window_metrics.reset()

        batch = batch.to(device)

        step_metrics: dict[str, float] = {
            "batch_size": float(len(batch.n_node)),
            "batch_num_edges": float(batch.n_edge.sum()),
            "batch_num_nodes": float(batch.n_node.sum()),
        }

        with torch.autocast("cuda", enabled=False):
            gnn_out = model.model(batch)
            node_features = gnn_out["node_features"]

            graph_idx = build_graph_index(batch.n_node, node_features.device)
            graph_features = attention_pool(node_features, graph_idx)

            pred_eigenvalues = eigenvalue_head(graph_features)

            if couple_heads:
                node_eigenvalues = pred_eigenvalues[graph_idx]
                if detach_coupling:
                    node_eigenvalues = node_eigenvalues.detach()
                pred_weights = weight_head(node_features, node_eigenvalues)
            else:
                pred_weights = weight_head(node_features)

            true_eigenvalues = batch.system_targets["eigenvalues"]
            if mean_eigenvalues is not None and std_eigenvalues is not None:
                true_eigenvalues_norm = (true_eigenvalues - mean_eigenvalues) / std_eigenvalues
            else:
                true_eigenvalues_norm = true_eigenvalues

            true_weights = batch.node_targets["weights"]

            eig_loss, weight_loss, electronic_diag = compute_electronic_losses(
                pred_eigenvalues, true_eigenvalues_norm,
                pred_weights, true_weights,
                device,
            )

            has_physics = (energy_loss_weight > 0.0 or forces_loss_weight > 0.0)

            if has_physics:
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

                # Denormalize forces to physical units before adding residual/computing loss
                pred_forces = model.heads["forces"].denormalize(pred_forces, batch)

                # Apply force residual correction if available
                if force_residual_head is not None:
                    force_correction = force_residual_head(node_features.detach())
                    pred_forces = pred_forces + force_correction

                true_forces = batch.node_targets["forces"]
                forces_loss, force_diag = compute_force_loss(
                    pred_forces, true_forces,
                    std_forces=std_forces,
                )

                if energy_loss_weight > 0.0:
                    pred_energy = model.heads["energy"].denormalize(pred_energy, batch)
                    true_energy = batch.system_targets["energy"]
                    energy_loss, energy_diag = compute_energy_loss(pred_energy, true_energy)
                else:
                    energy_loss = torch.tensor(0.0, device=device)
                    energy_diag = {}
            else:
                energy_loss = torch.tensor(0.0, device=device)
                forces_loss = torch.tensor(0.0, device=device)
                force_diag = {}

            if uncertainty_weighting is not None:
                losses_to_weight = {}
                if eigenvalue_loss_weight > 0.0:
                    losses_to_weight["eigenvalues"] = eig_loss
                if weight_loss_weight > 0.0:
                    losses_to_weight["weights"] = weight_loss
                if has_physics:
                    if energy_loss_weight > 0.0:
                        losses_to_weight["energy"] = energy_loss
                    if forces_loss_weight > 0.0:
                        losses_to_weight["forces"] = forces_loss
                
                total_loss, uw_weights = uncertainty_weighting(losses_to_weight)
            else:
                total_loss = (
                    (energy_loss_weight * energy_loss)
                    + (forces_loss_weight * forces_loss)
                    + (eigenvalue_loss_weight * eig_loss)
                    + (weight_loss_weight * weight_loss)
                )
                uw_weights = {}

            scaled_loss = total_loss / accumulation_steps

            batch_outputs: dict[str, torch.Tensor] = {
                "loss/eigenvalues": eig_loss.detach(),
                "loss/weights": weight_loss.detach(),
                "loss/total": total_loss.detach(),
            }
            if has_physics:
                batch_outputs["loss/energy"] = energy_loss.detach()
                batch_outputs["loss/forces"] = forces_loss.detach()

            for k, v in uw_weights.items():
                batch_outputs[k] = torch.tensor(v, device=device)

            # Force diagnostics as plain floats
            for k, v in force_diag.items():
                batch_outputs[f"forces/{k}"] = torch.tensor(v)

            # Energy diagnostics as plain floats
            if has_physics:
                for k, v in energy_diag.items():
                    batch_outputs[f"energy/{k}"] = torch.tensor(v, device=device)

            # Electronic diagnostics as plain floats
            for k, v in electronic_diag.items():
                batch_outputs[f"electronic/{k}"] = torch.tensor(v, device=device)

            window_metrics.update(batch_outputs)
            epoch_metrics.update(batch_outputs)

        # NaN guard: skip backward *before* calling it so that previously
        # accumulated (valid) gradients are preserved for the next step.
        if torch.isnan(scaled_loss):
            logging.warning(f"NaN scaled_loss at step {i}. Skipping backward for this batch.")
            continue

        scaled_loss.backward()
        valid_accum_count += 1

        if valid_accum_count >= accumulation_steps or (i + 1) == num_training_batches:
            if clip_grad is not None:
                backbone_gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                eig_head_gnorm = torch.nn.utils.clip_grad_norm_(eigenvalue_head.parameters(), clip_grad)
                w_head_gnorm = torch.nn.utils.clip_grad_norm_(weight_head.parameters(), clip_grad)
                attn_pool_gnorm = torch.nn.utils.clip_grad_norm_(attention_pool.parameters(), clip_grad)
                grad_metrics = {
                    "grad_norm/backbone": backbone_gnorm.detach(),
                    "grad_norm/eigenvalue_head": eig_head_gnorm.detach(),
                    "grad_norm/weight_head": w_head_gnorm.detach(),
                    "grad_norm/attention_pool": attn_pool_gnorm.detach(),
                }
                if force_residual_head is not None:
                    force_head_gnorm = torch.nn.utils.clip_grad_norm_(force_residual_head.parameters(), clip_grad)
                    grad_metrics["grad_norm/force_residual_head"] = force_head_gnorm.detach()
                window_metrics.update(grad_metrics)
                epoch_metrics.update(grad_metrics)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            valid_accum_count = 0

        window_metrics.update(step_metrics)
        epoch_metrics.update(step_metrics)

        if i % log_freq == 0:
            metrics_dict = window_metrics.get_metrics()
            loss_summary = ", ".join(f"{k}: {v:.4f}" for k, v in metrics_dict.items() if "loss/" in k)
            forces_summary = ", ".join(f"{k}: {v:.4f}" for k, v in metrics_dict.items() if "forces/" in k)
            log_msg = f"Epoch {epoch} [Step {i}/{num_training_batches}] — {loss_summary}"
            if forces_summary:
                log_msg += f" | {forces_summary}"
            logging.info(log_msg)
            if run_handle is not None:
                step = (epoch * num_training_batches) + i
                if run_handle.sweep_id is not None:
                    run_handle.log({"loss": metrics_dict["loss/total"]}, commit=False)
                run_handle.log({"step": step}, commit=False)
                run_handle.log(prefix_keys(metrics_dict, "finetune_step"), commit=True)

    return epoch_metrics.get_metrics()


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
    mean_eigenvalues: torch.Tensor | None = None,
    std_eigenvalues: torch.Tensor | None = None,
    force_residual_head: ForceResidualHead | None = None,
) -> dict[str, float]:
    """Evaluate current model checkpoint on cached validation frames."""
    model.eval()
    eigenvalue_head.eval()
    weight_head.eval()
    attention_pool.eval()
    if force_residual_head is not None:
        force_residual_head.eval()

    results: dict[str, list[Any]] = {
        "forces_true": [], "forces_pred": [],
        "eigs_true": [], "eigs_pred": [],
        "weights_true": [], "weights_pred": [],
    }

    frames_to_eval = eval_frames[:100] if fast_eval else eval_frames

    for single_graph, gt in frames_to_eval:
        inputs = atoms_adapter.batch([single_graph]).to(device)
        if inputs.system_features is None:
            inputs.system_features = {}
        if "total_charge" not in inputs.system_features:
            inputs.system_features["total_charge"] = torch.tensor([0.0], dtype=torch.float32, device=device)
        if "spin_multiplicity" not in inputs.system_features:
            inputs.system_features["spin_multiplicity"] = torch.tensor([1.0], dtype=torch.float32, device=device)

        if is_conservative_model:
            with torch.set_grad_enabled(True):
                inputs.positions.requires_grad_(True)
                base_out = model(inputs)
                pred_forces = base_out["grad_forces"]
        else:
            with torch.no_grad():
                base_out = model(inputs)
                pred_forces = base_out["forces"]

        # Denormalize to physical units
        pred_forces = model.heads["forces"].denormalize(pred_forces, inputs)

        with torch.no_grad():
            gnn_out = model.model(inputs)
            node_feats = gnn_out["node_features"]

            # Apply force residual correction if available
            if force_residual_head is not None:
                force_correction = force_residual_head(node_feats)
                pred_forces = pred_forces + force_correction

        results["forces_true"].append(gt["forces"])
        results["forces_pred"].append(pred_forces.detach().cpu().numpy())

        with torch.no_grad():
            graph_idx = build_graph_index(inputs.n_node, node_feats.device)
            graph_feats = attention_pool(node_feats, graph_idx)

            pred_eigs_tensor = eigenvalue_head(graph_feats)
            if couple_heads:
                node_eigenvalues = pred_eigs_tensor[graph_idx]
                pred_weights_logits = weight_head(node_feats, node_eigenvalues)
            else:
                pred_weights_logits = weight_head(node_feats)
            pred_weights = torch.sigmoid(pred_weights_logits).cpu().numpy().flatten()

            if mean_eigenvalues is not None and std_eigenvalues is not None:
                pred_eigs_tensor_physical = pred_eigs_tensor * std_eigenvalues + mean_eigenvalues
            else:
                pred_eigs_tensor_physical = pred_eigs_tensor

            pred_eigs = pred_eigs_tensor_physical.cpu().numpy().flatten()

            results["eigs_true"].append(gt["eigenvalues"])
            results["eigs_pred"].append(pred_eigs)

            w_true_np = np.array(gt["weights"]).flatten()
            is_w_transformed = (w_true_np.min() < -0.1) or (w_true_np.max() > 1.1)
            if is_w_transformed:
                w_true_physical = 1.0 / (1.0 + np.exp(-w_true_np))
            else:
                w_true_physical = w_true_np

            results["weights_true"].append(w_true_physical)
            results["weights_pred"].append(pred_weights)

    f_true = np.concatenate(results["forces_true"]).flatten()
    f_pred = np.concatenate(results["forces_pred"]).flatten()
    forces_rmse = float(np.sqrt(np.mean((f_true - f_pred) ** 2)))
    forces_var = np.var(f_true)
    forces_r2 = float(1.0 - np.mean((f_true - f_pred) ** 2) / forces_var) if forces_var > 1e-8 else 0.0

    eig_true = np.array(results["eigs_true"]).flatten()
    eig_pred = np.array(results["eigs_pred"]).flatten()
    eigs_rmse = float(np.sqrt(np.mean((eig_true - eig_pred) ** 2)))
    eig_var = np.var(eig_true)
    eigs_r2 = float(1.0 - np.mean((eig_true - eig_pred) ** 2) / eig_var) if eig_var > 1e-8 else 0.0

    w_true = np.concatenate(results["weights_true"])
    w_pred = np.concatenate(results["weights_pred"])
    weights_rmse = float(np.sqrt(np.mean((w_true - w_pred) ** 2)))
    w_var = np.var(w_true)
    weights_r2 = float(1.0 - np.mean((w_true - w_pred) ** 2) / w_var) if w_var > 1e-8 else 0.0

    if plot_path is not None:
        _save_parity_plots(
            eig_true, eig_pred, eigs_rmse, eigs_r2,
            w_true, w_pred, weights_rmse, weights_r2,
            f_true, f_pred, forces_rmse, forces_r2,
            plot_path,
        )

    return {
        "forces_rmse": forces_rmse,
        "forces_r2": forces_r2,
        "eigs_rmse": eigs_rmse,
        "eigs_r2": eigs_r2,
        "weights_rmse": weights_rmse,
        "weights_r2": weights_r2,
    }


def _save_parity_plots(
    eig_true: np.ndarray,
    eig_pred: np.ndarray,
    eigs_rmse: float,
    eigs_r2: float,
    w_true: np.ndarray,
    w_pred: np.ndarray,
    weights_rmse: float,
    weights_r2: float,
    f_true: np.ndarray | None,
    f_pred: np.ndarray | None,
    forces_rmse: float,
    forces_r2: float,
    plot_path: str,
) -> None:
    """Save a 3-panel parity plot to disk."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].scatter(eig_true, eig_pred, alpha=0.1, s=0.5)
    ax[0].plot([eig_true.min(), eig_true.max()], [eig_true.min(), eig_true.max()], "r--")
    ax[0].set_title(f"Eigenvalues (RMSE: {eigs_rmse:.3f} eV, R²: {eigs_r2:.4f})")
    ax[0].set_xlabel("DFT Eigenvalues (eV)")
    ax[0].set_ylabel("ML Predicted (eV)")

    ax[1].scatter(w_true, w_pred, alpha=0.1, s=0.5)
    ax[1].plot([w_true.min(), w_true.max()], [w_true.min(), w_true.max()], "r--")
    ax[1].set_title(f"PDOS Weights (RMSE: {weights_rmse:.3f}, R²: {weights_r2:.4f})")
    ax[1].set_xlabel("DFT PDOS Weights")
    ax[1].set_ylabel("ML Predicted")

    if f_true is not None and f_pred is not None:
        ax[2].scatter(f_true, f_pred, alpha=0.3, s=1)
        ax[2].plot([f_true.min(), f_true.max()], [f_true.min(), f_true.max()], "r--")
        ax[2].set_title(f"Forces (RMSE: {forces_rmse:.3f} eV/Å, R²: {forces_r2:.4f})")
    else:
        ax[2].set_title("Forces (skipped — fast eval)")
    ax[2].set_xlabel("DFT Forces (eV/Å)")
    ax[2].set_ylabel("ML Predicted (eV/Å)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved parity plot to {plot_path}")
