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
from losses import compute_electronic_losses
from models import AttentionPool, WeightHead


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
    """Build the Adam optimizer with per-group learning rates."""
    include_backbone = (not args.freeze_backbone) or (args.unfreeze_epoch is not None)
    params: list[dict[str, Any]] = []

    if include_backbone:
        init_backbone_lr = (
            0.0 if (args.freeze_backbone and args.unfreeze_epoch is not None) else args.backbone_lr
        )
        logging.info(f"Including GNN backbone in optimizer with initial LR: {init_backbone_lr}")

        for name, param in model.named_parameters():
            if re.search(r"(.*bias|.*layer_norm.*|.*batch_norm.*)", name):
                params.append({"params": param, "weight_decay": 0.0, "lr": init_backbone_lr})
            else:
                params.append({"params": param, "lr": init_backbone_lr})
    else:
        logging.info("Excluding GNN backbone parameters from optimizer (permanently frozen).")

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

    When ``args.warmup_epochs`` > 0, prepends a linear warmup ramp to
    whatever base schedule is selected.
    """
    cosine_start_epoch: int | None = None
    warmup_epochs: int = getattr(args, "warmup_epochs", 0)

    if args.scheduler == "cosine":
        logging.info("Initializing CosineAnnealingLR scheduler.")
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=args.min_lr
        )
        cosine_start_epoch = warmup_epochs

    elif args.scheduler == "flat_cosine":
        logging.info("Initializing Flat-Cosine (SequentialLR) scheduler.")
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
            base_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[scheduler1, scheduler2], milestones=[T_flat]
            )
        else:
            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=args.min_lr
            )

    elif args.scheduler == "plateau":
        logging.info("Initializing ReduceLROnPlateau scheduler (stepped per epoch).")
        base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=args.min_lr
        )
        cosine_start_epoch = None

    else:
        logging.info("No learning rate scheduler specified (constant learning rate).")
        base_scheduler = None
        cosine_start_epoch = None

    # Prepend linear warmup if requested
    if warmup_epochs > 0 and base_scheduler is not None:
        if isinstance(base_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # ReduceLROnPlateau can't be composed with SequentialLR; skip warmup
            logging.warning("Warmup is not supported with ReduceLROnPlateau. Skipping warmup.")
            lr_scheduler = base_scheduler
        else:
            logging.info(f"Prepending {warmup_epochs}-epoch linear warmup to scheduler.")
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, base_scheduler],
                milestones=[warmup_epochs],
            )
    else:
        lr_scheduler = base_scheduler

    return lr_scheduler, cosine_start_epoch


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
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
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

    return start_epoch, mean_eigenvalues, std_eigenvalues


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
    eig_loss_type: str = "mse",
    huber_delta: float = 1.0,
    pdos_peak_boost: float = 20.0,
    pdos_active_threshold: float = 0.1,
    pdos_magnitude_weight: float = 3.0,
    pdos_cramer_weight: float = 0.5,
    couple_heads: bool = False,
    detach_coupling: bool = False,
    mean_eigenvalues: torch.Tensor | None = None,
    std_eigenvalues: torch.Tensor | None = None,
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

            eig_loss, weight_loss = compute_electronic_losses(
                pred_eigenvalues, true_eigenvalues_norm,
                pred_weights, true_weights,
                device,
                eig_loss_type=eig_loss_type,
                huber_delta=huber_delta,
                peak_boost=pdos_peak_boost,
                active_threshold=pdos_active_threshold,
                magnitude_weight=pdos_magnitude_weight,
                cramer_weight=pdos_cramer_weight,
            )

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

            total_loss = (
                (energy_loss_weight * energy_loss)
                + (forces_loss_weight * forces_loss)
                + (eigenvalue_loss_weight * eig_loss)
                + (weight_loss_weight * weight_loss)
            )
            scaled_loss = total_loss / accumulation_steps

            batch_outputs: dict[str, torch.Tensor] = {
                "loss/eigenvalues": eig_loss.detach(),
                "loss/weights": weight_loss.detach(),
                "loss/total": total_loss.detach(),
            }
            if is_physics_active:
                batch_outputs["loss/energy"] = energy_loss.detach()
                batch_outputs["loss/forces"] = forces_loss.detach()
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
                window_metrics.update(grad_metrics)
                epoch_metrics.update(grad_metrics)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            valid_accum_count = 0

        window_metrics.update(step_metrics)
        epoch_metrics.update(step_metrics)

        if i % log_freq == 0:
            metrics_dict = window_metrics.get_metrics()
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
) -> dict[str, float]:
    """Evaluate current model checkpoint on cached validation frames."""
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
        if inputs.system_features is None:
            inputs.system_features = {}
        if "total_charge" not in inputs.system_features:
            inputs.system_features["total_charge"] = torch.tensor([0.0], dtype=torch.float32, device=device)
        if "spin_multiplicity" not in inputs.system_features:
            inputs.system_features["spin_multiplicity"] = torch.tensor([1.0], dtype=torch.float32, device=device)

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

        with torch.no_grad():
            gnn_out = model.model(inputs)
            node_feats = gnn_out["node_features"]

            graph_idx = build_graph_index(inputs.n_node, node_feats.device)
            graph_feats = attention_pool(node_feats, graph_idx)

            pred_eigs_tensor = eigenvalue_head(graph_feats)
            if couple_heads:
                node_eigenvalues = pred_eigs_tensor[graph_idx]
                pred_weights = weight_head(node_feats, node_eigenvalues).cpu().numpy().flatten()
            else:
                pred_weights = weight_head(node_feats).cpu().numpy().flatten()

            if mean_eigenvalues is not None and std_eigenvalues is not None:
                pred_eigs_tensor_physical = pred_eigs_tensor * std_eigenvalues + mean_eigenvalues
            else:
                pred_eigs_tensor_physical = pred_eigs_tensor

            pred_eigs = pred_eigs_tensor_physical.cpu().numpy().flatten()

            results["eigs_true"].append(gt["eigenvalues"])
            results["eigs_pred"].append(pred_eigs)
            results["weights_true"].append(np.array(gt["weights"]).flatten())
            results["weights_pred"].append(pred_weights)

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
    """Save a 3-panel parity plot to disk."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].scatter(eig_true, eig_pred, alpha=0.1, s=0.5)
    ax[0].plot([eig_true.min(), eig_true.max()], [eig_true.min(), eig_true.max()], "r--")
    ax[0].set_title(f"Eigenvalues (RMSE: {eigs_rmse:.3f} eV)")
    ax[0].set_xlabel("DFT Eigenvalues (eV)")
    ax[0].set_ylabel("ML Predicted (eV)")

    ax[1].scatter(w_true, w_pred, alpha=0.1, s=0.5)
    ax[1].plot([w_true.min(), w_true.max()], [w_true.min(), w_true.max()], "r--")
    ax[1].set_title(f"PDOS Weights (RMSE: {weights_rmse:.3f})")
    ax[1].set_xlabel("DFT PDOS Weights")
    ax[1].set_ylabel("ML Predicted")

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
