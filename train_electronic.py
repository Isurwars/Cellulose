# @file train_electronic.py
# @copyright Copyright © 2026 Isaías Rodríguez (isurwars@gmail.com)
# @par License
# SPDX-License-Identifier: AGPL-3.0-only

"""
train_electronic.py — Electronic Structure Finetuning CLI Entrypoint

Refactored to import sub-modules: utils, losses, models, data, trainer.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from datetime import datetime
from typing import Any

import ase.db
import numpy as np
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from orb_models.common.dataset import property_definitions
from orb_models.common.training.util import init_device
from orb_models.common.utils import seed_everything
from orb_models.forcefield import pretrained

# Side-effect: Importing data registers "eigenvalues" and "weights" in the global registry
import data
from utils import split_train_val
from models import build_heads
from trainer import (
    build_loss_weights,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    resume_checkpoint,
    finetune,
    evaluate_model,
)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logging(log_dir: str) -> str:
    """Configure logging to write to both console and a timestamped logfile.

    Returns the path to the created logfile.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"training_{timestamp}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers (e.g. from basicConfig or reimport)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)

    logging.info(f"Logging to: {log_path}")
    return log_path

LATENT_DIM: int = 256
"""Node embedding dimensionality of orb_v3 omol models."""


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


def _format_lr_summary(optimizer: torch.optim.Optimizer) -> str:
    """Return a compact LR string: single value if uniform, backbone/heads split otherwise."""
    lrs = [g["lr"] for g in optimizer.param_groups]
    unique_lrs = set(f"{lr:.2e}" for lr in lrs)
    if len(unique_lrs) == 1:
        return next(iter(unique_lrs))
    # Group by distinct LR values and count param groups
    counts = Counter(f"{lr:.2e}" for lr in lrs)
    return ", ".join(f"{lr} ({n} groups)" for lr, n in counts.items())


def run(args: argparse.Namespace) -> None:
    """Top-level training orchestrator."""
    log_path = setup_logging(args.checkpoint_path)
    logging.info(f"Full configuration: {vars(args)}")

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
        custom_refs = data.load_custom_reference_energies(args.custom_reference_energies).to(device)

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
    eigenvalue_head, weight_head, attention_pool, force_residual_head = build_heads(
        LATENT_DIM, device, dropout=args.dropout, couple_heads=args.couple_heads,
        use_force_residual=args.use_force_residual
    )
    model.to(device=device)

    # --- Freeze backbone if requested ---
    if args.freeze_backbone:
        logging.info("Initially freezing GNN backbone parameters.")
        for param in model.parameters():
            param.requires_grad = False

    # --- Optimizer ---
    optimizer = build_optimizer(
        model, eigenvalue_head, weight_head, attention_pool, args,
        force_residual_head=force_residual_head
    )

    # --- Scheduler ---
    lr_scheduler, cosine_start_epoch = build_scheduler(
        optimizer, args,
        total_epochs=args.max_epochs,
        unfreeze_offset=args.unfreeze_epoch,
    )

    # --- Resume from checkpoint ---
    start_epoch = 0
    checkpoint_mean_eigs = None
    checkpoint_std_eigs = None
    checkpoint_std_forces = None
    if args.resume_from_checkpoint:
        start_epoch, checkpoint_mean_eigs, checkpoint_std_eigs, checkpoint_std_forces = resume_checkpoint(
            args.resume_from_checkpoint,
            model, eigenvalue_head, weight_head, attention_pool,
            optimizer, lr_scheduler, device,
            unfreeze_epoch=args.unfreeze_epoch,
            force_residual_head=force_residual_head,
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

    mean_eigenvalues: torch.Tensor | None = None
    std_eigenvalues: torch.Tensor | None = None
    if args.normalize_eigenvalues:
        if checkpoint_mean_eigs is not None and checkpoint_std_eigs is not None:
            mean_eigenvalues = checkpoint_mean_eigs
            std_eigenvalues = checkpoint_std_eigs
            logging.info("Using eigenvalue normalization statistics from resumed checkpoint.")
        else:
            logging.info("Computing training set eigenvalue statistics for normalization...")
            db = ase.db.connect(args.data_path)
            train_indices_set = set(train_indices) if train_indices is not None else set(range(db.count()))
            all_eigs = []
            for idx, row in enumerate(db.select()):
                if idx in train_indices_set:
                    all_eigs.append(row.data["eigenvalues"])
            all_eigs = np.array(all_eigs)
            mean_eigenvalues = torch.tensor(all_eigs.mean(axis=0), dtype=torch.float32, device=device)
            std_eigenvalues = torch.tensor(all_eigs.std(axis=0), dtype=torch.float32, device=device)
            std_eigenvalues = torch.clamp(std_eigenvalues, min=1e-5)
            logging.info("Eigenvalue statistics computed.")

    std_forces: torch.Tensor | None = None
    if args.normalize_forces:
        if checkpoint_std_forces is not None:
            std_forces = checkpoint_std_forces
            logging.info("Using force normalization statistics from resumed checkpoint.")
        else:
            logging.info("Computing training set force statistics for normalization...")
            db = ase.db.connect(args.data_path)
            train_indices_set = set(train_indices) if train_indices is not None else set(range(db.count()))
            all_forces = []
            for idx, row in enumerate(db.select()):
                if idx in train_indices_set and hasattr(row, 'forces') and row.forces is not None:
                    all_forces.append(row.forces)
            if len(all_forces) > 0:
                all_forces = np.concatenate(all_forces, axis=0)
                var_forces = float(np.var(all_forces))
                std_forces = torch.tensor(np.sqrt(var_forces), dtype=torch.float32, device=device)
                std_forces = torch.clamp(std_forces, min=1e-5)
                logging.info(f"Force statistics computed. Force variance: {var_forces:.6f} (std: {std_forces.item():.6f})")
            else:
                logging.warning("No forces found in training set. Force normalization disabled.")

    train_loader = data.build_train_loader(
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
        target_property_config = property_definitions.instantiate_property_config(
            {"graph": graph_targets, "node": ["forces", "weights"]}
        )
        eval_dataset = data.AseSqliteDataset(
            args.dataset + "_eval",
            args.data_path,
            atoms_adapter=atoms_adapter,
            target_config=target_property_config,
            augmentations=[],
        )
        eval_frames = data.cache_eval_frames(eval_dataset, val_indices=val_indices)

    # --- Training loop ---
    logging.info("Starting training!")
    num_steps = args.num_steps if args.num_steps > 0 else None
    best_composite_metric = float("inf")
    config_dict = vars(args)

    for epoch in range(start_epoch, args.max_epochs):
        is_currently_frozen = args.freeze_backbone and (
            args.unfreeze_epoch is None or epoch < args.unfreeze_epoch
        )

        for param in model.parameters():
            param.requires_grad = not is_currently_frozen

        if args.unfreeze_epoch is not None and epoch == args.unfreeze_epoch:
            logging.info(f"--- Unfreezing GNN backbone at epoch {epoch} ---")
            head_params_list = list(eigenvalue_head.parameters()) + list(weight_head.parameters()) + list(attention_pool.parameters())
            if force_residual_head is not None:
                head_params_list += list(force_residual_head.parameters())
            head_params_set = set(head_params_list)
            for idx, group in enumerate(optimizer.param_groups):
                is_backbone = any(p not in head_params_set for p in group["params"])
                if is_backbone:
                    group["lr"] = args.backbone_lr
                    logging.info(f"  GNN Backbone group {idx} LR set to: {args.backbone_lr}")
                else:
                    group["lr"] = args.lr
                    logging.info(f"  Head group {idx} LR reset to: {args.lr}")
                group.pop("initial_lr", None)

            remaining_epochs = args.max_epochs - epoch
            lr_scheduler, cosine_start_epoch = build_scheduler(
                optimizer, args,
                total_epochs=remaining_epochs,
            )
            if cosine_start_epoch is not None:
                cosine_start_epoch += epoch

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

        lr_summary = _format_lr_summary(optimizer)
        logging.info(f"Epoch {epoch} — LR: {lr_summary}")

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
            pdos_cosine_weight=args.pdos_cosine_weight,
            couple_heads=args.couple_heads,
            detach_coupling=args.detach_coupling,
            mean_eigenvalues=mean_eigenvalues,
            std_eigenvalues=std_eigenvalues,
            std_forces=std_forces,
            force_residual_head=force_residual_head,
            force_huber_delta=args.force_huber_delta,
        )

        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(epoch_metrics["loss/total"])
                logging.info(f"Stepped ReduceLROnPlateau with loss: {epoch_metrics['loss/total']:.4f}")
            else:
                lr_scheduler.step()
                logging.info("Stepped LR scheduler.")

        is_ckpt_epoch = (epoch % args.save_every_x_epochs == 0) or (epoch == args.max_epochs - 1)

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
                mean_eigenvalues=mean_eigenvalues,
                std_eigenvalues=std_eigenvalues,
                force_residual_head=force_residual_head,
            )
            logging.info("=" * 60)
            logging.info(f"Epoch {epoch} Evaluation Metrics:")
            logging.info(f"  Eigenvalues RMSE: {eval_metrics['eigs_rmse']:.4f} eV (R²: {eval_metrics['eigs_r2']:.4f})")
            logging.info(f"  Weights RMSE:     {eval_metrics['weights_rmse']:.4f} (R²: {eval_metrics['weights_r2']:.4f})")
            forces_rmse_str = (
                f"{eval_metrics['forces_rmse']:.4f} eV/Å"
                if not np.isnan(eval_metrics["forces_rmse"])
                else "N/A (fast eval)"
            )
            forces_r2_str = (
                f"{eval_metrics['forces_r2']:.4f}"
                if not np.isnan(eval_metrics["forces_r2"])
                else "N/A"
            )
            logging.info(f"  Forces RMSE:      {forces_rmse_str} (R²: {forces_r2_str})")
            logging.info("=" * 60)

            if wandb_run is not None:
                wandb_run.log({
                    "eval/eigs_rmse": eval_metrics["eigs_rmse"],
                    "eval/eigs_r2": eval_metrics["eigs_r2"],
                    "eval/weights_rmse": eval_metrics["weights_rmse"],
                    "eval/weights_r2": eval_metrics["weights_r2"],
                    "eval/forces_rmse": eval_metrics["forces_rmse"],
                    "eval/forces_r2": eval_metrics["forces_r2"],
                    "epoch": epoch,
                })

            # Composite metric: sum of eigenvalue, weights, and force RMSE
            forces_val = eval_metrics["forces_rmse"] if not np.isnan(eval_metrics["forces_rmse"]) else 0.0
            composite_metric = eval_metrics["eigs_rmse"] + eval_metrics["weights_rmse"] + 0.5 * forces_val
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
                    mean_eigenvalues=mean_eigenvalues,
                    std_eigenvalues=std_eigenvalues,
                    std_forces=std_forces,
                    force_residual_head=force_residual_head,
                )

            is_unfrozen = args.unfreeze_epoch is None or epoch >= args.unfreeze_epoch
            exploding_eigs = eval_metrics["eigs_rmse"] > 5.0
            exploding_forces = (
                not np.isnan(eval_metrics["forces_rmse"]) and eval_metrics["forces_rmse"] > 2.0
            )
            if is_unfrozen and (exploding_eigs or exploding_forces):
                logging.warning("Exploding metrics detected! Terminating training run early.")
                break

        if is_ckpt_epoch:
            save_checkpoint(
                args.checkpoint_path, epoch,
                model, eigenvalue_head, weight_head, attention_pool,
                optimizer, lr_scheduler,
                config=config_dict,
                mean_eigenvalues=mean_eigenvalues,
                std_eigenvalues=std_eigenvalues,
                std_forces=std_forces,
                force_residual_head=force_residual_head,
            )

    if wandb_run is not None:
        wandb_run.finish()


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
    parser.add_argument("--warmup_epochs", default=1, type=int, help="Number of epochs for linear LR warmup (0 to disable).")
    parser.add_argument("--weight_head_noise_std", default=0.0, type=float, help="Standard deviation of noise to inject into weight_head parameters to break false minima.")
    parser.add_argument("--weight_head_noise_interval", default=5, type=int, help="Epoch interval at which noise is injected into weight_head.")
    parser.add_argument("--eigenvalue_loss_weight", default=0.02, type=float, help="Loss weight scaling factor for eigenvalues.")
    parser.add_argument("--normalize_eigenvalues", action="store_true", help="Normalize target eigenvalues using training set mean/std.")
    parser.add_argument("--normalize_forces", action="store_true", help="Normalize target forces by dividing the loss by the training set force variance.")
    parser.add_argument("--weight_loss_weight", default=1.0, type=float, help="Loss weight scaling factor for PDOS weights.")
    parser.add_argument("--eval_every_x_epochs", default=1, type=int, help="Frequency of running evaluation on cached database frames. Set to 0 to disable.")
    parser.add_argument("--val_fraction", default=0.1, type=float, help="Fraction of data to hold out for validation (0.0 = train on all, evaluate on all).")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate in the heads (0.0 to disable).")
    parser.add_argument("--eig_loss_type", default="mse", choices=["mse", "huber"], help="Loss type for eigenvalues ('mse' or 'huber').")
    parser.add_argument("--huber_delta", default=1.0, type=float, help="Delta threshold for Huber loss.")
    parser.add_argument("--pdos_peak_boost", default=5.0, type=float, help="PDOS peak boost factor inside loss.")
    parser.add_argument("--pdos_active_threshold", default=0.1, type=float, help="Threshold for active nodes in Cramér shape loss.")
    parser.add_argument("--pdos_magnitude_weight", default=3.0, type=float, help="Magnitude component weight in PDOS loss.")
    parser.add_argument("--pdos_cramer_weight", default=0.5, type=float, help="Cramér shape component weight in PDOS loss.")
    parser.add_argument("--pdos_cosine_weight", default=0.3, type=float, help="Cosine shape component weight in PDOS loss.")
    parser.add_argument("--no_couple_heads", action="store_false", dest="couple_heads", help="Do not couple the eigenvalue head to the weight head.")
    parser.add_argument("--detach_coupling", action="store_true", help="Detach the predicted eigenvalues before feeding them to the weight head to prevent weight gradients from flowing back to the eigenvalue head.")
    parser.add_argument("--use_force_residual", action="store_true", help="Use ForceResidualHead to learn domain-specific force corrections.")
    parser.add_argument("--force_huber_delta", default=0.1, type=float, help="Huber delta for force loss.")

    args, _ = parser.parse_known_args()
    run(args)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    main()