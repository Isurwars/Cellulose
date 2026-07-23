import torch
import torch.nn.functional as F


def compute_r2(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Computes the R² coefficient of determination (unbiased variance normalized)."""
    with torch.no_grad():
        pred_flat = pred.detach().flatten()
        true_flat = true.detach().flatten()
        var_true = torch.var(true_flat, unbiased=False)
        if var_true < 1e-8:
            return 0.0
        mse = torch.mean((pred_flat - true_flat) ** 2)
        r2 = 1.0 - mse / var_true
        return float(r2.item())


def compute_force_loss(
    pred_forces: torch.Tensor,
    true_forces: torch.Tensor,
    std_forces: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes standard MSE loss and diagnostics (MAE, RMSE, Max Error, R²) for forces."""
    if std_forces is not None:
        pred_norm = pred_forces / std_forces
        true_norm = true_forces / std_forces
    else:
        pred_norm = pred_forces
        true_norm = true_forces

    loss = F.mse_loss(pred_norm, true_norm)

    with torch.no_grad():
        errors = pred_forces - true_forces
        mae = errors.abs().mean().item()
        rmse = errors.pow(2).mean().sqrt().item()
        max_err = errors.abs().max().item()
        r2 = compute_r2(pred_forces, true_forces)
        mse_val = F.mse_loss(pred_forces, true_forces).item()

    diagnostics = {
        "force_mae": mae,
        "force_rmse": rmse,
        "force_max_err": max_err,
        "force_mse": mse_val,
        "force_r2": r2,
    }
    return loss, diagnostics


def compute_energy_loss(
    pred_energy: torch.Tensor,
    true_energy: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes MSE loss and diagnostics (MAE, RMSE, R²) for energies."""
    loss = F.mse_loss(pred_energy, true_energy)
    with torch.no_grad():
        errors = pred_energy - true_energy
        mae = errors.abs().mean().item()
        rmse = errors.pow(2).mean().sqrt().item()
        r2 = compute_r2(pred_energy, true_energy)

    diagnostics = {
        "energy_mae": mae,
        "energy_rmse": rmse,
        "energy_mse": loss.item(),
        "energy_r2": r2,
    }
    return loss, diagnostics


def compute_stress_loss(
    pred_stress: torch.Tensor,
    true_stress: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes MSE loss and diagnostics (MAE, RMSE, R²) for stress."""
    loss = F.mse_loss(pred_stress, true_stress)
    with torch.no_grad():
        errors = pred_stress - true_stress
        mae = errors.abs().mean().item()
        rmse = errors.pow(2).mean().sqrt().item()
        r2 = compute_r2(pred_stress, true_stress)

    diagnostics = {
        "stress_mae": mae,
        "stress_rmse": rmse,
        "stress_mse": loss.item(),
        "stress_r2": r2,
    }
    return loss, diagnostics


def compute_eigenvalue_loss(
    pred_eigenvalues: torch.Tensor,
    true_eigenvalues: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes standard MSE loss and diagnostics (MAE, RMSE, R²) for eigenvalues."""
    loss = F.mse_loss(pred_eigenvalues, true_eigenvalues)

    with torch.no_grad():
        errors = pred_eigenvalues - true_eigenvalues
        mae = errors.abs().mean().item()
        rmse = errors.pow(2).mean().sqrt().item()
        r2 = compute_r2(pred_eigenvalues, true_eigenvalues)
        mse_val = F.mse_loss(pred_eigenvalues, true_eigenvalues).item()

    diagnostics = {
        "eigenvalues_mae": mae,
        "eigenvalues_rmse": rmse,
        "eigenvalues_mse": mse_val,
        "eigenvalues_r2": r2,
    }
    return loss, diagnostics


def compute_weight_loss(
    pred_weights: torch.Tensor,
    true_weights: torch.Tensor,
    device: torch.device,
    is_transformed: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes weight loss. Uses standard MSE loss for logit-transformed targets,
    and falls back to fractional Binary Cross Entropy (BCE) with logits otherwise.
    """
    if is_transformed is None:
        is_transformed = bool((true_weights.min() < -0.1) or (true_weights.max() > 1.1))

    if is_transformed:
        # Standard MSE in logit space with peak focus
        # true_weights > -6.0 corresponds to physical weights > 0.0025 (10x scaling for peaks)
        weight_mask = torch.where(true_weights > -6.0, 5.0, 1.0)
        loss = (weight_mask * (pred_weights - true_weights) ** 2).mean()
        with torch.no_grad():
            pred_weights_orig = torch.sigmoid(pred_weights)
            true_weights_orig = torch.sigmoid(true_weights)
            diff_orig = pred_weights_orig - true_weights_orig
            mae = diff_orig.abs().mean().item()
            rmse = diff_orig.pow(2).mean().sqrt().item()
            r2 = compute_r2(pred_weights_orig, true_weights_orig)
            mse_val = F.mse_loss(pred_weights_orig, true_weights_orig).item()
    else:
        loss = F.binary_cross_entropy_with_logits(pred_weights, true_weights)
        with torch.no_grad():
            pred_weights_orig = torch.sigmoid(pred_weights)
            diff_orig = pred_weights_orig - true_weights
            mae = diff_orig.abs().mean().item()
            rmse = diff_orig.pow(2).mean().sqrt().item()
            r2 = compute_r2(pred_weights_orig, true_weights)
            mse_val = F.mse_loss(pred_weights_orig, true_weights).item()

    diagnostics = {
        "weights_mae": mae,
        "weights_rmse": rmse,
        "weights_mse": mse_val,
        "weights_r2": r2,
    }
    return loss, diagnostics


def compute_electronic_losses(
    pred_eigenvalues: torch.Tensor,
    true_eigenvalues: torch.Tensor,
    pred_weights: torch.Tensor,
    true_weights: torch.Tensor,
    device: torch.device,
    is_transformed: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Orchestrates standard MSE loss computation for eigenvalues and logit-space PDOS weights."""
    eig_loss, eig_diag = compute_eigenvalue_loss(
        pred_eigenvalues, true_eigenvalues,
    )

    weight_loss, weight_diag = compute_weight_loss(
        pred_weights, true_weights, device, is_transformed=is_transformed, **kwargs
    )

    # Merge diagnostics
    diagnostics = {}
    diagnostics.update(eig_diag)
    diagnostics.update(weight_diag)

    return eig_loss, weight_loss, diagnostics
