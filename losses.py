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
    huber_delta: float = 0.1,
    force_loss_type: str = "mse",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes MSE (or Huber) loss and diagnostics (MAE, RMSE, Max Error, R²) for forces."""
    if std_forces is not None:
        pred_norm = pred_forces / std_forces
        true_norm = true_forces / std_forces
    else:
        pred_norm = pred_forces
        true_norm = true_forces

    if force_loss_type == "huber":
        loss = F.huber_loss(pred_norm, true_norm, delta=huber_delta)
    else:
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
    eig_loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes MSE (or Huber) loss and diagnostics (MAE, RMSE, R²) for eigenvalues."""
    if eig_loss_type == "huber":
        loss = F.huber_loss(pred_eigenvalues, true_eigenvalues, delta=huber_delta)
    else:
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
    *,
    peak_boost: float = 5.0,
    active_threshold: float = 0.05,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Simplified PDOS weight loss using a peak-searching (peak-weighted) loss."""
    # Find peak locations in true_weights (local maxima)
    # Pad to check boundary conditions
    inner_is_peak = (true_weights[..., 1:-1] > true_weights[..., :-2]) & (true_weights[..., 1:-1] > true_weights[..., 2:])
    is_peak = F.pad(inner_is_peak, (1, 1), mode="constant", value=False)

    # Filter by threshold to ignore noise peaks
    is_peak = is_peak & (true_weights > active_threshold)

    # Peak-weighted MSE: boost the error weight at peak positions
    weights = torch.ones_like(true_weights)
    weights[is_peak] = 1.0 + peak_boost

    diff = pred_weights - true_weights
    loss = torch.mean(weights * (diff ** 2))

    with torch.no_grad():
        mae = diff.abs().mean().item()
        rmse = diff.pow(2).mean().sqrt().item()
        r2 = compute_r2(pred_weights, true_weights)
        mse_val = F.mse_loss(pred_weights, true_weights).item()

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
    *,
    eig_loss_type: str = "mse",
    huber_delta: float = 1.0,
    peak_boost: float = 5.0,
    active_threshold: float = 0.05,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Orchestrates simplified eigenvalue and peak-searching PDOS weight loss."""
    eig_loss, eig_diag = compute_eigenvalue_loss(
        pred_eigenvalues, true_eigenvalues,
        eig_loss_type=eig_loss_type,
        huber_delta=huber_delta,
    )

    weight_loss, weight_diag = compute_weight_loss(
        pred_weights, true_weights, device,
        peak_boost=peak_boost,
        active_threshold=active_threshold,
    )

    # Merge diagnostics
    diagnostics = {}
    diagnostics.update(eig_diag)
    diagnostics.update(weight_diag)

    return eig_loss, weight_loss, diagnostics
