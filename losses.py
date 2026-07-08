import torch
import torch.nn.functional as F


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
    active_threshold: float = 0.1,
    magnitude_weight: float = 3.0,
    cramer_weight: float = 0.5,
    cosine_weight: float = 0.3,
    r2_weight: float = 1.0,
    deriv_weight: float = 2.0,
    peak_scaling: str = "linear",
    cramer_scale: float = 100.0,
    magnitude_loss_type: str = "log_cosh",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes training losses for electronic structure (eigenvalues & PDOS weights)."""
    # 1. Eigenvalue loss (Huber or MSE)
    if eig_loss_type == "huber":
        eig_loss = F.huber_loss(
            pred_eigenvalues, true_eigenvalues, delta=huber_delta
        )
    else:
        eig_loss = F.mse_loss(pred_eigenvalues, true_eigenvalues)

    # 2. PDOS weights loss
    weight_loss = compute_weight_loss(
        pred_weights, true_weights, device,
        peak_boost=peak_boost,
        active_threshold=active_threshold,
        magnitude_weight=magnitude_weight,
        cramer_weight=cramer_weight,
        cosine_weight=cosine_weight,
        r2_weight=r2_weight,
        deriv_weight=deriv_weight,
        peak_scaling=peak_scaling,
        cramer_scale=cramer_scale,
        magnitude_loss_type=magnitude_loss_type,
    )

    return eig_loss, weight_loss


def compute_weight_loss(
    pred_weights: torch.Tensor,
    true_weights: torch.Tensor,
    device: torch.device,
    *,
    peak_boost: float = 5.0,
    active_threshold: float = 0.1,
    magnitude_weight: float = 3.0,
    cramer_weight: float = 0.5,
    cosine_weight: float = 0.3,
    r2_weight: float = 1.0,
    deriv_weight: float = 2.0,
    peak_scaling: str = "linear",
    cramer_scale: float = 100.0,
    magnitude_loss_type: str = "log_cosh",
) -> torch.Tensor:
    """Combined magnitude + shape + R2 + derivative loss for PDOS weights.

    Five complementary terms:
      1. **Log-cosh magnitude loss** — smooth Huber-like, with configurable
         peak scaling (sqrt, linear, or quadratic) to boost peak gradients.
      2. **Masked Cramér shape loss** — L2 Wasserstein on cumulative densities.
      3. **Cosine similarity shape loss** — global shape alignment.
      4. **Variance-normalised R² loss** — scale-invariant MSE representation.
      5. **Spectral derivative loss** — matches peak slopes and sharpness.
    """
    diff = pred_weights - true_weights

    # 1. Magnitude loss with configurable type and peak emphasis
    if magnitude_loss_type == "mse":
        base_mag_loss = diff.pow(2)
    elif magnitude_loss_type == "huber":
        base_mag_loss = F.huber_loss(pred_weights, true_weights, delta=0.1, reduction="none")
    else:  # "log_cosh"
        base_mag_loss = torch.log(torch.cosh(diff.clamp(-20, 20)) + 1e-8)
    
    if peak_scaling == "sqrt":
        peak_mask = 1.0 + torch.sqrt(true_weights.clamp(min=0.0) * peak_boost)
    elif peak_scaling == "quadratic":
        peak_mask = 1.0 + (true_weights.clamp(min=0.0) * peak_boost) ** 2
    else:  # "linear"
        peak_mask = 1.0 + (true_weights.clamp(min=0.0) * peak_boost)
        
    magnitude_loss = torch.mean(base_mag_loss * peak_mask)

    # 2. Masked Cramér shape loss with configurable scaling
    true_sums = true_weights.sum(dim=-1, keepdim=True)
    active_mask = (true_sums > active_threshold).squeeze(-1)

    if active_mask.any():
        pred_weights_active = pred_weights[active_mask]
        true_weights_active = true_weights[active_mask]

        pred_pdf = pred_weights_active / (pred_weights_active.sum(dim=-1, keepdim=True) + 1e-8)
        true_pdf = true_weights_active / (true_weights_active.sum(dim=-1, keepdim=True) + 1e-8)
        pred_cdf = torch.cumsum(pred_pdf, dim=-1)
        true_cdf = torch.cumsum(true_pdf, dim=-1)

        cramer_loss = torch.mean((pred_cdf - true_cdf) ** 2) * cramer_scale
    else:
        cramer_loss = torch.tensor(0.0, device=device)

    # 3. Cosine similarity shape loss (per-sample, no normalisation needed)
    cos_sim = F.cosine_similarity(pred_weights, true_weights, dim=-1)
    shape_loss = (1.0 - cos_sim).mean()

    # 4. R² Loss (variance-normalized MSE)
    var_true = torch.var(true_weights).clamp(min=1e-6)
    r2_loss = torch.mean(diff.pow(2)) / var_true

    # 5. Spectral derivative loss for sharp peak matching
    pred_diff = pred_weights[..., 1:] - pred_weights[..., :-1]
    true_diff = true_weights[..., 1:] - true_weights[..., :-1]
    deriv_diff = pred_diff - true_diff
    # Use mean peak mask of adjacent bins
    deriv_peak_mask = 0.5 * (peak_mask[..., 1:] + peak_mask[..., :-1])
    deriv_loss = torch.mean(deriv_diff.pow(2) * deriv_peak_mask)

    return (
        magnitude_weight * magnitude_loss
        + cramer_weight * cramer_loss
        + cosine_weight * shape_loss
        + r2_weight * r2_loss
        + deriv_weight * deriv_loss
    )


def compute_force_loss(
    pred_forces: torch.Tensor,
    true_forces: torch.Tensor,
    std_forces: torch.Tensor | None = None,
    huber_delta: float = 0.1,
    force_loss_type: str = "mse",
) -> tuple[torch.Tensor, dict[str, float]]:
    """MSE or Huber loss on forces with optional normalisation and per-step diagnostics.

    Returns ``(loss, diagnostics_dict)`` where diagnostics are detached scalars.
    """
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

    # Per-step diagnostics (detached, no grad overhead)
    with torch.no_grad():
        errors = pred_forces - true_forces
        mae = errors.abs().mean().item()
        rmse = errors.pow(2).mean().sqrt().item()
        max_err = errors.abs().max().item()

    diagnostics = {
        "force_mae": mae,
        "force_rmse": rmse,
        "force_max_err": max_err,
    }

    return loss, diagnostics
