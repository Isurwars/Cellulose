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
) -> torch.Tensor:
    """Combined magnitude + shape loss for PDOS weights.

    Three complementary terms:
      1. **Log-cosh magnitude loss** — smooth Huber-like, no gradient explosion
         on large errors, with sqrt-peak weighting (gentler than linear).
      2. **Masked Cramér shape loss** — L2 Wasserstein on cumulative densities
         for active samples (preserved from the original implementation).
      3. **Cosine similarity shape loss** — lightweight global shape alignment
         that doesn't require PDF normalisation.
    """
    diff = pred_weights - true_weights

    # 1. Log-cosh magnitude loss with sqrt-peak emphasis
    log_cosh = torch.log(torch.cosh(diff.clamp(-20, 20)) + 1e-8)
    peak_mask = 1.0 + torch.sqrt(true_weights.clamp(min=0.0) * peak_boost)
    magnitude_loss = torch.mean(log_cosh * peak_mask)

    # 2. Masked Cramér shape loss (preserved from original)
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

    # 3. Cosine similarity shape loss (per-sample, no normalisation needed)
    cos_sim = F.cosine_similarity(pred_weights, true_weights, dim=-1)
    shape_loss = (1.0 - cos_sim).mean()

    return (
        magnitude_weight * magnitude_loss
        + cramer_weight * cramer_loss
        + cosine_weight * shape_loss
    )


def compute_force_loss(
    pred_forces: torch.Tensor,
    true_forces: torch.Tensor,
    std_forces: torch.Tensor | None = None,
    huber_delta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Huber loss on forces with optional normalisation and per-step diagnostics.

    Returns ``(loss, diagnostics_dict)`` where diagnostics are detached scalars.
    """
    if std_forces is not None:
        pred_norm = pred_forces / std_forces
        true_norm = true_forces / std_forces
    else:
        pred_norm = pred_forces
        true_norm = true_forces

    loss = F.huber_loss(pred_norm, true_norm, delta=huber_delta)

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
