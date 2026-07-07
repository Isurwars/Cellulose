import torch

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
    """Computes training losses for electronic structure (eigenvalues & PDOS weights)."""
    # 1. Eigenvalue loss (Huber or MSE)
    if eig_loss_type == "huber":
        eig_loss = torch.nn.functional.huber_loss(
            pred_eigenvalues, true_eigenvalues, delta=huber_delta
        )
    else:
        eig_loss = torch.nn.functional.mse_loss(pred_eigenvalues, true_eigenvalues)

    # 2. PDOS weights loss
    # 2a. Peak-weighted MSE (focuses on matching regions with high PDOS weight)
    squared_errors = (pred_weights - true_weights) ** 2
    peak_multiplier = 1.0 + (true_weights * peak_boost)
    magnitude_loss = torch.mean(squared_errors * peak_multiplier)

    # 2b. Shape alignment via Masked Cramér (L2 Wasserstein distance on cumulative densities)
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
