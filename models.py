import torch
import torch.nn as nn
from utils import scatter_mean, scatter_sum

NUM_BANDS = 250

class ResidualBlock(nn.Module):
    """Simple linear-residual block with layer normalisation, SiLU activation, and dropout."""
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
    """Softmax attention pooling to aggregate node representations into graph-level features.

    Uses per-graph softmax normalisation so that attention weights sum to 1
    within each graph, producing size-invariant graph representations.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, 1)  # raw logits (no activation)

    def forward(self, x: torch.Tensor, graph_idx: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)  # [N_nodes, 1]

        # Numerically stable per-graph softmax
        num_graphs = int(graph_idx.max().item()) + 1
        max_logits = logits.new_full((num_graphs, 1), float("-inf"))
        max_logits.scatter_reduce_(0, graph_idx.unsqueeze(1), logits, reduce="amax")
        logits = logits - max_logits[graph_idx]  # shift for stability

        exp_logits = logits.exp()
        sum_exp = exp_logits.new_zeros((num_graphs, 1))
        sum_exp.index_add_(0, graph_idx, exp_logits)
        attn_weights = exp_logits / sum_exp[graph_idx].clamp(min=1e-8)  # [N_nodes, 1]

        return scatter_sum(x * attn_weights, graph_idx, dim=0)


class ForceResidualHead(nn.Module):
    """Learns a domain-specific correction to pretrained force predictions.

    Zero-initialized so initial output ≈ 0 — training starts from the
    pretrained Orb baseline and the head only learns the *residual* error.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, 3),
        )
        # Zero-init final layer so initial predictions ≈ pretrained
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Return per-atom force corrections [N_atoms, 3]."""
        return self.mlp(node_features)


class WeightHead(nn.Module):
    """PDOS Weight Prediction Head with Group Normalisation.

    Applies LayerNorm to node-level GNN features and eigenvalues separately
    before concatenating and processing through an MLP.  Uses two residual
    blocks and a pre-output LayerNorm for extra capacity and stable gradients.
    """
    def __init__(
        self,
        latent_dim: int,
        num_bands: int,
        hidden_dim: int,
        dropout: float = 0.0,
        couple_heads: bool = False,
    ) -> None:
        super().__init__()
        self.couple_heads = couple_heads
        self.node_norm = nn.LayerNorm(latent_dim)

        if couple_heads:
            self.eig_norm = nn.LayerNorm(num_bands)
            in_dim = latent_dim + num_bands
        else:
            in_dim = latent_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),   # extra capacity
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.LayerNorm(hidden_dim),             # stable pre-output norm
            nn.Linear(hidden_dim, num_bands),
            nn.Sigmoid(),
        )

        # Initialize the pre-sigmoid linear bias so outputs start near 0.3
        # (middle-low range) — avoids saturation at extremes.
        nn.init.constant_(self.mlp[-2].bias, -1.0)

    def forward(self, node_features: torch.Tensor, node_eigenvalues: torch.Tensor | None = None) -> torch.Tensor:
        x = self.node_norm(node_features)
        if self.couple_heads and node_eigenvalues is not None:
            eig_normed = self.eig_norm(node_eigenvalues)
            x = torch.cat([x, eig_normed], dim=-1)
        return self.mlp(x)


def build_heads(
    latent_dim: int,
    device: torch.device,
    dropout: float = 0.0,
    couple_heads: bool = False,
    use_force_residual: bool = False,
) -> tuple[nn.Module, WeightHead, AttentionPool, ForceResidualHead | None]:
    """Helper to instantiate prediction heads and place them on device.

    Returns ``(eigenvalue_head, weight_head, attention_pool, force_residual_head)``.
    ``force_residual_head`` is ``None`` when *use_force_residual* is ``False``.
    """
    hidden_dim = 1024

    eigenvalue_head = nn.Sequential(
        nn.LayerNorm(latent_dim),
        nn.Linear(latent_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        ResidualBlock(hidden_dim, dropout),
        nn.Linear(hidden_dim, NUM_BANDS),
    ).to(device)

    weight_head = WeightHead(
        latent_dim=latent_dim,
        num_bands=NUM_BANDS,
        hidden_dim=hidden_dim,
        dropout=dropout,
        couple_heads=couple_heads,
    ).to(device)

    attention_pool = AttentionPool(latent_dim).to(device)

    force_residual_head: ForceResidualHead | None = None
    if use_force_residual:
        force_residual_head = ForceResidualHead(
            latent_dim, hidden_dim=256, dropout=dropout,
        ).to(device)

    return eigenvalue_head, weight_head, attention_pool, force_residual_head
