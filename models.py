import torch
import torch.nn as nn
from utils import scatter_mean

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
    """Gated attention pooling to aggregate node representations into graph-level features."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, graph_idx: torch.Tensor) -> torch.Tensor:
        weights = self.gate(x)  # [N_nodes, 1]
        return scatter_mean(x * weights, graph_idx, dim=0)


class WeightHead(nn.Module):
    """PDOS Weight Prediction Head with Group Normalisation.

    Applies LayerNorm to node-level GNN features and eigenvalues separately
    before concatenating and processing through an MLP.
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
            nn.Linear(hidden_dim, num_bands),
            nn.Sigmoid(),
        )

        # Standard initialization to prevent zero weight collapse on sigmoids
        nn.init.constant_(self.mlp[-2].bias, -4.5)

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
) -> tuple[nn.Module, WeightHead, AttentionPool]:
    """Helper to instantiate all three prediction heads and place them on device."""
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

    return eigenvalue_head, weight_head, attention_pool
