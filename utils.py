import logging
from typing import Any
import numpy as np
import torch

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Computes the sum of all elements in the src tensor grouped by index.

    Equivalent to torch_scatter.scatter_add.
    """
    dim_size = int(index.max().item()) + 1
    out = src.new_zeros((dim_size, src.size(1)))
    out.index_add_(dim, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Computes the mean values of all elements in the src tensor grouped by index.

    Equivalent to torch_scatter.scatter_mean.  Uses integer counts for
    numerical stability on large graphs.
    """
    dim_size = int(index.max().item()) + 1
    out = src.new_zeros((dim_size, src.size(1)))
    out.index_add_(dim, index, src)
    # Integer counts avoid float rounding drift
    count = torch.zeros(dim_size, dtype=torch.long, device=src.device)
    count.scatter_add_(0, index, torch.ones_like(index, dtype=torch.long))
    return out / count.unsqueeze(1).clamp(min=1).to(out.dtype)


def prefix_keys(dict_to_prefix: dict[str, Any], prefix: str, sep: str = "/") -> dict[str, Any]:
    """Helper function to prefix dict keys with a string."""
    return {f"{prefix}{sep}{k}": v for k, v in dict_to_prefix.items()}


def build_graph_index(n_node: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Builds a graph index tensor from node counts.

    Example: [2, 3] -> [0, 0, 1, 1, 1]
    """
    return torch.repeat_interleave(
        torch.arange(len(n_node), device=device), n_node
    )


def split_train_val(
    total_size: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Generates a random block train/val split of size val_fraction to prevent leakage."""
    rng = np.random.RandomState(seed)
    n_val = max(1, int(total_size * val_fraction))
    
    if total_size > n_val:
        start_idx = rng.randint(0, total_size - n_val + 1)
    else:
        start_idx = 0
        n_val = total_size

    val_indices = list(range(start_idx, start_idx + n_val))
    train_indices = [i for i in range(total_size) if i not in val_indices]
    
    logging.info(
        f"Random block train/val split (validation block [{start_idx}, {start_idx + n_val})): "
        f"{len(train_indices)} train, {len(val_indices)} val "
        f"({val_fraction:.0%} val, seed={seed})"
    )
    return train_indices, val_indices

