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
    """Generates random train/val split indices given a dataset size and validation fraction."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(total_size).tolist()
    n_val = max(1, int(total_size * val_fraction))
    val_indices = sorted(indices[:n_val])
    train_indices = sorted(indices[n_val:])
    logging.info(
        f"Train/val split: {len(train_indices)} train, {len(val_indices)} val "
        f"({val_fraction:.0%} val, seed={seed})"
    )
    return train_indices, val_indices
