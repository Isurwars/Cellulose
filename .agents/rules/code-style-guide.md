---
trigger: always_on
---

# Rule: Python & PyTorch Code Style, Type Safety, and Standards

*Activation Mode: Glob (`**/*.py`)*

## 1. Tooling Compliance & Linting
- **PEP 8 Alignment:** All Python code must strictly align with PEP 8 and workspace `ruff` / `black` configuration.
- **Verification Gate:** Verify modified code using `python -m py_compile` or `ruff check` prior to task completion.
- **Explicit Imports:** Avoid wildcard imports (`from module import *`). Use explicit relative or absolute imports.
- **Zero Silent Exception Swallowing:** Never use bare `except:` or `except Exception: pass` without logging or explicit error handling logic.

## 2. Docstring & Type Annotation Standards
- **Scope:** Every module, `class`, method, and public function must include type annotations and structured docstrings (Google or NumPy style).
- **Format:** Annotate tensor shapes and device requirements in docstrings where applicable.

### Example
```python
import torch
import torch.nn as nn

class AttentionPool(nn.Module):
    """Softmax attention pooling to aggregate node representations into graph-level features.

    Uses per-graph softmax normalisation so that attention weights sum to 1
    within each graph, producing size-invariant graph representations.

    Args:
        dim (int): Feature dimension of input node embeddings.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, graph_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass for attention pooling.

        Args:
            x (torch.Tensor): Node embeddings of shape [N_nodes, dim].
            graph_idx (torch.Tensor): Graph indices of shape [N_nodes].

        Returns:
            torch.Tensor: Pooled graph representations of shape [N_graphs, dim].

        Raises:
            ValueError: If x and graph_idx size mismatch.
        """
        if x.size(0) != graph_idx.size(0):
            raise ValueError("Mismatched node and graph_idx dimensions")
        # ... logic ...
```

## Reference

- **See skill:** `python-pytorch-standards` for comprehensive PyTorch coding standards (Vectorization, CUDA memory management, autograd safety, model modularity, and loss evaluation patterns).
