---
name: python-pytorch-standards
description: Python and PyTorch coding standards for deep learning, GNN embeddings, and atomic property prediction. Use when writing, reviewing, or refactoring Python/PyTorch code.
metadata:
  origin: ECC
triggers:
  - on_artifact_generation
  - on_code_modification
---

# Python & PyTorch Coding Standards

Comprehensive coding standards for modern Python (3.10+) and PyTorch (2.x) deep learning applications, GNN graph pooling, and atomic property analysis. Enforces performance, memory efficiency, tensor safety, and vectorization.

## When to Use

- Writing new PyTorch neural network modules, loss functions, datasets, or trainers
- Reviewing or refactoring existing Python / PyTorch code
- Optimizing GPU memory usage, CUDA tensor operations, and data pipelines
- Ensuring numerical stability in custom loss functions or attention pooling layers

## Cross-Cutting Principles

1. **Vectorization everywhere**: Eliminate explicit Python `for` loops over batch nodes or graphs; use `torch` tensor operations or `torch_scatter`.
2. **Explicit Device & Type Contracts**: Always match tensor devices (`.device`), floating-point dtypes (`torch.float32`), and check index ranges explicitly.
3. **Graph Detachment for Metrics**: Detach tensors when computing validation metrics or tracking loss (`loss.item()`, `output.detach()`) to prevent CUDA graph memory leaks.
4. **Numerical Stability by Default**: Avoid unchecked `log()` or `exp()`; use `log_softmax`, max-shifting tricks, or `torch.clamp` to prevent NaNs.
5. **Modularity & Single Responsibility**: Separate model architecture (`models.py`), data loading (`data.py`), loss functions (`losses.py`), and execution logic (`trainer.py`).

---

## 1. PyTorch Module Architecture (`nn.Module`)

### Rules
- Call `super().__init__()` in every `nn.Module` subclass.
- Initialize submodules inside `__init__`, never dynamically inside `forward`.
- Specify explicit type hints for `forward` parameters and return types.

```python
# DO
class ResidualBlock(nn.Module):
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
```

---

## 2. Graph Aggregations & Numerical Stability

### Rules
- In custom graph attention or pooling layers, shift logits by `max_logits` prior to exponentiation to prevent overflow/underflow.
- Use `scatter_reduce_` or `torch-scatter` helpers (`scatter_mean`, `scatter_sum`) for batch graph aggregations.

```python
# DO: Numerically stable graph softmax attention
def forward(self, x: torch.Tensor, graph_idx: torch.Tensor) -> torch.Tensor:
    logits = self.gate(x)  # [N_nodes, 1]
    num_graphs = int(graph_idx.max().item()) + 1
    
    max_logits = logits.new_full((num_graphs, 1), float("-inf"))
    max_logits.scatter_reduce_(0, graph_idx.unsqueeze(1), logits, reduce="amax")
    logits = logits - max_logits[graph_idx]  # numerical stability shift
    
    weights = logits.exp()
    sum_weights = weights.new_zeros((num_graphs, 1))
    sum_weights.scatter_add_(0, graph_idx.unsqueeze(1), weights)
    
    norm_weights = weights / (sum_weights[graph_idx] + 1e-12)
    return scatter_sum(x * norm_weights, graph_idx, dim=0)
```

---

## 3. Device & Memory Management

### Rules
- Always transfer inputs explicitly using `.to(device, non_blocking=True)`.
- Use `torch.no_grad()` or `@torch.inference_mode()` during evaluation and inference loops.
- Call `torch.cuda.empty_cache()` if encountering memory fragmentation in long evaluation scripts.
- Never append un-detached loss tensors to history lists: `history.append(loss.item())`.

```python
# DON'T: Memory leak via graph retention
losses = []
for data in dataloader:
    out = model(data)
    loss = criterion(out, target)
    losses.append(loss)  # LEAK! Retains entire computation graph in memory

# DO: Detach scalar item
losses.append(loss.item())
```

---

## 4. Training Loop & Optimizer Best Practices

### Rules
- Zero gradients with `optimizer.zero_grad(set_to_none=True)` for faster execution and lower memory overhead.
- Apply gradient clipping (`torch.nn.utils.clip_grad_norm_`) when training deep interatomic or graph networks.
- Wrap model evaluation in `model.eval()` to freeze Dropout and LayerNorm behavior.

```python
model.train()
for batch in train_loader:
    optimizer.zero_grad(set_to_none=True)
    pred = model(batch.x, batch.graph_idx)
    loss = criterion(pred, batch.y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

---

## 5. Quick Reference Checklist

Before marking Python/PyTorch work complete:

- [ ] All tensors moved to target `device` explicitly
- [ ] No explicit Python loops over graph nodes/batch items (vectorized)
- [ ] `loss.item()` or `.detach()` used when recording metrics
- [ ] `optimizer.zero_grad(set_to_none=True)` used
- [ ] `model.eval()` and `torch.no_grad()` active during validation
- [ ] Function arguments and returns fully type-annotated
- [ ] Shape assertions in place for ambiguous tensor transformations
- [ ] Code formatted according to PEP 8 / Ruff style
