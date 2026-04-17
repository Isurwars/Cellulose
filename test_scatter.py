import torch
from torch import Tensor

def scatter_mean(src: Tensor, index: Tensor, dim: int = 0) -> Tensor:
    dim_size = int(index.max().item()) + 1
    out = src.new_zeros((dim_size, src.size(1)))
    out.index_add_(dim, index, src)
    count = src.new_zeros((dim_size, src.size(1)))
    count.index_add_(dim, index, torch.ones_like(src))
    return out / count.clamp(min=1)

src = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
index = torch.tensor([0, 0, 1])

res = scatter_mean(src, index, dim=0)
print(res)
