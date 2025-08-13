from math import sqrt
from typing import Optional
from cs336_basics.utils import stable_softmax
import torch
import torch.nn as nn


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    scaled_qk = torch.einsum(
        "...id, ...dj -> ...ij", queries, keys.transpose(-2, -1)
    ) / sqrt(keys.size(-1))
    if mask is not None:
        scaled_qk[..., ~mask] -= torch.inf
    softmax_scaled_qk = stable_softmax(scaled_qk, dim=-1)
    return torch.einsum("...ij, ...jd -> ...id", softmax_scaled_qk, values)
