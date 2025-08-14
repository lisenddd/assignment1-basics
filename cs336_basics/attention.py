from math import sqrt
from typing import Optional
from cs336_basics.embedding import RotaryPositionalEmbedding
from cs336_basics.linear import Linear
from cs336_basics.utils import stable_softmax
import torch
import torch.nn as nn
from einops import rearrange


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


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, theta: int = 0, max_seq_len: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.rope = (
            RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
            if theta != 0 and max_seq_len != 0
            else None
        )
        self.q_proj_weight = Linear(d_model, d_model)
        self.k_proj_weight = Linear(d_model, d_model)
        self.v_proj_weight = Linear(d_model, d_model)
        self.o_proj_weight = Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        qkv_proj_weight = torch.cat(
            [
                self.q_proj_weight.weights,
                self.k_proj_weight.weights,
                self.v_proj_weight.weights,
            ]
        )  # [3 * d_model, d_model]
        qkv_proj = x @ qkv_proj_weight.T
        q, k, v = torch.chunk(qkv_proj, 3, -1)

        q = rearrange(
            q, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        k = rearrange(
            k, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        v = rearrange(
            v, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq_len = x.size(dim=-2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        attention = scaled_dot_product_attention(q, k, v, causal_mask)
        attention = rearrange(
            attention, "... h seq_len d_head -> ... seq_len (h d_head)"
        )
        return self.o_proj_weight(attention)
