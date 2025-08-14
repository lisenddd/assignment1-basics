import torch
import torch.nn as nn
from cs336_basics.utils import init_embedding_weights


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.embedding_weights = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        init_embedding_weights(self.embedding_weights)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_weights[token_ids]


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        frequency = 1.0 / torch.pow(
            torch.tensor(theta), torch.arange(0, d_k, 2).float() / d_k
        )
        positions = torch.arange(max_seq_len)
        angles = torch.outer(positions, frequency)
        self.register_buffer(
            "cos_cache", torch.cos(angles).to(device), persistent=False
        )
        self.register_buffer(
            "sin_cache", torch.sin(angles).to(device), persistent=False
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cache[token_positions].unsqueeze(0)
        sin = self.sin_cache[token_positions].unsqueeze(0)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        output1 = x1 * cos - x2 * sin
        output2 = x2 * cos + x1 * sin
        return torch.stack((output1, output2), dim=-1).flatten(-2)
