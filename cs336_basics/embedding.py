import torch
import torch.nn as nn
from cs336_basics.utils import init_embedding_weights

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.embedding_weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        init_embedding_weights(self.embedding_weights)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_weights[token_ids]