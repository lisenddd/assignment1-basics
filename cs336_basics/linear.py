import torch
import torch.nn as nn
from cs336_basics.utils import init_linear_weights

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.weights = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        init_linear_weights(self.weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T