import torch
import torch.nn as nn
from cs336_basics.utils import init_linear_weights, silu

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.weights = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        init_linear_weights(self.weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T

class Swiglu(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super(Swiglu, self).__init__()
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))

        init_linear_weights(self.w1)
        init_linear_weights(self.w2)
        init_linear_weights(self.w3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T