import torch
import torch.nn as nn

def init_linear_weights(weights: nn.Parameter):
    out_features, in_features = weights.shape
    std = 2 / (out_features + in_features)
    nn.init.trunc_normal_(weights, 0, std, a=-3.0 * std, b=3.0 * std)

def init_embedding_weights(embedding_weights: nn.Parameter):
    nn.init.trunc_normal_(embedding_weights, 0, 1, a=-3.0, b=3.0)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def stable_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shift_x = x - torch.max(x, dim=dim, keepdim=True)[0]
    return shift_x.exp() / shift_x.exp().sum(dim=dim, keepdim=True)