import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.gain: nn.Parameter = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model: int = d_model
        self.eps: float = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.square().sum(dim=-1, keepdim=True) / self.d_model + self.eps)
        normed_x = x / rms * self.gain
        return normed_x.to(in_dtype)
