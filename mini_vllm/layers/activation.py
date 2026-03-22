import torch
from torch import nn
import torch.nn.functional as F

from mini_vllm.utils.compile import safe_compile


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @safe_compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
