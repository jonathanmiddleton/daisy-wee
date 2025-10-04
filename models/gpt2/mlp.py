import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.gpt2.functional import init_linear


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc_w = nn.Parameter(init_linear(torch.empty(hdim, dim)).bfloat16())
        self.proj_w = nn.Parameter(torch.zeros(dim, hdim).bfloat16())
        self.fc_w.wd_mul = 2.0
        self.proj_w.wd_mul = 2.0

    def forward(self, x: Tensor):
        x = F.linear(x, self.fc_w)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.linear(x, self.proj_w)
        return x

