import torch
from torch import Tensor
import torch.nn.functional as F

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

@torch.no_grad()
def init_linear(w: Tensor):
    std = 0.5 * (w.size(-1) ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
    bound = (3 ** 0.5) * std
    return w.uniform_(-bound, bound)