import torch

class KVCache:
    def __init__(self, L, B, H, W, D, device, dtype):
        self.k = torch.empty(L, B, H, W, D, device=device, dtype=dtype)
        self.v = torch.empty(L, B, H, W, D, device=device, dtype=dtype)
        self.W = W
        self.t = 0

    def reset(self):
        self.t = 0

    def view(self, layer):
        n = min(self.t, self.W)
        if n == 0:
            return self.k[layer, :, :, :0].transpose(1, 2), self.v[layer, :, :, :0].transpose(1, 2)
        if n < self.W:
            return self.k[layer, :, :, :n].transpose(1, 2), self.v[layer, :, :, :n].transpose(1, 2)
        s = self.t % self.W
        k1, k2 = self.k[layer, :, :, s:], self.k[layer, :, :, :s]
        v1, v2 = self.v[layer, :, :, s:], self.v[layer, :, :, :s]
        return torch.cat([k1, k2], 2).transpose(1, 2), torch.cat([v1, v2], 2).transpose(1, 2)

    def write(self, layer, k_new, v_new):
        if k_new.size(1) == 1:
            k_new = k_new.transpose(1, 2)
            v_new = v_new.transpose(1, 2)
        i = self.t % self.W
        self.k[layer, :, :, i:i+1] = k_new
        self.v[layer, :, :, i:i+1] = v_new

    def advance(self):
        self.t += 1
