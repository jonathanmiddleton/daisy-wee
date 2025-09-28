import torch

class KVCache:
    def __init__(self, L, B, H, D, W, device, dtype):
        self.k = torch.empty(L, B, H, W, D, device=device, dtype=dtype)
        self.v = torch.empty(L, B, H, W, D, device=device, dtype=dtype)
        self.W = W
        self.t = 0

    def reset(self):
        self.t = 0

    def view(self, layer):
        n = min(self.t, self.W)
        if n == 0:
            return self.k[layer, :, :, :0], self.v[layer, :, :, :0]
        if n < self.W:
            return self.k[layer, :, :, :n], self.v[layer, :, :, :n]
        s = self.t % self.W
        k1, k2 = self.k[layer, :, :, s:], self.k[layer, :, :, :s]
        v1, v2 = self.v[layer, :, :, s:], self.v[layer, :, :, :s]
        return torch.cat([k1, k2], 3), torch.cat([v1, v2], 3)

    def write(self, layer, k_new, v_new):
        i = self.t % self.W
        self.k[layer, :, :, i:i+1] = k_new
        self.v[layer, :, :, i:i+1] = v_new

    def advance(self):
        self.t += 1
