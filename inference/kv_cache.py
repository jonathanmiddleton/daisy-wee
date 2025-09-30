import torch

class KVCache:
    def __init__(self, L, B, H, W, D, device, dtype):
        self.k = torch.zeros(L, B, H, W, D, device=device, dtype=dtype)
        self.v = torch.zeros(L, B, H, W, D, device=device, dtype=dtype)
        self.W = W
        self.t = 0

    def reset(self):
        self.t = 0

    def view(self, layer):
        n = min(self.t, self.W)
        if n == 0:
            z = self.k[layer, :, :, :0]
            return z.transpose(1, 2), z.transpose(1, 2)
        start = (self.t - n) % self.W
        end = start + n
        if end <= self.W:
            k = self.k[layer, :, :, start:end]
            v = self.v[layer, :, :, start:end]
        else:
            k = torch.cat([self.k[layer, :, :, start:], self.k[layer, :, :, :end - self.W]], dim=2)
            v = torch.cat([self.v[layer, :, :, start:], self.v[layer, :, :, :end - self.W]], dim=2)
        return k.transpose(1, 2), v.transpose(1, 2)

    def write(self, layer, k_new, v_new):
        if k_new.size(1) == 1:
            k_new = k_new.transpose(1, 2)
            v_new = v_new.transpose(1, 2)
        i = self.t % self.W
        self.k[layer, :, :, i:i+1] = k_new
        self.v[layer, :, :, i:i+1] = v_new

    def advance(self):
        self.t += 1
