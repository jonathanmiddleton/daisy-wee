import torch
import torch.nn.functional as F
from inference.kv_cache import KVCache

def _apply_repetition_penalty(logits, prev_ids, rep_p, rep_w=128, rep_h=140.0, cap=3.0):
    if rep_p == 1.0 or prev_ids is None or prev_ids.numel() == 0:
        return logits
    prev = prev_ids[-rep_w:]
    V = logits.size(-1)
    d = torch.arange(prev.numel(), 0, -1, device=logits.device, dtype=logits.dtype)
    w = (0.5 ** (d / rep_h))
    s = torch.bincount(prev, weights=w, minlength=V).to(logits.dtype).clamp_max_(cap)
    scale = rep_p ** s
    return torch.where(logits > 0, logits / scale, logits * scale)


class Generator:
    def __init__(self, model, window, device=None, dtype=torch.bfloat16, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, eos_token_id=None):
        self.model = model.eval()
        self.device = next(model.parameters()).device if device is None else device
        assert self.device is not None
        h = None; d = None
        for b in self.model.blocks:
            a = getattr(b, "attn", None)
            if a is not None:
                h = a.num_heads; d = a.head_dim; break
        self.cache = KVCache(L=len(self.model.blocks), B=1, H=h, W=window, D=d, device=self.device, dtype=dtype)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.eos_token_id = eos_token_id
        self.history = torch.empty(0, dtype=torch.long, device=self.device)
        self.window = window

    @torch.no_grad()
    def reset(self):
        self.cache.reset()
        self.history = torch.empty(0, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def prefill(self, prompt_ids):
        # TODO horrible tmp method, replace with full-sequence pass
        self.reset()
        if isinstance(prompt_ids, torch.Tensor):
            ids = prompt_ids.tolist()
        else:
            ids = list(prompt_ids)
        logits = None
        for tok in ids:
            k_ctxs, v_ctxs = [], []
            for i in range(len(self.model.blocks)):
                kc, vc = self.cache.view(i)
                k_ctxs.append(kc); v_ctxs.append(vc)
            token = torch.tensor(tok, device=self.device, dtype=torch.long)
            logits, k_new, v_new = self.model.step(token, k_ctxs, v_ctxs, self.cache.t, self.window)
            for i in range(len(self.model.blocks)):
                if k_new[i] is not None:
                    self.cache.write(i, k_new[i], v_new[i])
            self.cache.advance()
            self.history = torch.cat([self.history, token.view(1)], dim=0)
        return logits

    @torch.no_grad()
    def step(self, token_id):
        k_ctxs, v_ctxs = [], []
        for i in range(len(self.model.blocks)):
            kc, vc = self.cache.view(i)
            k_ctxs.append(kc); v_ctxs.append(vc)
        token = torch.tensor(token_id, device=self.device, dtype=torch.long)
        logits, k_new, v_new = self.model.step(token, k_ctxs, v_ctxs, self.cache.t, self.window)
        for i in range(len(self.model.blocks)):
            if k_new[i] is not None:
                self.cache.write(i, k_new[i], v_new[i])
        self.cache.advance()
        self.history = torch.cat([self.history, token.view(1)], dim=0)
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens):
        assert prompt_ids.ndim == 1
        assert prompt_ids.size(0) > 0 # must have at least one token
        logits = self.prefill(prompt_ids)
        out = list(self.history.tolist())
        for _ in range(max_new_tokens):
            next_id = self._sample(logits[0])
            if self.eos_token_id is not None and next_id == self.eos_token_id:
                break
            logits = self.step(next_id)
            out.append(int(next_id))
        return out

    def _sample(self, logits_1xb):
        x = logits_1xb.float().view(-1)
        x = _apply_repetition_penalty(x, self.history, self.repetition_penalty)
        if self.temperature == 0.0:
            return int(torch.argmax(x))
        elif self.temperature != 1.0:
            x = x / self.temperature
        if self.top_k is not None and self.top_k > 0:
            v, _ = torch.topk(x, self.top_k)
            x = torch.where(x >= v[-1], x, torch.tensor(float("-inf"), device=x.device))
        if self.top_p is not None and 0.0 < self.top_p < 1.0:
            probs = F.softmax(x, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            mask = cdf <= self.top_p
            mask[..., 0] = True
            keep = sorted_idx[mask]
            new_x = torch.full_like(x, float("-inf"))
            new_x[keep] = x[keep]
            x = new_x
        probs = F.softmax(x, dim=-1)
        return int(torch.multinomial(probs, 1))

