from typing import Generator, Callable

import torch
import torch.nn.functional as F
from inference.kv_cache import KVCache
from tools.helpers import measure_time

def _topk_filter(x, k):
    if k is None or k <= 0 or k >= x.size(-1):
        return x
    v, i = torch.topk(x, k)
    y = torch.full_like(x, float("-inf"))
    y.scatter_(dim=-1, index=i, src=v)
    return y

def _topp_filter(x, p):
    if p is None or p <= 0.0 or p >= 1.0:
        return x
    probs = F.softmax(x, dim=-1)
    s, i = torch.sort(probs, descending=True)
    c = torch.cumsum(s, dim=-1)
    m = c <= p
    m[..., 0] = True
    keep = i[m]
    y = torch.full_like(x, float("-inf"))
    y.scatter_(dim=-1, index=keep, src=x[keep])
    return y

def _gumbel(shape, device, dtype, generator=None):
    u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    return -torch.log(-torch.log(u.clamp_min_(1e-6)))

def sample_device(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    x = logits if temperature == 1.0 else logits / max(temperature, 1e-6)
    x = _topk_filter(x, top_k)
    x = _topp_filter(x, top_p)
    g = _gumbel(x.shape, x.device, x.dtype, generator)
    return torch.argmax(x + g, dim=-1)

def repetition_penalty_device(logits, prev_ids, rep_p, rep_w=128, rep_h=140.0, cap=3.0):
    if rep_p == 1.0 or prev_ids is None or prev_ids.numel() == 0:
        return logits
    prev = prev_ids[-rep_w:]
    d = torch.arange(prev.numel(), 0, -1, device=logits.device, dtype=logits.dtype)
    w = (0.5 ** (d / rep_h))
    h = torch.zeros_like(logits)
    h.scatter_add_(0, prev, w)
    h = h.clamp_max_(cap)
    pos = logits > 0
    x = torch.empty_like(logits)
    x[pos] = logits[pos] / rep_p.pow(h[pos])
    x[~pos] = logits[~pos] * rep_p.pow(h[~pos])
    return x


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
    def __init__(self, model, window, device=None, dtype=torch.bfloat16, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, eos_token_id=None, seed=None):
        assert eos_token_id is not None
        self.model = model.eval()
        self.device = next(model.parameters()).device if device is None else device
        assert self.device is not None
        devtype = "cuda" if str(device).startswith("cuda") else ("mps" if str(device).startswith("mps") else "cpu")
        self.amp_ctx = torch.autocast(device_type=devtype, dtype=torch.bfloat16)
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
        self.eos_token_id = torch.tensor(eos_token_id, device=self.device)
        self.history = torch.empty(0, dtype=torch.long, device=self.device)
        self.window = window
        self.vocab_size = self.model.embed.num_embeddings
        self.rng = None
        if seed is not None:
            self.set_seed(seed)

    @torch.inference_mode()
    def set_temperature(self, temperature: float):
        self.temperature = float(temperature)
        return self

    @torch.inference_mode()
    def set_repetition_penalty(self, repetition_penalty: float):
        self.repetition_penalty = float(repetition_penalty)
        return self

    @torch.inference_mode()
    def set_seed(self, seed: int):
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        self.rng = g
        return self

    @torch.inference_mode()
    def reset(self):
        self.cache.reset()
        self.history = torch.empty(0, dtype=torch.long, device=self.device)

    def _prefill(self, prompt_ids):
        with self.amp_ctx:
            logits, kv = self.model.prefill_batch(prompt_ids[None,:], window=self.window)
        self.cache.bulk_write_packed(kv.bfloat16(), len(prompt_ids), window=self.window)
        self.history = torch.cat([self.history, prompt_ids], dim=0)
        return logits

    def _step_device(self, token: torch.Tensor):
        k_ctxs, v_ctxs = [], []
        for i in range(len(self.model.blocks)):
            kc, vc = self.cache.view(i)
            k_ctxs.append(kc); v_ctxs.append(vc)
        with self.amp_ctx:
            logits, k_new, v_new = self.model.step(token, k_ctxs, v_ctxs, self.cache.t, self.window)
        logits = logits[..., :self.vocab_size]
        for i in range(len(self.model.blocks)):
            if k_new[i] is not None:
                self.cache.write(i, k_new[i], v_new[i])
        self.cache.advance()
        self.history = torch.cat([self.history, token.view(1)], dim=0)
        return logits

    def _step(self, token_id):
        k_ctxs, v_ctxs = [], []
        for i in range(len(self.model.blocks)):
            kc, vc = self.cache.view(i)
            k_ctxs.append(kc); v_ctxs.append(vc)
        token = torch.tensor(token_id, device=self.device, dtype=torch.long)
        with self.amp_ctx:
            logits, k_new, v_new = self.model.step(token, k_ctxs, v_ctxs, self.cache.t, self.window)
        logits = logits[..., :self.vocab_size]
        for i in range(len(self.model.blocks)):
            if k_new[i] is not None:
                self.cache.write(i, k_new[i], v_new[i])
        self.cache.advance()
        self.history = torch.cat([self.history, token.view(1)], dim=0)
        return logits


    def generate(self, prompt_ids: torch.Tensor, max_new_tokens, seed=None) -> Generator[torch.Tensor, None, tuple[list[int], float, float]]:
        if seed is not None:
            self.set_seed(seed)
        assert prompt_ids.ndim == 1
        assert prompt_ids.size(0) > 0 # must have at least one token
        assert prompt_ids.size(0) <= self.window, "prompt length must be <= attention window"
        assert prompt_ids.size(0)
        prompt_ids = prompt_ids[self.history.size(0):]
        with torch.inference_mode():
            with measure_time() as pre_time:
                logits = self._prefill(prompt_ids)
        prefill_duration = pre_time()
        out = list(self.history.tolist())
        step_duration = 0
        for _ in range(max_new_tokens):
            # with torch.inference_mode():
            #     next_id = self._sample(logits[0])
            #     if self.eos_token_id is not None and next_id == self.eos_token_id:
            #         break
            # yield next_id
            # with torch.inference_mode():
            #     with measure_time() as step_time:
            #         logits = self._step(next_id)
            #     step_duration += step_time()
            #     out.append(int(next_id))
            with torch.inference_mode():
                logits = repetition_penalty_device(logits[-1], self.history, self.repetition_penalty)
                tok = sample_device(logits, self.temperature, self.top_k, self.top_p, self.rng)
                self.history = torch.cat([self.history, tok.view(1)], dim=0)
                with measure_time() as step_time:
                    logits = self._step_device(tok)
                step_duration += step_time()
                out.append(int(tok))

        return out, prefill_duration, step_duration

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
        if self.rng is not None:
            return int(torch.multinomial(probs, 1, generator=self.rng))
        else:
            return int(torch.multinomial(probs, 1))