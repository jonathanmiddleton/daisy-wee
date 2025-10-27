import time
from typing import Generator
import torch
import torch.nn.functional as F
from inference.kv_cache import KVCache

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
    s, i = torch.sort(probs, dim=-1, descending=True)
    c = torch.cumsum(s, dim=-1)
    ms = c <= p
    ms[..., 0] = True
    m = torch.zeros_like(ms, dtype=torch.bool).scatter(-1, i, ms)
    return x.masked_fill(~m, float("-inf"))

def _gumbel(shape, device, dtype):
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u.clamp_min_(1e-6)))

def _sample_device(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    x = logits / max(temperature, 1e-6)
    x = _topk_filter(x, top_k)
    x = _topp_filter(x, top_p)
    # Gumbel-Softmax trick
    g = _gumbel(x.shape, x.device, x.dtype)
    y = torch.argmax(x + g, dim=-1)
    return y

def _repetition_penalty_device(logits, prev_ids, rep_p: torch.Tensor, _one: torch.Tensor, rep_w=128, rep_h=140.0, cap=3.0):
    if torch.equal(rep_p, _one) or prev_ids is None or prev_ids.numel() == 0:
        return logits
    prev = prev_ids[-rep_w:]
    d = torch.arange(prev.numel(), 0, -1, device=logits.device, dtype=logits.dtype)
    w = (0.5 ** (d / rep_h))
    h = torch.zeros_like(logits)
    h.scatter_add_(0, prev, w)
    h.clamp_max_(cap)
    scale = torch.pow(rep_p, h)
    factor = torch.where(logits > 0, 1.0 / scale, scale)
    return logits * factor


class Generator:
    def __init__(self, model, window, seed, device=None, dtype=torch.float16, max_seq_len: int = 65536, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, eos_token_id=None):
        if eos_token_id is None:
            eos_token_id = getattr(model, "eos_token_id", None)
        assert eos_token_id is not None
        self.model = model.eval()
        self.device = next(model.parameters()).device if device is None else device
        self.dtype = dtype
        assert self.device is not None
        devtype = "cuda" if str(device).startswith("cuda") else ("mps" if str(device).startswith("mps") else "cpu")
        h = None; d = None
        for b in self.model.blocks:
            a = getattr(b, "attn", None)
            if a is not None:
                h = a.num_heads; d = a.head_dim; break
        self.cache = KVCache(L=len(self.model.blocks), B=1, H=h, W=window, D=d, device=self.device, dtype=dtype)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.rep_p_t = torch.tensor(repetition_penalty, device=self.device, dtype=dtype)
        self.eos_token_id = torch.tensor(eos_token_id, device=self.device)
        self.max_seq_len = max_seq_len
        self.window = window
        self.vocab_size = self.model.embed.num_embeddings
        self._one = torch.tensor(1, device=self.device, dtype=dtype) # cached value for passing to compiled function
        # set global seed across all devices
        self.seed = seed
        self.reset()
        # compile functions
        COMPILE_SAMPLE = True # no observed benefit on MPS for compile
        COMPILE_RP = True
        COMPILE_PREFILL = True
        self.sample = torch.compile(_sample_device) if devtype != 'cpu' and COMPILE_SAMPLE else _sample_device
        self.apply_repetition_penalty = torch.compile(_repetition_penalty_device) if devtype != 'cpu' and COMPILE_RP else _repetition_penalty_device
        self.model_prefill = torch.compile(self.model.prefill_batch, dynamic=False, options={"max_autotune_gemm": False}) if devtype != 'cpu' and COMPILE_PREFILL else self.model.prefill_batch

        # finish initialization
        self.reset()
        # perform brief warmup for stable metrics
        self.warmup()

    def warmup(self, steps=3):
        for _ in range(steps):
            max_tokens = 2; prompt_len = self.window-max_tokens
            gen = self.generate(torch.randint(0, self.vocab_size, (prompt_len,), device=self.device), max_tokens)
            for i in range(max_tokens): next(gen)
            self.reset()

    def _sync(self):
        d = str(self.device)
        if d.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif d.startswith("mps") and hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.synchronize()

    def set_temperature(self, temperature: float):
        self.temperature = float(temperature)
        return self

    def set_repetition_penalty(self, repetition_penalty: float):
        self.rep_p_t = torch.tensor(repetition_penalty, device=self.rep_p_t.device, dtype=self.rep_p_t.dtype)
        return self

    def reset(self):
        self.model.reset()
        torch.manual_seed(self.seed)
        self.cache.reset()
        self.history = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.device)
        self.history_len = 0

    def _prefill(self, prompt_ids: torch.Tensor):
        logits, kv = self.model_prefill(prompt_ids[None,:], window=self.window)
        logits, kv = logits.to(self.dtype), kv.to(self.dtype)
        logits = logits[..., :self.vocab_size]
        self.cache.bulk_write_packed(kv, len(prompt_ids), window=self.window)
        end = self.history_len + prompt_ids.numel()
        self.history[self.history_len:end] = prompt_ids
        self.history_len = end
        return logits

    def _step(self, token: torch.Tensor):
        k_ctxs, v_ctxs = [], []
        for i in range(len(self.model.blocks)):
            kc, vc = self.cache.view(i)
            assert(kc.dtype == self.dtype); assert(vc.dtype == self.dtype)
            k_ctxs.append(kc); v_ctxs.append(vc)
        logits, k_new, v_new = self.model.step(token, k_ctxs, v_ctxs, self.cache.t, self.window)
        logits = logits.to(self.dtype)
        k_new, v_new = [x.to(self.dtype) for x in k_new], [x.to(self.dtype) for x in v_new]
        logits = logits[..., :self.vocab_size]
        for i in range(len(self.model.blocks)):
            assert(k_new[i] is not None); assert(v_new[i] is not None)
            self.cache.write(i, k_new[i], v_new[i])
        self.cache.advance()
        self.history[self.history_len] = token
        self.history_len += 1
        return logits

    @torch.inference_mode()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens) -> Generator[torch.Tensor, None, tuple[torch.Tensor, float, float]]:
        assert prompt_ids.ndim == 1
        assert prompt_ids.size(0) > 0 # must have at least one token
        assert prompt_ids.size(0) <= self.window, "prompt length must be <= attention window"
        assert prompt_ids.size(0)
        assert max_new_tokens > 0
        self.reset()
        self._sync(); t0 = time.perf_counter()
        logits = self._prefill(prompt_ids)
        self._sync(); t1 = time.perf_counter()
        for i in range(max_new_tokens):
            logits = self.apply_repetition_penalty(logits[-1], self.history, self.rep_p_t, self._one)
            tok = self.sample(logits, self.temperature, self.top_k, self.top_p)
            if tok == self.eos_token_id:
                break
            yield tok.view(1)
            logits = self._step(tok) if i < max_new_tokens - 1 else None # don't step on terminal case
        self._sync(); t2 = time.perf_counter()
        prefill_duration = t1 - t0
        step_duration = t2 - t1
        out = self.history[:self.history_len].detach().clone()
        return out, prefill_duration, step_duration
