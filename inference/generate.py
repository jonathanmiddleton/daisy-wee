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

def _repetition_penalty_device(logits, prev: torch.Tensor, rep_p: torch.Tensor, _one: torch.Tensor, rep_h=140.0, cap=3.0):
    # if torch.equal(rep_p, _one) or prev_ids is None or prev_ids.numel() == 0:
    #     return logits
    # prev = prev_ids[-rep_w:]
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
        devtype = "cuda" if str(self.device).startswith("cuda") else ("mps" if str(self.device).startswith("mps") else "cpu")
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
        self.history = torch.empty(0, dtype=torch.long, device=self.device)
        # compile functions
        with torch.no_grad():
            COMPILE_SAMPLE = False and devtype != 'cpu'
            COMPILE_RP = False and devtype != 'cpu'
            COMPILE_PREFILL = True and devtype != 'cpu'
            self.sample = torch.compile(_sample_device, dynamic=False) if COMPILE_SAMPLE else _sample_device
            self.apply_repetition_penalty = torch.compile(_repetition_penalty_device, dynamic=False) if COMPILE_RP else _repetition_penalty_device
            self.model_prefill = torch.compile(self.model.prefill_batch, dynamic=False, options={"max_autotune_gemm": False}) if COMPILE_PREFILL else self.model.prefill_batch

        # finish initialization
        self.reset()
        # warmup
        self.warmup()

    def maybe_compile_prefill(self, max_sequence_length: int):
        # torch==2.9.0: torch/_meta_registrations.py
        # calling a compiled function that calls into SDPA from within torch.no_grad() seems to trigger an issue
        # creating static guards. As a workaround I compile for different cases implied by the guards and indirect accordingly
        pass

    def warmup(self, steps=2):
        print("Warming up...")
        with torch.no_grad():
            prompt_len = 666
            prompt = torch.randint(0, self.vocab_size, (1,prompt_len), device=self.device)
            min_sdpa = 9
            max_for_fast_sdpa = 1023
            torch._dynamo.mark_dynamic(prompt, 1, min=min_sdpa, max=max_for_fast_sdpa)
            self.model_prefill(prompt, self.window)

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
        # dynamic-sized history to match tests' expectations
        self.history = torch.empty(0, dtype=torch.long, device=self.device)

    def _prefill(self, prompt_ids: torch.Tensor):
        with torch.no_grad():
            prompt_ids = prompt_ids[None,:]
            # torch._dynamo.mark_dynamic(prompt_ids, 1, min=1, max=self.window - 1)
            # TODO remove these hardcoded guards
            min_sdpa = 9
            max_for_fast_sdpa = 1023
            torch._dynamo.mark_dynamic(prompt_ids, 1, min=min_sdpa, max=max_for_fast_sdpa)
            logits, kv = self.model_prefill(prompt_ids, window=self.window)
            logits, kv = logits.to(self.dtype), kv.to(self.dtype)
            logits = logits[..., :self.vocab_size]
            prompt_ids = prompt_ids.squeeze(0)
            self.cache.bulk_write_packed(kv, len(prompt_ids), window=self.window)
            self.history = torch.cat([self.history, prompt_ids.to(dtype=torch.long, device=self.device)], dim=0)
            return logits

    def _step(self, token: torch.Tensor | int):
        # ensure token is a 0-dim long tensor on device
        # if not isinstance(token, torch.Tensor):
        #     token = torch.tensor(token, dtype=torch.long, device=self.device)
        # else:
        #     token = token.to(device=self.device, dtype=torch.long).reshape(())
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
        self.history = torch.cat([self.history, token.view(1)], dim=0)
        return logits

    @torch.no_grad()
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
            logits = logits[-1]
            torch._dynamo.mark_dynamic(logits, 0, min=1, max=self.max_seq_len)
            rep_pen_w = 128
            prev_ids = self.history[-rep_pen_w:]
            torch._dynamo.mark_dynamic(prev_ids, 0, min=2, max=self.max_seq_len)
            # if self.history.numel() == 128:
            #     torch._dynamo.mark_dynamic(self.history, 0, min=2, max=self.max_seq_len)
            #     self.apply_repetition_penalty = torch.compile(_repetition_penalty_device, dynamic=False)
            logits = self.apply_repetition_penalty(logits, prev_ids, self.rep_p_t, self._one)
            tok = self.sample(logits, self.temperature, self.top_k, self.top_p)
            if tok == self.eos_token_id:
                break
            yield tok.view(1)
            if i < max_new_tokens - 1:
                logits = self._step(tok)
            else:
                # Final token: append to history to reflect all generated tokens in the return value
                t = tok.to(device=self.device, dtype=torch.long).reshape(())
                self.history = torch.cat([self.history, t.view(1)], dim=0)
                logits = None
        self._sync(); t2 = time.perf_counter()
        prefill_duration = t1 - t0
        step_duration = t2 - t1
        out = self.history.detach().clone()
        return out, prefill_duration, step_duration
