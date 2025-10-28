import time
from typing import Generator, Callable, Any
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
    def __init__(self, model, window, seed, device=None, dtype=torch.float16, max_seq_len: int = 65536, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, eos_token_id=None, torch_compile=True):
        if eos_token_id is None:
            eos_token_id = getattr(model, "eos_token_id", None)
        assert eos_token_id is not None
        self.model = model.eval()
        self.device = next(model.parameters()).device if device is None else device
        self.dtype = dtype
        assert self.device is not None
        self.devtype = "cuda" if str(self.device).startswith("cuda") else ("mps" if str(self.device).startswith("mps") else "cpu")
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
        # set seed across all devices
        self.seed = seed
        torch.manual_seed(self.seed)
        self.history = torch.empty(0, dtype=torch.long, device=self.device)
        self.torch_compile = torch_compile
        # maybe compile
        COMPILE_SAMPLE = False
        self.sample = torch.compile(_sample_device, dynamic=False) if COMPILE_SAMPLE else _sample_device
        self.apply_repetition_penalty = self._maybe_compile(_repetition_penalty_device)
        self.model_prefill_not_compiled = self.model.prefill
        self.model_prefill_medium = None
        self.model_prefill_large = None
            # finish initialization
        self.reset_history()
        self.warmup()

    def _maybe_compile(self, func):
        """
        Compile if a device is available, with the appropriate context and options. Otherwise return the original function.
        """
        if not self.torch_compile:
            return func

        with torch.inference_mode():
            if self.devtype == 'mps':
                options = {"max_autotune_gemm": False}
                return torch.compile(func, dynamic=False, options=options)
            elif self.devtype == 'cuda':
                return torch.compile(func, dynamic=False)
            else:
                return func

    def _prefill_fn_for_input(self, x: torch.Tensor):
        """
        This is an ugly hack to make compiling work on MPS.

        Lazily compile model_prefill for different sequence lengths. Mark the input tensor as dynamic as appropriate.
        This is a workaround for MPS related defects in torch 2.8.0, 2.9.0, and 2.10.0-pre.
        (1) dynamic=True fails on MPS.
        (2) in torch/_meta_registrations.py:
            calling a compiled function that calls into SDPA from within torch.no_grad/inference_mode context seems creates
            a static guard on a SymInt based on internal SDPA indirection. As a workaround I compile for different
            cases implied by the guards and indirect accordingly.
        """
        assert x.ndim == 2
        assert x.size(1) > 0
        min_sdpa = 9 # magic
        max_for_fast_sdpa = 1023 # inferred from torch 2.9.0 MPS SDPA kernels

        """
            max_seq_len_sdpa:
            Torch infers that sequence length is bounded by our window_size-1:
            torch.fx.experimental.symbolic_shapes.ConstraintViolationError: Constraints violated ...
            ...Not all values of L['input_ids'].size()[1] = L['input_ids'].size()[1] in the specified range 2048 <= L['input_ids'].size()[1] <= 65536 satisfy the generated guard 2048 <= L['input_ids'].size()[1] and L['input_ids'].size()[1] <= {window_size-1}
        """
        max_seq_len_sdpa = self.window-1

        if x.size(1) > max_seq_len_sdpa:
            # compile will fail
            return self.model.prefill

        rules: list[tuple[Callable[[int], bool], tuple[str, tuple[int,int]]]] = [
            # (Predicate, (Function name, (min,max))
            (lambda n: n <= min_sdpa, ('model_prefill_not_compiled', (1, min_sdpa)),),
            (lambda n: min_sdpa < n <= max_for_fast_sdpa, ('model_prefill_medium', (min_sdpa+1, max_for_fast_sdpa))),
            (lambda n: max_for_fast_sdpa < n, ('model_prefill_large', (max_for_fast_sdpa+1, max_seq_len_sdpa))),
        ]
        fn = None
        for pred, (fn_name,(min,max)) in rules:
            if pred(x.size(1)):
                fn = getattr(self, fn_name)
                if fn is None:
                    # lazy maybe-compile
                    compiled = self._maybe_compile(self.model.prefill)
                    setattr(self, fn_name, compiled)
                    fn = compiled
                    torch._dynamo.mark_dynamic(x, 1, min=min, max=max)
                    break

        assert fn is not None
        return fn


    def warmup(self):
        """
        Force precompile of functions with inputs that induce appropriate guards. Primarily a concern for MPS.
        """
        print("Generator warming up...", end="", flush=True)

        with torch.inference_mode():
            print("[1/2]...", end="", flush=True)
            prompt_len_med = 10
            prompt_med = torch.randint(0, self.vocab_size, (1,prompt_len_med), device=self.device)
            fn_med = self._prefill_fn_for_input(prompt_med)
            fn_med(prompt_med, self.window)
            print("[2/2]...", end="", flush=True)
            prompt_len_lg = 1024
            prompt_lg = torch.randint(0, self.vocab_size, (1,prompt_len_lg), device=self.device)
            fn_lg = self._prefill_fn_for_input(prompt_lg)
            fn_lg(prompt_lg, self.window)
        print("...done.", flush=True)

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

    def reset_history(self):
        self.model.reset_history()
        self.cache.reset_history()
        # dynamic-sized history to match tests' expectations
        self.history = torch.empty(0, dtype=torch.long, device=self.device)

    def _prefill(self, prompt_ids: torch.Tensor):
        prompt_ids = prompt_ids[None,:]
        fn = self._prefill_fn_for_input(prompt_ids)
        logits, kv = fn(prompt_ids, window=self.window)
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

    @torch.inference_mode()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens) -> Generator[torch.Tensor, None, tuple[torch.Tensor, float, float]]:
        assert prompt_ids.ndim == 1
        assert prompt_ids.size(0) > 0 # must have at least one token
        assert prompt_ids.size(0) <= self.window, "prompt length must be <= attention window"
        assert prompt_ids.size(0)
        assert max_new_tokens > 0
        self.reset_history()
        self._sync(); t0 = time.perf_counter()
        logits = self._prefill(prompt_ids)
        self._sync(); t1 = time.perf_counter()
        for i in range(max_new_tokens):
            # prepare tensors for passing to potentially compiled functions, avoiding recompiles for shape changes
            logits = logits[-1]
            torch._dynamo.mark_dynamic(logits, 0, min=1, max=self.max_seq_len)
            rep_pen_w = 128
            prev_ids = self.history[-rep_pen_w:]
            torch._dynamo.mark_dynamic(prev_ids, 0, min=2, max=self.max_seq_len)
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
