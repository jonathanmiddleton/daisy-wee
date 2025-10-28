import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from tools.checkpoint import model_from_checkpoint
from tools.helpers import measure_time
from inference.generate import Generator

from tools.numerics.config import RunConfig

from .io_utils import Prompt, ensure_dir, prompts_from_token_shards, read_prompts_jsonl, write_jsonl, write_parquet_or_csv, write_prompts_manifest
from .metrics import (
    bootstrap_ci,
    cosine_similarity,
    delta_nll,
    edit_distance,
    jensen_shannon,
    kl_divergence,
    ref_margin,
    stable_log_softmax,
    top1_flip,
    topk_overlap,
)
from .env import record_env, record_unsupported


@dataclass
class Case:
    device: str
    dtype_policy: str
    compile: bool

    @property
    def case_id(self) -> str:
        comp = "comp" if self.compile else "eager"
        dtype = {
            "fp32": "fp32",
            "bf16": "bf16",
            "fp16": "fp16",
            "autocast_bf16": "amx",
        }.get(self.dtype_policy, self.dtype_policy)
        return f"{self.device}.{dtype}.{comp}"


class PrecisionRunner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.root = cfg.outputs.root
        ensure_dir(self.root)
        record_env(self.root)
        self.unsupported: List[Dict[str, Any]] = []
        self.cached_models = {}

    def _should_skip(self, case: Case) -> str | None:
        if case.device == "cpu" and case.dtype_policy == "fp16":
            return "cpu_fp16_autocast_unsupported"
        if case.device == "mps" and case.dtype_policy == "fp32":
            return "mps_fp32_autocast_unsupported"
        return None

    def _autocast_ctx(self, device: str, dtype_policy: str):
        devtype = "cuda" if device.startswith("cuda") else ("mps" if device.startswith("mps") else "cpu")
        if dtype_policy == "fp32":
            return torch.cuda.amp.autocast(enabled=False) if devtype == "cuda" else torch.autocast(device_type=devtype, dtype=torch.float32, enabled=False)
        if dtype_policy in ("bf16", "autocast_bf16"):
            return torch.autocast(device_type=devtype, dtype=torch.bfloat16)
        if dtype_policy == "fp16":
            return torch.autocast(device_type=devtype, dtype=torch.float16)
        # default: no autocast
        return torch.autocast(device_type=devtype, dtype=torch.float32, enabled=False)

    def _maybe_compile(self, model: torch.nn.Module, device: str, enable: bool) -> tuple[torch.nn.Module, bool, Optional[str]]:
        if not enable:
            return model, False, None
        try:
            compiled = torch.compile(model, backend="inductor", mode="default")  # type: ignore[attr-defined]
            return compiled, True, None
        except Exception as e:
            # Fallback to eager
            return model, False, str(e)

    def _load_model(self, checkpoint: str, device: str) -> tuple[torch.nn.Module, Dict[str, Any]]:
        m, h = model_from_checkpoint(checkpoint, device=device) if (checkpoint,device) not in self.cached_models else self.cached_models[(checkpoint,device)]
        self.cached_models[(checkpoint,device)] = (m, h)
        m.eval()
        return m, h

    def _get_prompts(self, device: str) -> List[Prompt]:
        ds = self.cfg.dataset
        # Prefer JSONL if provided, else sample from shards if available, else very small fallback
        if ds.path and os.path.exists(ds.path):
            try:
                import tiktoken
                enc = tiktoken.get_encoding("gpt2")
                tokenizer = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            except Exception:
                tokenizer = None
            prompts = read_prompts_jsonl(ds.path, tokenizer=tokenizer, max_prompts=ds.max_prompts)
        elif ds.shard_pattern:
            prompts = prompts_from_token_shards(ds.shard_pattern, device=device, count=ds.max_prompts, bucket_lens=ds.length_buckets)
        else:
            # Fallback: read lines from data/the_time_machine.txt
            path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "the_time_machine.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    texts = [next(f).strip() for _ in range(100)]
            else:
                texts = ["Hello world."] * 16
            try:
                import tiktoken
                enc = tiktoken.get_encoding("gpt2")
                tokenizer = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            except Exception:
                tokenizer = lambda s: [0, 1, 2, 3]
            prompts = [Prompt(id=f"p{i:05d}", text=t, tokens=tokenizer(t)[:min(ds.length_buckets)]) for i, t in enumerate(texts[: self.cfg.dataset.max_prompts])]
        # Filter out any prompts with zero tokens
        prompts = [p for p in prompts if len(p.tokens) > 0]
        if not prompts:
            raise ValueError("No non-empty prompts available; ensure dataset has at least one token per prompt.")
        # Persist manifest per spec (id, text hash)
        write_prompts_manifest(os.path.join(self.root, "prompts", "prompts.jsonl"), prompts)
        return prompts

    @torch.inference_mode()
    def _prefill_logits(self, model: torch.nn.Module, input_ids_1d: torch.Tensor) -> torch.Tensor:
        # Returns logits for last position (1, V)
        logits, _ = model.prefill(input_ids_1d[None, :], window=int(input_ids_1d.numel()))
        return logits

    @torch.inference_mode()
    def _all_prefix_logits(
        self,
        model: torch.nn.Module,
        input_ids_1d: torch.Tensor,
        *,
        window: int | None = None,
        amp_ctx=None,
    ) -> List[torch.Tensor]:
        """
        Compute logits for every prefix (positions 0..T-2) with a single full prefill + O(T) step calls.
        Returns a list of T-1 tensors with shape (1, V), where entry t corresponds to prefix length t+1.
        """
        ids = input_ids_1d
        assert ids.ndim == 1
        T = int(ids.numel())
        if T <= 1:
            return []
        win = int(window) if window is not None else T
        # 1) Run a single prefill on the full sequence to build K/V for all layers/positions
        if amp_ctx is not None:
            with amp_ctx:
                _, kv = model.prefill(ids[None, :], window=win)
        else:
            _, kv = model.prefill(ids[None, :], window=win)
        # kv shape: (2, L, B=1, H, T, D)
        L = int(kv.size(1))
        # 2) First position (t=1, pos=0): do a tiny prefill once
        if amp_ctx is not None:
            with amp_ctx:
                z1, _ = model.prefill(ids[:1][None, :], window=1)
        else:
            z1, _ = model.prefill(ids[:1][None, :], window=1)
        out: List[torch.Tensor] = [z1]
        # 3) Positions t=2..T-1 via step() using sliced KV as context
        for t in range(2, T):
            pos = t - 1  # index of current token
            tok = ids[pos]
            k_ctxs: List[torch.Tensor] = []
            v_ctxs: List[torch.Tensor] = []
            # Enforce sliding window context: last (win-1) tokens before current pos
            start = max(0, pos - win + 1)
            for i in range(L):
                # kv[0]=K, kv[1]=V; shape per layer: (B=1, H, T, D)
                k_layer = kv[0, i, 0, :, start:pos, :]
                v_layer = kv[1, i, 0, :, start:pos, :]
                # Step expects context shaped (B, Tctx, H, D) where dim=1 is time
                k_ctxs.append(k_layer.permute(1, 0, 2).unsqueeze(0).contiguous())
                v_ctxs.append(v_layer.permute(1, 0, 2).unsqueeze(0).contiguous())
            if amp_ctx is not None:
                with amp_ctx:
                    z, _, _ = model.step(tok, k_ctxs, v_ctxs, pos, win)
            else:
                z, _, _ = model.step(tok, k_ctxs, v_ctxs, pos, win)
            out.append(z)
        return out

    @torch.inference_mode()
    def _open_loop_case(self, case: Case, model: torch.nn.Module, prompts: List[Prompt], ref_logits: Dict[str, List[torch.Tensor]], topk: List[int]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for p in prompts:
            ids = torch.tensor(p.tokens, dtype=torch.long, device=case.device)
            T = ids.numel()
            # Targets are next tokens
            targets = ids[1:]
            if T <= 1:
                continue
            # Compute all prefix logits in O(T) using full prefill + step()
            amp = self._autocast_ctx(case.device, case.dtype_policy)
            hat_list = self._all_prefix_logits(model, ids, window=T, amp_ctx=amp)
            # Safety: ensure we have T-1 entries
            if not hat_list:
                continue
            for t in range(min(len(hat_list), T - 1)):
                z_hat = hat_list[t][0]
                z_ref = ref_logits[p.id][t].to(device=case.device)[0]
                # Basic metrics
                diff = (z_hat - z_ref).float()
                l2 = torch.linalg.vector_norm(diff, ord=2).item()
                linf = torch.max(torch.abs(diff)).item()
                cos = float(cosine_similarity(z_hat, z_ref).item())
                rel_l2 = float(l2 / (z_ref.float().norm() + 1e-12))
                logp_ref = stable_log_softmax(z_ref)
                logp_var = stable_log_softmax(z_hat)
                kl_pq = float(kl_divergence(logp_ref, logp_var).item())
                kl_qp = float(kl_divergence(logp_var, logp_ref).item())
                js = float(jensen_shannon(logp_ref, logp_var).item())
                flips = top1_flip(z_hat, z_ref)
                overlaps = topk_overlap(z_hat, z_ref, topk)
                margin = ref_margin(z_ref)
                y = int(targets[t].item())
                d_nll = delta_nll(logp_ref, logp_var, y)
                row = {
                    "prompt_id": p.id,
                    "pos": t,
                    "case_id": case.case_id,
                    "l2": float(l2),
                    "linf": float(linf),
                    "cosine": float(cos),
                    "rel_l2": float(rel_l2),
                    "kl_ref_to_var": float(kl_pq),
                    "kl_var_to_ref": float(kl_qp),
                    "js": float(js),
                    "flip_top1": bool(flips),
                    "margin": float(margin),
                    "delta_nll": float(d_nll),
                }
                for k, v in zip(topk, overlaps):
                    row[f"topk_overlap@{k}"] = int(v)
                rows.append(row)
        return rows

    @torch.inference_mode()
    def _closed_loop_case(self, case: Case, model: torch.nn.Module, prompts: List[Prompt], ref_outputs: Dict[str, List[int]], hparams: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        gens_rows: List[Dict[str, Any]] = []
        div_rows: List[Dict[str, Any]] = []
        window = int(hparams.get("train_attention_window_len", hparams.get("max_seq_len", 2048)))
        gen = Generator(model=model, window=window, seed=1337, device=case.device, dtype=torch.bfloat16, temperature=0.0, top_k=None, top_p=1.0, eos_token_id=int(hparams["eos_token_id"]))
        # Honor dtype policy by overriding the generator's autocast context
        gen.amp_ctx = self._autocast_ctx(case.device, case.dtype_policy)
        max_new = int(self.cfg.decoding.mode_closed_loop.get("max_new_tokens", 256))
        for p in prompts:
            gen.reset()
            toks = p.tokens[:window]
            if len(toks) == 0:
                continue
            prompt_ids = torch.tensor(toks, dtype=torch.long, device=case.device)
            it = gen.generate(prompt_ids, max_new_tokens=max_new)
            out_ids = []
            try:
                while True:
                    t = next(it)
                    out_ids.append(int(t))
            except StopIteration as e:
                (full_out, prefill_dur, step_dur) = e.value
                full_out = full_out.tolist()
                # full_out contains history prompt + generated
                out_ids = full_out[len(toks):]
            gens_rows.append({"prompt_id": p.id, "case_id": case.case_id, "output_ids": out_ids})
            # Divergence vs reference
            ref_out = ref_outputs[p.id]
            first_div = 0
            for i, (a, b) in enumerate(zip(out_ids, ref_out)):
                if a != b:
                    first_div = i
                    break
            else:
                # either equal up to min length or both equal; set to min length
                first_div = min(len(out_ids), len(ref_out))
            em_at_T = 1.0 if out_ids == ref_out else 0.0
            ed = edit_distance(out_ids, ref_out)
            # ref-NLL of variant outputs under reference model will be filled by caller (we don't have ref model here)
            div_rows.append({
                "prompt_id": p.id,
                "case_id": case.case_id,
                "first_div_idx": int(first_div),
                "em_at_T": float(em_at_T),
                "edit_distance": int(ed),
                "ref_nll": float("nan"),
                "ctx_time_ms": float(prefill_dur * 1000.0),
                "tok_time_ms": float(step_dur * 1000.0),
            })
            # print(f"ctx_time_ms: {prefill_dur}, tok_time_ms: {step_dur}")
        return gens_rows, div_rows

    @torch.inference_mode()
    def _compute_reference(self, ref_case: Case, checkpoint: str, prompts: List[Prompt]) -> Tuple[torch.nn.Module, Dict[str, List[torch.Tensor]], Dict[str, List[int]], Dict[str, Any]]:
        model, hparams = self._load_model(checkpoint, ref_case.device)
        model, compiled, err = self._maybe_compile(model, ref_case.device, ref_case.compile)
        if err:
            self.unsupported.append({"case_id": ref_case.case_id, "reason": f"compile_failed: {err}"})
        # Open-loop logits per position
        logits_map: Dict[str, List[torch.Tensor]] = {}
        for p in prompts:
            ids = torch.tensor(p.tokens, dtype=torch.long, device=ref_case.device)
            T = ids.numel()
            # Compute all prefix logits using one full prefill + step()
            logits_list = self._all_prefix_logits(model, ids, window=T, amp_ctx=None)
            logits_map[p.id] = [z.detach().cpu() for z in logits_list]
        # Closed-loop greedy reference outputs
        window = int(hparams.get("train_attention_window_len", hparams.get("max_seq_len", 2048)))
        gen = Generator(model=model, window=window, seed=1337, device=ref_case.device, dtype=torch.bfloat16, temperature=0.0, top_k=None, top_p=1.0, eos_token_id=int(hparams["eos_token_id"]))
        ref_outs: Dict[str, List[int]] = {}
        max_new = int(self.cfg.decoding.mode_closed_loop.get("max_new_tokens", 256))
        for p in prompts:
            gen.reset()
            toks = p.tokens[:window]
            if len(toks) == 0:
                continue
            ids = torch.tensor(toks, dtype=torch.long, device=ref_case.device)
            it = gen.generate(ids, max_new_tokens=max_new)
            out_ids: List[int] = []
            try:
                while True:
                    t = next(it)
                    out_ids.append(int(t))
            except StopIteration as e:
                (full_out, _, _) = e.value
                full_out = full_out.tolist()
                out_ids = full_out[len(toks):]
            ref_outs[p.id] = out_ids
        return model, logits_map, ref_outs, hparams

    @torch.inference_mode()
    def _nll_under(self, model: torch.nn.Module, device: str, prompt_tokens: List[int], cont_tokens: List[int]) -> float:
        # Compute negative log-likelihood under model for the continuation tokens given the prompt
        ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
        nll = 0.0
        for t, y in enumerate(cont_tokens):
            prefix = torch.cat([ids, torch.tensor(cont_tokens[:t], device=device, dtype=torch.long)])
            logits, _ = model.prefill(prefix[None, :], window=int(prefix.numel()))
            logp = torch.log_softmax(logits[0], dim=-1)
            nll += float(-logp[int(y)].item())
        return nll

    def run(self, checkpoint: str) -> None:
        cfg = self.cfg
        root = self.root
        # Persist config
        try:
            import yaml  # type: ignore
            os.makedirs(os.path.join(root, "configs"), exist_ok=True)
            with open(os.path.join(root, "configs", "run.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg.to_dict(), f, sort_keys=True)
        except Exception:
            pass
        # Controls: deterministic and seeds
        if cfg.controls.deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        torch.manual_seed(1337)
        # Build prompt set
        device0 = cfg.device_override or cfg.reference.device
        prompts = self._get_prompts(device=device0)
        # Compute reference
        ref_case = Case(device=cfg.reference.device, dtype_policy=cfg.reference.dtype, compile=bool(cfg.reference.compile))
        ref_model, ref_logits, ref_outs, hparams = self._compute_reference(ref_case, checkpoint, prompts)
        # Iterate cases
        tokens_rows_all: List[Dict[str, Any]] = []
        gens_rows_all: List[Dict[str, Any]] = []
        diverge_rows_all: List[Dict[str, Any]] = []
        for device in cfg.devices:
            for dtype in cfg.dtype_policies:
                for comp in cfg.compile_modes:
                    case = Case(device=device, dtype_policy=dtype, compile=bool(comp))
                    # Skip the reference cell duplication
                    if case.case_id == ref_case.case_id:
                        continue
                    reason = self._should_skip(case)
                    if reason:
                        self.unsupported.append({"case_id": case.case_id, "reason": reason})
                        continue
                    # Try load model on device
                    try:
                        model, _ = self._load_model(checkpoint, case.device)
                    except Exception as e:
                        self.unsupported.append({"case_id": case.case_id, "reason": f"load_failed: {e}"})
                        continue
                    # Try compile
                    model, compiled, err = self._maybe_compile(model, case.device, case.compile)
                    if err:
                        self.unsupported.append({"case_id": case.case_id, "reason": f"compile_failed: {err}"})
                    # Open-loop
                    if cfg.decoding.mode_open_loop.get("enabled", True):
                        try:
                            rows = self._open_loop_case(case, model, prompts, ref_logits, cfg.metrics.topk)
                            tokens_rows_all.extend(rows)
                        except Exception as e:
                            self.unsupported.append({"case_id": case.case_id, "reason": f"open_loop_failed: {e}"})
                    # Closed-loop
                    if cfg.decoding.mode_closed_loop.get("enabled", True):
                        try:
                            gens_rows, div_rows = self._closed_loop_case(case, model, prompts, ref_outs, hparams)
                            # Fill ref-NLL under ref model
                            for r in div_rows:
                                pid = r["prompt_id"]
                                ref_nll = self._nll_under(ref_model, ref_case.device, prompts[[p.id for p in prompts].index(pid)].tokens, next(g for g in gens_rows if g["prompt_id"] == pid)["output_ids"])
                                r["ref_nll"] = float(ref_nll)
                            gens_rows_all.extend(gens_rows)
                            diverge_rows_all.extend(div_rows)
                        except Exception as e:
                            self.unsupported.append({"case_id": case.case_id, "reason": f"closed_loop_failed: {e}"})
        # Write artifacts
        write_parquet_or_csv(os.path.join(root, "open_loop", "tokens.parquet"), tokens_rows_all)
        write_jsonl(os.path.join(root, "closed_loop", "generations.jsonl"), gens_rows_all)
        write_parquet_or_csv(os.path.join(root, "closed_loop", "divergence.parquet"), diverge_rows_all)
        # Summaries
        self._write_summaries(tokens_rows_all, diverge_rows_all)
        # Unsupported
        if self.unsupported:
            record_unsupported(root, self.unsupported)

    def _write_summaries(self, token_rows: List[Dict[str, Any]], div_rows: List[Dict[str, Any]]) -> None:
        # Aggregate per case basic stats and 95% CIs for key metrics
        cases = sorted(set(r["case_id"] for r in token_rows) | set(r["case_id"] for r in div_rows))
        summary: Dict[str, Any] = {}
        for cid in cases:
            rows = [r for r in token_rows if r["case_id"] == cid]
            def vals(key):
                return [float(r[key]) for r in rows if key in r]
            metrics: Dict[str, Any] = {}
            for k in ["delta_nll", "js", "flip_top1"]:
                v = vals(k)
                if k == "flip_top1":
                    # treat as rate
                    rate = sum(1 for x in v if x) / max(1, len(v))
                    metrics[k] = {"mean": rate}
                else:
                    ci = bootstrap_ci(v)
                    metrics[k] = {
                        "mean": ci.mean,
                        "median": ci.median,
                        "ci95": [ci.low, ci.high],
                    }
            # mean top-k overlaps
            for k in self.cfg.metrics.topk:
                kk = f"topk_overlap@{k}"
                lv = vals(kk)
                if lv:
                    ci = bootstrap_ci(lv)
                    metrics[kk] = {"mean": ci.mean, "median": ci.median, "ci95": [ci.low, ci.high]}
            # Flip-given-margin rates
            bins = self.cfg.metrics.margin_bins
            if bins:
                mvals = vals("margin")
                fvals = [bool(r["flip_top1"]) for r in rows]
                cond: Dict[str, Any] = {}
                edges = [0.0] + list(bins)
                # Build bin ranges
                ranges: List[Tuple[float, float | None]] = []
                for i in range(len(edges)):
                    if i == len(edges) - 1:
                        ranges.append((edges[i], None))  # > last
                    else:
                        lo = edges[i]
                        hi = edges[i+1]
                        ranges.append((lo, hi))
                for (lo, hi) in ranges:
                    name = f"({lo},{hi}]" if hi is not None else f">{edges[-1]}"
                    idxs = [j for j, m in enumerate(mvals) if (m > lo and (m <= hi if hi is not None else True))]
                    succ = sum(1 for j in idxs if fvals[j])
                    total = len(idxs)
                    rate = (succ/total) if total else float('nan')
                    cond[name] = {"rate": rate, "n": total, "successes": succ}
                metrics["flip_given_margin"] = cond
            drows = [r for r in div_rows if r["case_id"] == cid]
            if drows:
                fd = [float(r["first_div_idx"]) for r in drows]
                em = [float(r["em_at_T"]) for r in drows]
                ed = [float(r["edit_distance"]) for r in drows]
                rn = [float(r["ref_nll"]) for r in drows if ("ref_nll" in r and r["ref_nll"] == r["ref_nll"])]
                if fd:
                    ci = bootstrap_ci(fd)
                    metrics.setdefault("first_div_idx", {})["mean"] = ci.mean
                    metrics["first_div_idx"].update({"median": ci.median, "ci95": [ci.low, ci.high]})
                if em:
                    metrics["em_at_T"] = {"mean": sum(em) / max(1, len(em))}
                if ed:
                    ci = bootstrap_ci(ed)
                    metrics["edit_distance"] = {"mean": ci.mean, "median": ci.median, "ci95": [ci.low, ci.high]}
                if rn:
                    ci = bootstrap_ci(rn)
                    metrics["ref_nll"] = {"mean": ci.mean, "median": ci.median, "ci95": [ci.low, ci.high]}
            summary[cid] = metrics
        ensure_dir(os.path.join(self.root, "summaries"))
        with open(os.path.join(self.root, "summaries", "case_summaries.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        # Closed-loop divergence distribution summary
        if div_rows:
            by_case: Dict[str, List[int]] = {}
            for r in div_rows:
                by_case.setdefault(r["case_id"], []).append(int(r["first_div_idx"]))
            dist: Dict[str, Any] = {}
            for cid, arr in by_case.items():
                import torch as _t
                arr_t = _t.tensor(arr, dtype=_t.float32)
                dist[cid] = {
                    "mean_first_div": float(arr_t.mean().item()),
                    "p50": float(arr_t.median().item()),
                    "p90": float(_t.quantile(arr_t, 0.90).item()),
                    "p95": float(_t.quantile(arr_t, 0.95).item()),
                }
            with open(os.path.join(self.root, "summaries", "divergence_summary.json"), "w", encoding="utf-8") as f:
                json.dump(dist, f, indent=2)
        # Minimal comparisons vs reference (delta of means)
        if cases:
            ref_cid = f"{self.cfg.reference.device}.{('amx' if self.cfg.reference.dtype=='autocast_bf16' else self.cfg.reference.dtype)}.{('comp' if self.cfg.reference.compile else 'eager')}"
            comp = {}
            if ref_cid in summary:
                refm = summary[ref_cid]
                for cid in cases:
                    if cid == ref_cid:
                        continue
                    comp[cid] = {}
                    for k in ["delta_nll", "js"]:
                        try:
                            comp[cid][k] = float(summary[cid][k]["mean"]) - float(refm[k]["mean"])  # type: ignore[index]
                        except Exception:
                            pass
            with open(os.path.join(self.root, "summaries", "comparisons.json"), "w", encoding="utf-8") as f:
                json.dump(comp, f, indent=2)
