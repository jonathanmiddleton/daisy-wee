# Numerical Precision Characterization Spec — GPT Decoding across Device × Compile × DType
**Version:** 1.0 • **Date:** 2025-10-19 • **Reference:** CPU–FP32 (compile=off)

## 1) Objective
Quantify and localize numerical deviations in GPT decoding caused by device, compiler fusion/graph capture, and dtype policy. Establish bounded error vs a CPU–FP32 reference and identify dominant sources of drift.

## 2) Scope
- Models: GPT-style decoder-only transformers (causal attention), any size.
- Devices: `cpu`, `mps` (extendable).
- Compile: PyTorch eager vs `torch.compile` (backend=inductor or best-available).
- DType policies:
  - `fp32`
  - `bf16`
  - `fp16`
  - `autocast(bf16)` — `with torch.autocast(device_type=..., dtype=torch.bfloat16)`
- Decoding modes:
  - **Open-loop** (teacher forcing): pure forward-pass numerics on fixed token sequences.
  - **Closed-loop** (greedy decode): end-user-visible divergence.

## 3) Test Matrix
For each device *d* ∈ {cpu, mps} evaluate eight cases: 4 dtype policies × 2 compile modes.
Reference cell = (device=cpu, dtype=fp32, compile=off).

| Device | Compile | DType policy         | Case ID |
|--------|---------|----------------------|---------|
| cpu    | off     | fp32                 | cpu.fp32.eager (REF) |
| cpu    | on      | fp32                 | cpu.fp32.comp  |
| cpu    | off     | bf16                 | cpu.bf16.eager |
| cpu    | on      | bf16                 | cpu.bf16.comp  |
| cpu    | off     | fp16                 | cpu.fp16.eager |
| cpu    | on      | fp16                 | cpu.fp16.comp  |
| cpu    | off     | autocast(bf16)       | cpu.amx.eager* |
| cpu    | on      | autocast(bf16)       | cpu.amx.comp*  |
| mps    | off     | fp32                 | mps.fp32.eager |
| mps    | on      | fp32                 | mps.fp32.comp  |
| mps    | off     | bf16                 | mps.bf16.eager |
| mps    | on      | bf16                 | mps.bf16.comp  |
| mps    | off     | fp16                 | mps.fp16.eager |
| mps    | on      | fp16                 | mps.fp16.comp  |
| mps    | off     | autocast(bf16)       | mps.amx.eager* |
| mps    | on      | autocast(bf16)       | mps.amx.comp*  |

Notes: `*` “amx” is shorthand for autocast(bf16) policy; rename as desired. If a combo is unsupported or falls back, mark as **SKIPPED** with reason.

## 4) Definitions & Assumptions
- **Autocast policy**: ops inside the autocast context choose kernel-specific compute dtypes; reductions in attention softmax and layernorm are explicitly controlled (see Controls).
- **Compile(on)**: `torch.compile(model, backend="inductor", mode="default")` where supported; otherwise use `aot_eager` and record fallback. Shapes fixed to avoid graph breakage.
- **Greedy decode**: temperature=0, top-p=1.0, top-k=None, no sampling.
- **Teacher forcing**: at step *t*, logits computed on the same prefix tokens for all cases; targets fixed.

## 5) Controls (strict)
- Identical weights, tokenizer, prompts, attention masks, padding, BOS/EOS policy.
- `model.eval()`, no dropout; set seeds; `torch.use_deterministic_algorithms(True)` where available.
- **Accumulation dtype**: force FP32 for QKᵀ matmul/reductions, softmax, AV, and normalization reductions when possible. Record any cases where the backend refuses FP32 accumulations.
- **Epsilon and impls**: fix LayerNorm/RMSNorm ε and implementation; same GELU/SiLU variant; identical RoPE scaling and precompute dtype.
- **KV-cache dtype** fixed across runs; log dtype and layout.
- **Masking**: same additive mask dtype and application order.
- **Compile inputs**: fix sequence lengths, strides, and batch sizes to a single static signature for compiled runs.

## 6) Datasets
- 200–500 prompts spanning short (≤64), medium (≤512), long (≥2k) contexts.
- Domains: prose, code, math. Public prompts or internal benchmark set.
- Target ≥50k evaluated token positions for open-loop metrics.
- Store canonicalized prompt set (`utf-8`, normalized whitespace); ship a content hash list.

## 7) Procedures
### 7.1 Open-loop (Teacher Forcing)
1. Tokenize all prompts; truncate/pad per configured sequence length policy.
2. For each case, run forward passes over fixed prefixes without generating new tokens.
3. At each position *t*:
   - Capture final logits `ẑ_t` and reference logits `z_t`.
   - Optionally capture layerwise activations after each block (post-attn, post-mlp) for drift localization.
4. Log per-position metrics (Section 8.1).

### 7.2 Closed-loop (Greedy Decode)
1. Generate continuations up to `max_new_tokens` with greedy decoding.
2. Record first divergence index vs reference (token mismatch position).
3. Compute sequence-level metrics (Section 8.2).

## 8) Metrics (computed w.r.t. reference)
### 8.1 Per-token (Open-loop)
- **Logit error**: L2 = ‖ẑ−z‖₂; L∞ = ‖ẑ−z‖∞; cosine sim = ⟨ẑ,z⟩/(‖ẑ‖‖z‖); relative error = ‖ẑ−z‖₂/‖z‖₂.
- **Distributional**: KL(p‖p̂), KL(p̂‖p), JS(p,p̂), where p = softmax(z), p̂ = softmax(ẑ).
- **Top-k stability**: overlap@k; **Top-1 flip rate**: I[argmax(ẑ) ≠ argmax(z)].
- **Margin sensitivity**: for reference margin δ = z₁−z₂, compute flip rate conditioned on δ bins (e.g., (0,0.1], (0.1,0.5], (0.5,1], >1).
- **NLL gap (teacher-forced)**: ΔNLL = (−log p̂(y)) − (−log p(y)).
- **Layerwise drift** (optional): for each block output hᵢ, MSE(ĥᵢ,hᵢ), cosine(ĥᵢ,hᵢ).

Aggregate by mean, median, and 95% bootstrap CIs; also provide per-prompt summaries.

### 8.2 Sequence-level (Closed-loop)
- **First divergence index**.
- **Exact match rate (EM)** over the first *T* tokens.
- **Edit distance** to reference completion.
- **Reference-NLL of variant outputs**: evaluate variant completions under the reference model.
- **Throughput/latency**: context and per-token decode time (informational).

## 9) Statistical Analysis
- Paired tests (per token position) vs reference for ΔNLL and KL; report mean ± 95% bootstrap CI over tokens and prompts.
- For flip-given-margin, report conditional probabilities with Wilson CIs.
- Flag “material” deviations when mean ΔNLL > 0.02–0.05 nats/token or flip rate rises in δ>1 bin.

## 10) Artifacts & File Layout
```
artifacts_root/
  configs/
    run.yaml                          # config used
  prompts/
    prompts.jsonl                     # id, text, hash
  open_loop/
    tokens.parquet                    # per-position metrics (schema below)
    layer_drift.parquet               # optional per-layer drift
  closed_loop/
    generations.jsonl                 # per-prompt outputs
    divergence.parquet                # sequence metrics
  summaries/
    case_summaries.json               # aggregates per case
    comparisons.json                  # deltas vs reference
  reports/
    precision_report.html             # plots + tables
    precision_report.md               # markdown variant
  logs/
    env.json                          # torch, device, backend, compile info
    unsupported.json                  # skipped cases with reasons
```

## 11) Config Schema (YAML)
```yaml
run_id: "gpt-numerics-v1"
reference: {{ device: cpu, dtype: fp32, compile: false }}
devices: [cpu, mps]
compile_modes: [false, true]
dtype_policies: [fp32, bf16, fp16, autocast_bf16]
decoding:
  mode_open_loop: {{ enabled: true }}
  mode_closed_loop: {{ enabled: true, max_new_tokens: 256 }}
sampling:
  temperature: 0.0
  top_p: 1.0
controls:
  deterministic: true
  force_fp32_accum_attention: true
  force_fp32_accum_norms: true
  kv_cache_dtype: "fp16"            # or "bf16" / "fp32"; held constant
  rope_cache_dtype: "fp32"
dataset:
  path: "prompts.jsonl"
  min_prompts: 200
  max_prompts: 500
  length_buckets: [64, 512, 2048]
metrics:
  topk: [1, 5, 10]
  margin_bins: [0.1, 0.5, 1.0]
outputs:
  root: "./artifacts_root"
  write_parquet: true
```

## 12) Output Schemas
### 12.1 `open_loop/tokens.parquet`
- `prompt_id: str`
- `pos: int`
- `case_id: str`  (e.g., "mps.bf16.comp")
- `l2: float`, `linf: float`, `cosine: float`, `rel_l2: float`
- `kl_ref_to_var: float`, `kl_var_to_ref: float`, `js: float`
- `flip_top1: bool`
- `topk_overlap@k: int` (one column per k)
- `margin: float`
- `delta_nll: float`

### 12.2 `closed_loop/divergence.parquet`
- `prompt_id: str`
- `case_id: str`
- `first_div_idx: int`
- `em_at_T: float`
- `edit_distance: int`
- `ref_nll: float`
- `ctx_time_ms: float`
- `tok_time_ms: float`

### 12.3 `summaries/case_summaries.json`
Per `case_id`: means, medians, and 95% CIs for all metrics; distributional percentiles.


## 13) Device-Specific Notes
- **CPU**: FP16 ops may be emulated/slow; permitted for completeness—record fallbacks.
- **MPS**: FP16/BF16 support varies by OS/GPU; record exact macOS, PyTorch, and MPS backend versions. Expect sensitivity to reduction order and autocast choices.
- For any unsupported combo, mark **SKIPPED** and include backend messages.

## 14) Reproducibility
- Log: PyTorch version, commit, build flags; `torch.backends` settings; `torch.set_float32_matmul_precision`; `torch.compile` backend/mode; device properties; OS/kernel; model hash; tokenizer hash; prompt hashes; seeds; KV/rope dtypes; ε and norm impls.
- Persist deterministic flags and warn on backend nondeterminism.

## 15) Reporting
- Tables by case: mean ± 95% CI for ΔNLL, JS, flip rate, top-k overlaps.
- Boxplots of per-token L2 and JS by case.
- Heatmap: mean layerwise drift per block.
- Distribution of first divergence index per case.
- “Flip-given-margin” curves per case.

## 16) Pseudocode (non-normative)
```text
for case in cases:
  set_device_dtype_policy(case)
  model_case = maybe_compile(model_ref.clone())
  for prompt in prompts:
    # Open-loop
    for t in token_positions(prompt):
      z_hat = model_case.forward(prefix[:t+1])
      z_ref = logits_ref[prompt][t]
      record_token_metrics(z_hat, z_ref, target[t])
      if capture_layer_drift: record_layerwise(h_hat, h_ref)
    # Closed-loop
    out = greedy_generate(model_case, prompt, max_new_tokens)
    record_divergence(out, ref_out)
aggregate_and_write_reports()
```

## 17) Risks & Mitigations
- Graph breaks or backend fallbacks → log and demote to eager; retain case but mark “compiled=false (fallback)”.
- Implicit dtype conversions → wrap sensitive ops in explicit casts; assert dtypes at APIs.
- KV-cache policy mismatch → enforce via a single factory.
- Tokenizer discrepancies → lock exact files and versions; hash inputs.

---
**Deliverables**: artifacts folder with raw metrics (Parquet/JSONL), summary JSON, and an HTML/MD report implementing Section 16.


