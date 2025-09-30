# Daisy-Wee - Efficient GPT for Small‑B Models

<table>
  <tr>
    <td>
      <img src="assets/daisy-wee.png" alt="Daisy‑Wee" width="160">
    </td>
    <td>
      <i>Note: This is an exploratory project. Interfaces and results will evolve.</i>
    </td>
  </tr>
</table>


---

## Mission & Vision

**Mission.** Contribute to practical methods for training small‑B GPTs on commodity multi‑GPU hardware by exploring measurable efficiency, stable training, and long‑context instruction‑following—using sliding‑window/block‑sparse attention, RoPE, fused QKV(+O), BF16‑first training, and pragmatic optimizer design; a lightweight MoE path extends the same scaffold.

**Vision.** Soup-to-nuts, reliable, reproducible small models that practitioners can train and serve without large budgets, incrementally scaling from a 350M‑class baseline toward 1.5B+ while preserving throughput and memory efficiency. The emphasis is on openness, careful measurement, and simple designs that others can adapt.

---

## Purpose

- Evolve the core GPT architecture to **efficiently** support training and inference of small‑B models on commodity multi‑GPU setups.
- Provide an instruction‑following fine‑tuning pipeline.
- Explore MoE variants tuned for throughput and memory efficiency on small devices.
- Analyze Δ loss per token for training variations.
- Run basic analysis of prefill and sampling performance on commodity devices.

---

## Status

- A 350M‑class baseline is available with an efficient training loop and an inference path.
- Architectural components support long‑context training via sliding‑window/block‑sparse patterns and rotary embeddings.
- An MoE variant is planned and will layer onto the same training/runtime scaffolding.
- There is intentional overlap between model and training code paths; refactors will reduce duplication and centralize model definitions.

_This is research‑grade code: expect rapid iteration and occasional breaking changes._

---

## Highlights

- Efficient attention with sliding‑window and block‑sparse masking for long contexts.
- Rotary positional embeddings (RoPE).
- Fused QKV(+O) projections for speed.
- Learned residual gating/scaling to stabilize deeper networks.
- Parameter grouping and optimizer specialization (including a matrix‑preconditioned optimizer) for convergence and throughput.
- BF16‑first training with careful casting at hot spots.
- Data sharding pipeline for instruction‑tuning corpora.

---

## Architecture Overview

This project builds on a decoder‑only Transformer (GPT) with pragmatic, efficiency‑oriented enhancements.

**Token Embeddings and Head**
- Standard learned token embeddings and a projection to vocabulary logits.
- Weight shapes/casting organized for BF16 efficiency; numerically sensitive reductions are upcast.

**Rotary Positional Embeddings (RoPE)**
- RoPE is applied to queries/keys for position information compatible with long contexts and sliding windows.

**Attention: Fused, Windowed, Block‑Sparse**
- Multi‑head self‑attention with fused QKV(+O) projection to reduce kernel launches and improve locality.
- Sliding‑window attention bounds complexity to O(T·W) vs. O(T²); block masks blend local dense windows with periodic long‑range links.
- Masks are constructed at runtime to mix local and scheduled long‑distance connectivity.

**Feed‑Forward and Residual Path**
- Standard two‑layer MLP sized for target scales.
- Learned scalar gates on residual paths to stabilize depth and enable selective multi‑path mixing.

**Depth Skips with Learned Gating**
- Optional skip connections from earlier layers, gated by learned scalars, to improve gradient flow in deeper stacks.

**Value‑Embedding Side Channels (conditioning)**
- Lightweight value‑like embeddings can be injected at chosen layers for instruction conditioning; a no‑op when unused.

**Long‑Context via Block Masks**
- A mask generator produces short‑range and long‑range block patterns; layers can alternate or interleave them.

**Precision Strategy**
- BF16 by default in compute‑heavy paths; upcast where needed (e.g., reductions, logits) for stability.

---

## Training System and Optimizations

**Distributed Training**
- Multi‑GPU via DDP. Per‑rank sharded data, consistent validation, and checkpointing.

**Optimizer Strategy**
- Parameter grouping (embeddings, large matrices, scalars/gates, output head).
- Dual‑optimizer option: matrix‑preconditioned optimizer for large matrices; lightweight first‑order for small params.
- Preconditioning uses an approximate inverse square‑root (e.g., Newton–Schulz‑type updates) to control curvature at reasonable cost.

**LR Schedule**
- Warmup, then linear decay/cooldown variants tuned for depth and sequence length changes.

**BF16 & Casting Discipline**
- BF16‑first with targeted casts to mitigate precision issues while retaining memory savings.

**Attention Backends**
- Attention path is swappable; optimized kernels (e.g., Flash/Flex‑style) can be used when available.
- Sliding‑window and block masks are computed efficiently and applied compatibly with fused kernels.

**Data Pipeline**
- Tokenized datasets stored in compact binary shards with lightweight headers.
- Instruction‑mix builder generates training/validation shards; tiny validation shards support fast sanity checks.

---

## Instruction‑Following Fine‑Tuning

**Data Format**
- 16‑bit token ID streams with a small per‑shard header.
- Instruction–response pairs concatenated with consistent separators/special tokens for next‑token prediction.

**Sharding**
- Helper script builds/refreshes shards to a token budget and creates a small validation shard.

**Training**
- Fine‑tuning reuses the same model/training scaffolding with adjusted sequencing/sampling.
- For small/mid‑size models, use a modest LR, warmup, and conservative batch sizes to avoid overfitting.

---

## Roadmap: Mixture‑of‑Experts (MoE)

Planned features:
- **Expert‑Parallel FFN:** Replace dense MLPs with top‑k gated experts per token; attention remains shared.
- **Router Regularization & Load Balancing:** Prevent expert collapse.
- **Efficient Expert Placement:** Expert parallelism across devices; cache‑friendly layouts and fused scatter/collect.
- **Compatibility:** Same attention/masking stack and precision strategy; drop‑in dense‑MLP replacement.

MoE will be introduced incrementally and benchmarked against the dense baseline for quality and throughput.

---

## Setup

**Environment**
- Conda environment file and requirements are provided. Use your preferred Conda workflow.
- We use the Torch nightlies (currently 2.10) and CUDA 12.6
- Training requires your CUDA device of choice. run.sh assumes 8x H100 but 1x H100 is fine.

**Quickstart**
1. Download pretraining data. (data/cached_fineweb100B.py - you need only the first 103 files for 10B, edit the script to download more)
2. Launch training for the 350M baseline.
3. Prepare data shards for instruction fine‑tuning. 
4. Launch SFT.

**Example Commands**
- Inspect training options:
  - `python train.py --help`
- Start a pretraining run:
  - `python -u train.py config/pretrain.yml`
- Generate samples after training:
  - `python sample.py`
- Build SFT data shards:
  - `python build_sft_shards.py`
- Start an SFT run:
  - `python -u train.py config/instruct_sft.yml`

---

## Repository Structure (high level)

- Model definition: GPT blocks with RoPE, fused attention projections, block masks for sliding windows, and learned residual gating/scaling.
- Training script: ~350M baseline demonstrating data/optimizer/schedule setup and sliding‑window attention.
- Utilities: generation and instruction‑data sharding.

Refactors will consolidate model definitions as the single source of truth; training scaffolding remains modular.

---

## Contributing

Issues and PRs are welcome. Please open an issue to discuss substantial changes or new experiments.

---

## Acknowledgements

- Originally launched as a fork of **modded‑nanogpt** (Keller Jordan & contributors) and inspired by ← **nanoGPT** (Andrej Karpathy). This project currently reproduces
substantial portions of training code from **modded‑nanogpt**.
- Thanks to the open‑source community for ongoing work on efficient attention kernels, distributed training, and optimizer research.

---

## License

Released under the terms of the license in the `LICENSE` file.
