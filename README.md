# Daisy-Wee - Efficient GPT for Small‑B Models

<table>
  <tr>
    <td>
      <img src="assets/daisy-wee.png" alt="Daisy‑Wee" width="160">
    </td>
    <td>
      Soup-to-nuts hyper-efficient training and inference tools for small-B models for learners without large budgets. CUDA GPU is required.
    </td>
  </tr>
</table>


## Highlights

- Efficient attention with sliding‑window and block‑sparse masking for long contexts.
- Rotary positional embeddings (RoPE).
- Fused QKV(+O) projections for speed.
- Learned residual gating/scaling to stabilize deeper networks.
- Parameter grouping and optimizer specialization (including a matrix‑preconditioned optimizer) for convergence and throughput.
- BF16‑first training with careful casting at hot spots.
- Data sharding pipeline for instruction‑tuning corpora.
 
---

## Training and Inference

Below are the current, tested ways to launch training, run inference sampling, and work with the provided YAML configuration files.

### Training: run.sh (recommended wrapper)
- Syntax
  ```sh
  ./run.sh CONFIG_FILE [-n NUM_PROCS] [-p CHECKPOINT_PATH] [-s BEGIN_SHARD] [--ignore-prior-schedule] [key=value ...]
  ```
- Arguments
  - CONFIG_FILE: Path to a YAML config (e.g., config/pretrain_350m.yml, config/pretrain_1.6B.yml, config/instruct_sft.yml, or config/instruct_sft_1.6B.yml). Required.
  - -n NUM_PROCS: Number of processes per node (passed to torchrun --nproc_per_node). Default: 8.
  - -p CHECKPOINT_PATH: Optional checkpoint to load as --init_checkpoint for train.py (resume or warm-start).
  - -s BEGIN_SHARD: Optional starting shard index for training data (exported as BEGIN_SHARD). Useful for resuming data traversal.
  - --ignore-prior-schedule: When warm-starting from a checkpoint for SFT or domain adaptation, ignore the checkpoint's saved schedule length (schedule_total_iters) so LR and sliding-window schedules start fresh for this run. Aliases: --ignore_prior_schedule, and legacy --ignore-prior-steps/--ignore_prior_steps are accepted and mapped to this behavior. Weights are still loaded; step counters may continue for logging unless otherwise overridden.
  - key=value or --key=value: Any additional overrides forwarded to train.py as --key=value. Examples: num_iterations=6000, global_batch_size=8, max_seq_len=32768.
- Notes
  - run.sh requires CONFIG_FILE as the first positional argument. Options -n/-p must appear after CONFIG_FILE.
  - Overrides without a leading -- are automatically rewritten to --key=value.
  - The bare flag --ignore-prior-schedule (or --ignore_prior_schedule; legacy --ignore-prior-steps/--ignore_prior_steps) is accepted and normalized to --ignore_prior_schedule=true under the hood.
  - The script sets sensible environment defaults (e.g., OMP_NUM_THREADS) and launches torchrun in standalone mode.
- Examples
  - Pretraining (350M), 8 GPUs:
    ```sh
    ./run.sh config/pretrain_350m.yml -n 8 num_iterations=6000
    ```
  - Pretraining (1.6B), resume from checkpoint on 8 GPUs:
    ```sh
    ./run.sh config/pretrain_1.6B.yml -n 8 -p checkpoints/state_step_100000.pt
    ```
  - Single-GPU debug run:
    ```sh
    ./run.sh config/pretrain_350m.yml -n 1 val_loss_every=200
    ```
  - Supervised fine-tuning (SFT) from a pretraining checkpoint, starting a fresh schedule (recommended):
    ```sh
    ./run.sh config/instruct_sft.yml -n 8 -p checkpoints/state_step_200000.pt --ignore-prior-schedule
    ```

### Inference: sample.py
- Syntax
  ```sh
  python sample.py /path/to/checkpoint.pt [--device DEVICE] [--max_tokens N] [--temperature T] [--top_k K] [--repetition_penalty RP] [--seed SEED] [--max_seq_len L]
  ```
- Behavior
  - Loads the model class and hparams from the checkpoint and constructs the model accordingly.
  - Uses tiktoken (gpt2) encoding and an instruction-style prompt template:

    ```python
    """
    ### Instruction:
    {your prompt}
    
    ### Response:
    """
    ```
  - The default prompt string is defined in sample.py; edit the variable prompt in the file to change it, or adapt the script to accept a CLI prompt.
  - Generation runs under torch.bfloat16 autocast when available and stops on eos_token_id (from hparams, default 50256).
- Examples
  - Minimal GPU example:
    ```sh
    python sample.py checkpoints/state_step_100000.pt --device cuda
    ```
  - Deterministic sampling with a longer continuation:
    ```sh
    python sample.py checkpoints/state_step_100000.pt --device cuda --max_tokens 256 --temperature 0.7 --top_k 50 --repetition_penalty 1.15 --seed 123
    ```

### Configuration files: YAML (config/*.yml)
- Common fields
  - train_files, val_files: Glob patterns for tokenized shard files used for training/validation.
  - num_iterations: Total training iterations (global steps).
  - cooldown_frac: Fraction of the schedule spent in cooldown/decay.
  - val_tokens: Number of tokens to evaluate during each validation pass.
  - val_loss_every: Validate every N iterations.
  - val_snapshot_every: Save a validation snapshot every M validations.
  - snapshot_skip: Skip snapshots for the first K validations.
  - save_checkpoint: Whether to write training checkpoints.
- Model fields
  - max_seq_len: Maximum context length for training/inference.
  - vocab_size: Vocabulary size (e.g., 50257 for GPT-2 BPE).
  - num_layers, num_heads, head_dim, model_dim: Transformer depth/width and per-head/overall dimensions.
  - model_type: Dispatcher key passed to models.get_model_class (default: gpt2).
- Optimizer-related (when present)
  - embed_params_lr, scalar_params_lr, hidden_matrix_params_lr: Parameter-group learning rates.
  - adamw_weight_decay: Weight decay for AdamW-style optimizers.
- SFT-specific
  - init_checkpoint: Path to a pretraining checkpoint used to warm-start SFT.
  - ignore_prior_schedule: Boolean. When true, ignore the checkpoint's saved schedule_total_iters so LR and sliding-window schedules start fresh for this run (weights are still loaded). Can also be set via CLI as --ignore-prior-schedule (or --ignore_prior_schedule). Legacy --ignore-prior-steps/--ignore_prior_steps are accepted as aliases.
- Overriding config values at launch
  - Any YAML key can be overridden on the command line via run.sh using key=value (or --key=value), for example:
    ```sh
    ./run.sh config/pretrain_350m.yml -n 8 num_iterations=8000 max_seq_len=32768
    ./run.sh config/instruct_sft_1.6B.yml -n 8 -p checkpoints/state_step_200000.pt val_loss_every=100
    ```



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
  ```sh
  python train.py --help
  ```
- Start a pretraining run:
  ```sh
  python -u train.py config/pretrain.yml
  ```
- Generate samples after training:
  ```sh
  python sample.py
  ```
- Build SFT data shards:
  ```sh
  python build_sft_shards.py
  ```
- Start an SFT run:
  ```sh
  python -u train.py config/instruct_sft.yml
  ```
- Start SFT from a pretraining checkpoint and start a fresh schedule (recommended for fine-tuning on new data):
  ```sh
  python -u train.py config/instruct_sft.yml --init_checkpoint=checkpoints/state_step_200000.pt --ignore_prior_schedule=true
  ```


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
