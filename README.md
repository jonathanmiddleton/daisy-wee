# Daisy‑Wee — Efficient GPT Training & Inference for Small‑B Models

A compact, high‑efficiency GPT training stack optimized for small‑to‑mid‑scale decoder‑only models. Clear defaults, disciplined precision (BF16‑first), sliding‑window attention, and pragmatic tooling.



<table border="0" style="border: 0 !important; border-collapse: collapse;">
  <tr style="border: 0 !important;">
    <td style="border: 0 !important;">
        <ul>
        <li><b>Targets:</b> ~150M–1.6B GPT‑style models</li>
        <li><b>Context:</b> long‑context via sliding windows and block masks</li>
        <li><b>Precision:</b> BF16 by default; selective upcasting at numerically sensitive ops</li>
        <li><b>Attention:</b> fused projections, windowed/block‑sparse masks, RoPE</li>
        <li><b>Training:</b> DDP via torchrun, token‑based eval cadence, checkpoint utilities</li>
        <li><b>Configs:</b> YAML training configs + YAML model specs with CLI overrides</li>
        <li><b>Optional:</b> W&B logging</li>
        </ul>
    </td>
    <td style="border: 0 !important;">
      <img src="assets/daisy-wee.png" alt="Daisy‑Wee" width="160">
    </td>
  </tr>
</table>



## Quickstart

### 1) Environment
```bash
conda env create -f environment.yml
conda activate daisy-wee
# or: pip install -r requirements.txt
```

### 2) (Optional) Fetch sample data
```bash
python data/cached_fineweb100B.py
# or the EDU variant:
# python data/cached_fineweb_edu_100B.py
```

### 3) Train (recommended wrapper)
```bash
./run.sh config/pretrain_450m.yml -n 8 target_tokens=3000000000
```
Notes:
- `-n` → processes per node (passed to `torchrun --nproc_per_node`).
- Any YAML key can be overridden on the CLI: `key=value` or `--key=value`.

### 4) Sample from a checkpoint
```bash
python sample.py checkpoints/state_step_100000.pt --device cuda   --max_tokens 256 --temperature 0.7 --top_k 50 --repetition_penalty 1.15 --seed 123
```

---

## Training

### Using `run.sh`
```bash
./run.sh CONFIG [-n NUM_PROCS] [-p CHECKPOINT] [-s BEGIN_SHARD] [-r RUN_ID] [key=value ...]
```
Common patterns:
```bash
# 1.6B warm‑start (weights only), 8 GPUs
./run.sh config/pretrain_1.6B.yml -n 8 -p checkpoints/state_step_100000.pt

# Single‑GPU debug
./run.sh config/pretrain_450m.yml -n 1 val_loss_every_tokens=200000000

# Force full attention windows for the entire run
./run.sh config/pretrain_450m.yml -n 8 --full_windows
```
Behavior:
- `-p/--init_checkpoint` loads weights + essential arch hparams; **schedules/optimizer/steps do not resume**.
- Data traversal can be resumed approximately with `-s BEGIN_SHARD`.
- `--full_windows` forces long attention windows for the whole run.

### Direct `train.py`
```bash
python -u train.py config/pretrain_450m.yml
python train.py --help
```

### Weights & Biases (optional)
```bash
./run.sh config/pretrain_450m.yml -n 8 wandb_log=true wandb_project=myproj wandb_run_name=exp1
```
Logs: `train/loss`, tokens, time; `val/loss`, `val/ppl`.

---

## Configuration

Two layers of YAML:

1) **Training config** (`config/*.yml`): run hyperparameters and dataset shards.  
2) **Model spec** (`model_specs/*.yml`): architecture (`num_layers`, `num_heads`, `head_dim`, `model_dim`, `attention_window_len`, `max_seq_len`, `vocab_size`, `eos_token_id`, `model_class`).

Resolution order: model spec ← training config ← CLI overrides.

Key fields (training):
- `train_shards`, `val_shards`: glob(s) for tokenized binary shards
- `training_sequence_length`, `val_seq_len`
- `attention_window_len`, `window_block_size` (must divide both seq len and window)
- `target_tokens`, `cooldown_frac`
- `learning_rate_schedule`: `linear_decay` | `linear_warmup_cosine_decay` | `constant_with_cosine_decay`
- `val_loss_every_tokens`, `tot_val_tokens`
- `save_checkpoint`, `checkpoint_per_n_tokens`, `checkpoint_warmup_tokens`
- `full_windows`: force long attention windows throughout
- `optimizers`: list of optimizers with per‑group LRs (e.g., `embed_params`, `hidden_matrix_params`, `scalar_params`, `head_params`)

Example model spec (350M):
```yaml
model_class: models.daisy.daisy_core.DaisyCore
eos_token_id: 50256
vocab_size: 50257
num_layers: 24
num_heads: 16
head_dim: 64
model_dim: 1024
attention_window_len: 3456
max_seq_len: 65536
```

---

## Inference

`sample.py` reconstructs the model from the checkpointed hparams and runs generation:
```bash
python sample.py /path/to/checkpoint.pt --device cuda   --max_tokens 256 --temperature 0.7 --top_k 50 --repetition_penalty 1.15 --seed 123
```

Programmatic:

```python
import torch
from tools.checkpoint import model_from_checkpoint

m, hyperparameters = model_from_checkpoint('checkpoints/state_step_100000.pt', device='cuda')
m.eval()
```

---

## Checkpoints

Saved: model weights, merged `hparams`, `step`, `best_val`, `tokens_per_step`, and a compact progress state.

- **Warm‑start** (`-p/--init_checkpoint`): weights + core arch hparams; schedules and optimizer state start fresh.
- **Full resume** (optimizer/steps): not currently supported.
- Utilities:
  ```bash
  python -m tools.inspect_checkpoint checkpoints/state_step_100000.pt
  python -m tools.inspect_checkpoint checkpoints/state_step_100000.pt --json
  ```

---

## Repo Layout (typical)
```
config/           # training YAMLs
model_specs/      # architecture specs
models/daisy/      # attention, block, mlp, core
training/         # training loop, hparams, optimizers, progress
tools/            # checkpoint, inspect, LR sweeps, reports
data/             # dataset scripts (FineWeb, SFT)
inference/        # generation, KV cache
run.sh            # launcher
train.py          # training entrypoint
sample.py         # minimal sampling
```

---

## Design Notes

- Decoder‑only Transformer (GPT) with RoPE.  
- Sliding‑window + block‑mask attention; fused QKV(+O) projections.  
- Residual path uses learned scalar gates; optional depth skips.  
- BF16‑first; targeted upcasting for numerically fragile ops.  
- DDP training; sharded eval; token‑based checkpoint cadence.

---

## Roadmap

Mixture‑of‑Experts FFN (top‑k routing), router regularization, expert‑parallel placement, MLA.


---

## Acknowledgements

Originally launched as a fork of **modded‑nanogpt** (Keller Jordan & contributors) and inspired by **nanoGPT** (Andrej Karpathy). This project currently reproduces
substantial portions of optimized model training path code from **modded‑nanogpt** for training efficiency.


## License

See `LICENSE`.


### Learning rate schedules

Training uses a normalized progress meter s in [0,1] (tokens_processed/target_tokens) to drive schedules:
- linear_decay: Constant LR scale of 1.0 until the final cooldown_frac of training, then linear decay to 0.0.
- linear_warmup_cosine_decay: Linear warmup from 0.0 to 1.0 until 1 - cooldown_frac, then cosine decay to 0.0 over the final cooldown_frac.
- constant_with_cosine_decay: Constant LR scale of 1.0 until 1 - cooldown_frac, then cosine decay to 0.0 over the final cooldown_frac.

Notes:
- Schedules apply to all optimizers except Muon; Muon instead linearly warms its momentum based on s.
- cooldown_frac must be in [0,1]. When 0, decay is disabled for the constant schedules.
