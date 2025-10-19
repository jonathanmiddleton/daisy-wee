import argparse
import os
import math
from typing import List, Optional, Tuple

import torch
import tiktoken

from inference.generate import Generator
from tools.checkpoint import model_from_checkpoint
from tools.helpers import measure_time

"""
### Accuracy Report Summary
The script builds the reference continuation tokens (`ref_ids`) in this order:
- If you pass `--reference_text`, it tokenizes that text (GPT‑2 BPE via `tiktoken`) and truncates to `--max_new_tokens`.
- Else if you pass `--reference_file`, it searches that file for the first occurrence of your exact `--prompt` text and uses the text immediately after that occurrence as the reference continuation (tokenized and truncated to `--max_new_tokens`).
- Else it tries the bundled `data/the_time_machine.txt` in the same way (find prompt → take the following text); if nothing is found, no reference is used and accuracy metrics are skipped.

Notes:
- The prompt itself is tokenized and (if needed) truncated to the model’s attention window before generation, but the reference continuation is limited by `--max_new_tokens`.
- Tokenization is consistent throughout using GPT‑2 encoding.

### What gets compared to the reference
- The model generates up to `--max_new_tokens` tokens from the prompt on each device (CPU and MPS). Those generated tokens are compared against `ref_ids`.

### Metrics computed against the reference
If a reference exists, the following are computed per device:
- `token_accuracy`: exact token‑by‑token match rate over the overlapping length.
- `bleu1`: unigram BLEU (with brevity penalty) between generated tokens and reference tokens.
- `rougeL_f1`: ROUGE‑L F1 based on the longest common subsequence.
- `avg_nll` and `perplexity`: negative log‑likelihood and PPL of the reference continuation under the model conditioned on the prompt (teacher forcing). This is done by pre‑filling on the prompt, then stepping through each reference token and summing `-log_softmax(logits)[y]`.

### What is not a reference
- The CPU vs MPS "consistency" section is a separate side‑by‑side check (exact match and token agreement between the two generations). It is not used as the ground truth for accuracy.

### Bottom line
- Reference points = the ground‑truth continuation tokens after your prompt, sourced from `--reference_text`, or from `--reference_file` (text after the prompt in that file), or from `data/the_time_machine.txt` if available. All accuracy metrics compare the model’s generated continuation to those reference tokens.
"""

def _encode_gpt2(text: str):
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode(text, allowed_special={"<|endoftext|>"})


def _decode_gpt2(ids: List[int]):
    enc = tiktoken.get_encoding("gpt2")
    return enc.decode(ids)


def _truncate_to_window(ids: List[int], window: int) -> List[int]:
    if len(ids) <= window:
        return ids
    return ids[-window:]


def _find_reference_from_file(prompt_text: str, max_new_tokens: int, path: str) -> Optional[List[int]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None
    idx = text.find(prompt_text)
    if idx < 0:
        return None
    after = text[idx + len(prompt_text):]
    if not after:
        return []
    tok_ids = _encode_gpt2(after)
    if max_new_tokens is not None and max_new_tokens > 0:
        tok_ids = tok_ids[:max_new_tokens]
    return tok_ids


def _log_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(x.float(), dim=-1)


def nll_and_ppl_for_reference(gen: Generator, prompt_ids: torch.Tensor, ref_ids: List[int]) -> Tuple[float, float]:
    """
    Compute average negative log-likelihood (nats) and perplexity for a reference continuation
    using the incremental Generator API.
    """
    assert prompt_ids.ndim == 1
    gen.reset()
    # Prefill with the prompt to get logits for first next token
    with measure_time():
        logits = gen._prefill(prompt_ids)
    total_nll = 0.0
    count = 0
    for i, y in enumerate(ref_ids):
        # logits corresponds to next-token distribution at current position
        logp = _log_softmax(logits[0])[int(y)].item()
        total_nll += -logp
        count += 1
        # Advance with the ground-truth token to get logits for the next position
        logits = gen._step(int(y))
    avg_nll = total_nll / max(count, 1)
    ppl = float(math.exp(avg_nll))
    return avg_nll, ppl


def token_accuracy(pred: List[int], ref: List[int]) -> float:
    if not ref or not pred:
        return 0.0
    n = min(len(pred), len(ref))
    correct = sum(1 for i in range(n) if pred[i] == ref[i])
    return correct / n if n > 0 else 0.0


def bleu1(pred: List[int], ref: List[int]) -> float:
    # Unigram BLEU with brevity penalty
    if len(pred) == 0:
        return 0.0
    from collections import Counter
    pc = Counter(pred)
    rc = Counter(ref)
    overlap = sum(min(pc[t], rc[t]) for t in pc.keys())
    precision = overlap / max(len(pred), 1)
    # brevity penalty
    if len(pred) <= len(ref) and len(pred) > 0:
        bp = math.exp(1 - len(ref) / len(pred))
    else:
        bp = 1.0
    return bp * precision


def rouge_l_f1(pred: List[int], ref: List[int]) -> float:
    # LCS-based F1
    if not pred or not ref:
        return 0.0
    m, n = len(pred), len(ref)
    # DP for LCS length
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if pred[i - 1] == ref[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    L = dp[n]
    prec = L / m if m > 0 else 0.0
    rec = L / n if n > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def format_secs(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:.1f}µs"
    if s < 1:
        return f"{s * 1e3:.1f}ms"
    return f"{s:.3f}s"


def run_on_device(device: str, checkpoint: str, prompt_text: str, max_new_tokens: int, 
                  temperature: float, top_k: Optional[int], top_p: Optional[float], repetition_penalty: float) -> dict:
    model, hparams = model_from_checkpoint(checkpoint, device=device)
    model.eval()
    # Build generator
    window = int(hparams['train_attention_window_len'])
    eos_token_id = int(hparams['eos_token_id'])
    gen = Generator(model=model, window=window, device=device, eos_token_id=eos_token_id,
                    temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)

    start_ids = _encode_gpt2(prompt_text)
    start_ids = _truncate_to_window(start_ids, window)
    X = torch.tensor(start_ids, dtype=torch.long, device=device)

    # Generate
    step_tokens = []
    with measure_time() as total_timer:
        with measure_time() as pre_timer:
            logits = gen._prefill(X)
        prefill_time = pre_timer()
        # streaming generate deterministically if temperature==0, else stochastic
        step_time_accum = 0.0
        for _ in range(max_new_tokens):
            # sampling depends on internal generator state (temperature, top_k, etc.)
            next_id = gen._sample(logits[0])
            step_tokens.append(int(next_id))
            with measure_time() as st:
                logits = gen._step(int(next_id))
            step_time_accum += st()
    total_time = total_timer()

    pre_tps = len(start_ids) / prefill_time if prefill_time > 0 else float('inf')
    step_tps = len(step_tokens) / step_time_accum if step_time_accum > 0 else float('inf')

    return {
        "device": device,
        "window": window,
        "prefill_time": prefill_time,
        "step_time": step_time_accum,
        "total_time": total_time,
        "prefill_tps": pre_tps,
        "step_tps": step_tps,
        "start_ids": start_ids,
        "gen_ids": step_tokens,
        "generator": gen,
        "hparams": hparams,
    }


def main():
    parser = argparse.ArgumentParser(description="Accuracy report comparing CPU vs MPS using Generator")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pt)")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Prompt text to condition on")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--reference_text", type=str, default=None, help="Reference continuation text for accuracy metrics")
    parser.add_argument("--reference_file", type=str, default=None, help="File containing reference text to derive continuation from prompt occurrence")
    parser.add_argument("--seed", type=int, default=1337)

    cli = parser.parse_args()

    torch.manual_seed(int(cli.seed))

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    else:
        print("[warn] MPS device not available; running CPU only.")

    # Prepare reference token ids if possible
    ref_ids: Optional[List[int]] = None
    if cli.reference_text:
        ref_ids = _encode_gpt2(cli.reference_text)[: cli.max_new_tokens]
    elif cli.reference_file and os.path.exists(cli.reference_file):
        ref_ids = _find_reference_from_file(cli.prompt, cli.max_new_tokens, cli.reference_file)
    else:
        # Attempt known sample text
        default_ref_path = os.path.join(os.path.dirname(__file__), "..", "data", "the_time_machine.txt")
        default_ref_path = os.path.abspath(default_ref_path)
        if os.path.exists(default_ref_path):
            maybe = _find_reference_from_file(cli.prompt, cli.max_new_tokens, default_ref_path)
            ref_ids = maybe

    results = {}
    for dev in devices:
        r = run_on_device(
            dev,
            cli.checkpoint,
            cli.prompt,
            cli.max_new_tokens,
            cli.temperature,
            cli.top_k,
            cli.top_p,
            cli.repetition_penalty,
        )
        results[dev] = r

    # Compute accuracy metrics (if we have a reference)
    metrics = {}
    for dev, r in results.items():
        gen_ids = r["gen_ids"]
        # Token-level metrics vs reference
        tok_acc = None
        bleu = None
        rouge = None
        nll = None
        ppl = None
        if ref_ids is not None:
            tok_acc = token_accuracy(gen_ids, ref_ids)
            bleu = bleu1(gen_ids, ref_ids)
            rouge = rouge_l_f1(gen_ids, ref_ids)
            # Perplexity using ground truth continuation
            # Build prompt tensor and compute NLL
            device = r["device"]
            X = torch.tensor(r["start_ids"], dtype=torch.long, device=device)
            avg_nll, ppl_val = nll_and_ppl_for_reference(r["generator"], X, ref_ids)
            nll = avg_nll
            ppl = ppl_val
        metrics[dev] = {
            "token_accuracy": tok_acc,
            "bleu1": bleu,
            "rougeL_f1": rouge,
            "avg_nll": nll,
            "perplexity": ppl,
        }

    # Optional: compare CPU vs MPS output consistency
    consistency = None
    if "cpu" in results and "mps" in results:
        cpu_ids = results["cpu"]["gen_ids"]
        mps_ids = results["mps"]["gen_ids"]
        consistency = {
            "exact_match": 1.0 if cpu_ids == mps_ids else 0.0,
            "token_agreement": token_accuracy(cpu_ids, mps_ids) if cpu_ids and mps_ids else None,
        }

    # Pretty report
    print("\n==== Accuracy Report (CPU vs MPS) ====")
    print(f"Prompt: {cli.prompt!r}")
    if ref_ids is None:
        print("Reference: not provided (token-level accuracy/perplexity unavailable)")
    else:
        print(f"Reference: provided ({len(ref_ids)} tokens)")

    for dev in devices:
        r = results[dev]
        m = metrics.get(dev, {})
        print(f"\n-- Device: {dev} --")
        print(f"Prefill: {format_secs(r['prefill_time'])}  |  Step total: {format_secs(r['step_time'])}  |  Total: {format_secs(r['total_time'])}")
        print(f"Throughput: prefill {r['prefill_tps']:.1f} tok/s  |  step {r['step_tps']:.1f} tok/s")
        if ref_ids is not None:
            print("Metrics vs reference:")
            print(f"  - Token accuracy: {m['token_accuracy']:.3f}")
            print(f"  - BLEU-1:         {m['bleu1']:.3f}")
            print(f"  - ROUGE-L (F1):   {m['rougeL_f1']:.3f}")
            print(f"  - Avg NLL (nats): {m['avg_nll']:.4f}")
            print(f"  - Perplexity:     {m['perplexity']:.3f}")
        else:
            print("(No reference: showing generation only)")
        # Show a short preview of the generation
        preview = _decode_gpt2(r['gen_ids'][:64]).strip().replace("\n", " ")
        print(f"Output preview: {preview!r}")

    if consistency is not None:
        print("\nCPU vs MPS consistency:")
        print(f"  - Exact match:   {consistency['exact_match']:.0f}")
        if consistency['token_agreement'] is not None:
            print(f"  - Token overlap: {consistency['token_agreement']:.3f}")

    print("\n======================================\n")


if __name__ == "__main__":
    main()
