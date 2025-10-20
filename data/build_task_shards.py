import os, json, math, random, numpy as np, re, time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

MAGIC = 20240520
VERSION = 2

def _final_from_gsm8k(ans: str) -> str:
    m = re.findall(r"####\s*([^\n]+)", ans)
    return (m[-1] if m else ans).strip()

def _fmt(instr: str, resp: str) -> tuple[str, str]:
    instr, resp = (instr or "").strip(), (resp or "").strip()
    if not instr or not resp: return None
    return instr, resp

def _iter_arc(subset: str, split: str):
    ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=1337)
    for r in ds:
        q = r["question"]
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        key = r["answerKey"]
        table = {l: t for l, t in zip(labels, texts)}
        instr = q + "\nOptions:\n" + "\n".join(f"{l}. {t}" for l, t in zip(labels, texts)) + "\nAnswer with the correct option letter."
        resp = key
        y = _fmt(instr, resp)
        if y: yield y

def _iter_gsm8k(subset: str, split: str):
    ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=1337)
    for r in ds:
        instr = r["question"]
        resp = _final_from_gsm8k(r["answer"])
        y = _fmt(instr, resp)
        if y: yield y

def _iter_smoltalk(split: str, stop: int | None = None):
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=1337)
    n = 0
    for r in ds:
        msgs = r.get("messages") or []
        u, a = None, None
        for m in msgs:
            if m.get("role") == "user" and u is None:
                u = m.get("content")
            elif m.get("role") == "assistant" and u is not None:
                a = m.get("content"); break
        y = _fmt(u, a)
        if y:
            yield y
            n += 1
            if stop and n >= stop: break

def _tok_pair(tok, instr: str, resp: str, eos_id: int):
    prompt = f"### Instruction:\n{instr}\n\n### Response:\n"
    p = tok(prompt, add_special_tokens=False).input_ids
    r = tok(resp, add_special_tokens=False).input_ids + [eos_id]
    x = np.array(p + r, dtype=np.uint32)  # upcast; will downcast on save
    y = np.array([-100] * len(p) + r, dtype=np.int32)
    return x, y

def _mixture(train: list[tuple[str, dict]]):
    gens = []
    for name, kw in train:
        if name == "ARC-Easy": gens.append(_iter_arc("ARC-Easy", kw.get("split","train")))
        elif name == "ARC-Challenge": gens.append(_iter_arc("ARC-Challenge", kw.get("split","train")))
        elif name == "GSM8K": gens.append(_iter_gsm8k(kw.get("subset","main"), kw.get("split","train")))
        elif name == "smol-smoltalk": gens.append(_iter_smoltalk(kw.get("split","train"), kw.get("stop")))
        else: raise ValueError(f"unknown source {name}")
    while True:
        active = [g for g in gens]
        if not active: return
        for g in active:
            try:
                yield next(g)
            except StopIteration:
                gens.remove(g)
                if not gens: return

def _write_shard(out_dir: Path, shard_id: int, X: list[np.ndarray], Y: list[np.ndarray], tok_name: str, eos_id: int):
    L = [len(x) for x in X]
    offsets = np.zeros(len(L)+1, dtype=np.int64)
    np.cumsum(np.array(L, dtype=np.int64), out=offsets[1:])
    flat_x = np.concatenate(X).astype(np.uint16, copy=False)
    flat_y = np.concatenate(Y).astype(np.int32, copy=False)
    d = out_dir / f"shard_{shard_id:05d}"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "tokens.npy", flat_x)
    np.save(d / "labels.npy", flat_y)
    np.save(d / "offsets.npy", offsets)
    meta = {"magic": MAGIC, "version": VERSION, "num_examples": len(L), "num_tokens": int(offsets[-1]), "tokenizer": tok_name, "eos_id": int(eos_id), "format": "tasks-v1"}
    # meta.update({
    #     "tokenizer_len": int(len(tok)),
    #     "eos_id": int(eos),
    #     "tokenizer_name_or_path": str(getattr(tok, "name_or_path", "")),
    # })
    (d / "meta.json").write_text(json.dumps(meta))

def build_task_shards(out_dir: str, split: str, tokenizer_name: str, max_examples_per_shard: int, sources: list[tuple[str, dict]], seed: int = 1337):
    random.seed(seed)
    out = Path(out_dir) / split
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, model_max_length=1_000_000)
    assert tok.eos_token_id is not None, "Tokenizer has no EOS. Define EOS in the model/tokenizer config, resize embeddings, and re-export before building shards."
    eos = tok.eos_token_id
    t0 = time.time()
    shard, buf_x, buf_y, n = 0, [], [], 0
    for instr, resp in _mixture(sources):
        x, y = _tok_pair(tok, instr, resp, eos)
        buf_x.append(x); buf_y.append(y); n += 1
        if n >= max_examples_per_shard:
            _write_shard(out, shard, buf_x, buf_y, tokenizer_name, eos)
            shard += 1; buf_x, buf_y, n = [], [], 0
    if n:
        _write_shard(out, shard, buf_x, buf_y, tokenizer_name, eos)

if __name__ == "__main__":
    train_sources = [
        ("ARC-Easy", {"split": "train"}),
        ("ARC-Challenge", {"split": "train"}),
        ("GSM8K", {"subset": "main", "split": "train"}),
        ("smol-smoltalk", {"split": "train", "stop": 10_000}),
    ]
    val_sources = [("smol-smoltalk", {"split": "test"})]
    build_task_shards("data/instruct_tasks", "train", "gpt2", 100_000, train_sources)
    build_task_shards("data/instruct_tasks", "val",   "gpt2", 100_000, val_sources)