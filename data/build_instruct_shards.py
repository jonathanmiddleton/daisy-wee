import os, math, random, numpy as np, hashlib, time
from datasets import load_dataset, DatasetDict
from transformers import GPT2TokenizerFast

# settings
EOS = 50256
TOKENS_PER_SHARD = 50_000_000
MAGIC = 20240520
VERSION = 1
VAL_FRAC = 0.05
PROGRESS_STEP_TOKENS = 2_000_000

os.environ["TOKENIZERS_PARALLELISM"] = "true"
random.seed(1337)

tok = GPT2TokenizerFast.from_pretrained("gpt2", model_max_length=1_000_000)
assert tok.eos_token_id == EOS

def _fmt(instr, resp):
    if instr is None or resp is None: return None
    instr = instr.strip(); resp = resp.strip()
    if not instr or not resp: return None
    return f"### Instruction:\n{instr}\n\n### Response:\n{resp}\n"

def _pairs_oasst1():
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    id2row = {r["message_id"]: r for r in ds}
    for r in ds:
        if r.get("role") == "assistant":
            p = id2row.get(r.get("parent_id"))
            if p and p.get("role") == "prompter" and (r.get("lang") or "en").startswith("en"):
                y = _fmt(p.get("text"), r.get("text"))
                if y: yield y

def _pairs_dolly15k():
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    for r in ds:
        ctx = r.get("context") or ""
        instr = (r.get("instruction") or "") + (f"\n\n{ctx}" if ctx else "")
        y = _fmt(instr, r.get("response"))
        if y: yield y

def _pairs_openhermes():
    ds = load_dataset("HuggingFaceTB/OpenHermes-2.5-H4", split="train_sft")
    for r in ds:
        msgs = r.get("messages") or r.get("conversations")
        if not msgs: continue
        u, a = None, None
        for m in msgs:
            if m.get("role") in ("user","human") and u is None:
                u = m.get("content")
            elif m.get("role") in ("assistant","gpt") and u is not None:
                a = m.get("content"); break
        y = _fmt(u, a)
        if y: yield y

def _pairs_openorca():
    ds = load_dataset("Open-Orca/OpenOrca", split="train")
    for r in ds:
        instr = (r.get("system_prompt") or "").strip()
        q = (r.get("question") or r.get("input") or "").strip()
        resp = r.get("response") or r.get("output")
        if q: instr = (instr + ("\n\n" if instr else "" ) + q)
        y = _fmt(instr, resp)
        if y: yield y

def _pairs_norobots():
    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    for r in ds:
        msgs = r.get("messages")
        if msgs:
            u = next((m["content"] for m in msgs if m.get("role")=="user"), None)
            a = next((m["content"] for m in msgs if m.get("role")=="assistant"), None)
            y = _fmt(u, a)
        else:
            y = _fmt(r.get("prompt"), r.get("response") or r.get("output"))
        if y: yield y

def _pairs_codealpaca():
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    for r in ds:
        instr = r.get("instruction") or ""
        inp = r.get("input") or ""
        if inp: instr = instr + "\n\n" + inp
        y = _fmt(instr, r.get("output"))
        if y: yield y

def _pairs_metamath_subset(max_rows=30000):
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    n = min(len(ds), max_rows)
    for i in range(n):
        r = ds[i]
        q = r.get("query") or r.get("problem") or r.get("instruction")
        a = r.get("response") or r.get("answer") or r.get("output")
        y = _fmt(q, a)
        if y: yield y

SOURCES = [
    ("oasst1", _pairs_oasst1, 1.5),
    ("dolly15k", _pairs_dolly15k, 1.0),
    ("openhermes", _pairs_openhermes, 1.0),
    ("openorca", _pairs_openorca, 0.7),
    ("no_robots", _pairs_norobots, 1.5),
    ("codealpaca", _pairs_codealpaca, 0.3),
    ("metamath", _pairs_metamath_subset, 0.3),
]

def _split_by_hash(items, val_frac):
    tr, va = [], []
    thr = int(val_frac * 1000)
    for t in items:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % 1000
        (va if h < thr else tr).append(t)
    return tr, va

def _log(s): print(s, flush=True)

def summarize_sources(val_frac=VAL_FRAC):
    rows = []
    _log(f"Scanning sources and computing deterministic MD5 split (val_frac={val_frac:.3f})...")
    for name, fn, w in SOURCES:
        items = list(fn())
        if not items: continue
        tr, va = _split_by_hash(items, val_frac)
        rows.append((name, len(tr), len(va), w))
    if not rows:
        _log("No data available."); return None
    tot_tr = sum(r[1] for r in rows); tot_va = sum(r[2] for r in rows)
    def probs(rows, use_val=False):
        xs = [w * math.sqrt(v if use_val else t) for _, t, v, w in rows]
        s = sum(xs) or 1.0
        return [x / s for x in xs]
    p_tr, p_va = probs(rows, False), probs(rows, True)
    _log(f"{'source':<14}{'train':>12}{'val':>12}{'p_train':>12}{'p_val':>12}")
    for (name, trn, val, _), pt, pv in zip(rows, p_tr, p_va):
        _log(f"{name:<14}{trn:>12,}{val:>12,}{pt:>12.3f}{pv:>12.3f}")
    _log(f"{'TOTAL':<14}{tot_tr:>12,}{tot_va:>12,}")
    return {'rows': rows, 'tot_tr': tot_tr, 'tot_va': tot_va, 'p_tr': p_tr, 'p_va': p_va}

def iter_mixture(split="train", val_frac=VAL_FRAC):
    pools = []
    for name, fn, w in SOURCES:
        items = list(fn())
        if not items: continue
        tr, va = _split_by_hash(items, val_frac)
        pools.append((name, va if split == "val" else tr, w))
    active = [(n, it, w) for n, it, w in pools if it]
    if not active: return
    lens = [len(p[1]) for p in active]
    weights = [p[2] * math.sqrt(l) for p, l in zip(active, lens)]
    total = sum(weights) or 1.0
    probs = [w / total for w in weights]
    iters = [iter(random.sample(p[1], len(p[1]))) for p in active]
    while True:
        k = random.choices(range(len(active)), probs)[0]
        try:
            yield next(iters[k])
        except StopIteration:
            iters[k] = iter(random.sample(active[k][1], len(active[k][1])))

def write_shards(out_dir, split="train", target_tokens=200_000_000, max_tokens_per_shard=TOKENS_PER_SHARD):
    os.makedirs(out_dir, exist_ok=True)
    _log(f"Start {split}: target_tokens={target_tokens:,}, shard_cap={max_tokens_per_shard:,}")
    acc = []
    tokens_written = 0
    shard_id = 0
    examples = 0
    tokens_since_report = 0
    t0 = time.time()
    for text in iter_mixture(split=split, val_frac=VAL_FRAC):
        ids = tok.encode(text) + [EOS]
        acc.extend(ids)
        examples += 1
        tokens_since_report += len(ids)
        if tokens_since_report >= PROGRESS_STEP_TOKENS:
            elapsed = time.time() - t0
            speed = (tokens_written + tokens_since_report) / max(elapsed, 1)
            _log(f"[{split}] progress: {tokens_written + tokens_since_report:,} tokens, {examples:,} examples, ~{speed:,.0f} tok/s")
            tokens_since_report = 0
        if len(acc) >= max_tokens_per_shard or (target_tokens and tokens_written + len(acc) >= target_tokens):
            arr = np.array(acc, dtype=np.uint16)
            assert (arr < 65536).all()
            header = np.zeros(256, dtype=np.int32)
            header[0] = MAGIC; header[1] = VERSION; header[2] = len(arr)
            path = os.path.join(out_dir, f"instruct_{split}_{shard_id:02d}.bin")
            with open(path, "wb") as f:
                f.write(header.tobytes()); f.write(arr.tobytes())
            size_mb = os.path.getsize(path) / 1e6
            tokens_written += len(arr)
            shard_id += 1
            _log(f"[{split}] wrote {os.path.basename(path)}: {len(arr):,} tokens, {size_mb:.1f} MB")
            acc = []
            if target_tokens and tokens_written >= target_tokens:
                break
    elapsed = time.time() - t0
    avg_tok_per_ex = (tokens_written / max(examples, 1))
    avg_tok_per_shard = (tokens_written / max(shard_id, 1))
    _log(f"Done {split}: tokens={tokens_written:,}, examples={examples:,}, shards={shard_id}, "
         f"avg_tok/ex={avg_tok_per_ex:,.1f}, avg_tok/shard={avg_tok_per_shard:,.0f}, "
         f"time={elapsed:.1f}s, ~{tokens_written / max(elapsed, 1):,.0f} tok/s")

if __name__ == "__main__":
    out_dir = "data/instruct_mix"
    summarize_sources(VAL_FRAC)
    write_shards(out_dir, split="train", target_tokens=200_000_000, max_tokens_per_shard=50_000_000)
    write_shards(out_dir, split="val",   target_tokens=10_485_760,   max_tokens_per_shard=50_000_000)
