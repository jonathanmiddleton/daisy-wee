import os, random, math
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

random.seed(1234)

def fmt(instr, resp):
    if not instr or not resp: return None
    return f"### Instruction:\n{instr.strip()}\n\n### Response:\n{resp.strip()}\n"

def ds_dolly():
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    def mapr(r):
        instr = (r.get("instruction") or "")
        ctx = (r.get("context") or "")
        if ctx: instr = instr + "\n\n" + ctx
        out = fmt(instr, r.get("response"))
        return {"text": out} if out else None
    return ds.map(mapr, remove_columns=ds.column_names).filter(lambda r: r["text"] is not None)

def ds_oasst1():
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    id2 = {r["message_id"]: r for r in ds}
    rows = []
    for r in ds:
        if r.get("role")=="assistant":
            p = id2.get(r.get("parent_id"))
            if p and (r.get("lang") or "en").startswith("en"):
                out = fmt(p.get("text"), r.get("text"))
                if out: rows.append({"text": out})
    return Dataset.from_list(rows)

def ds_openhermes():
    ds = load_dataset("HuggingFaceTB/OpenHermes-2.5-H4", split="train")
    def mapr(r):
        msgs = r.get("messages") or r.get("conversations")
        if not msgs: return None
        u = next((m["content"] for m in msgs if m.get("role") in ("user","human")), None)
        a = next((m["content"] for m in msgs if m.get("role") in ("assistant","gpt")), None)
        out = fmt(u,a)
        return {"text": out} if out else None
    ds = ds.map(mapr, remove_columns=ds.column_names).filter(lambda r: r["text"] is not None)
    return ds.train_test_split(test_size=0.01, seed=1234)["train"]

def ds_openorca():
    ds = load_dataset("Open-Orca/OpenOrca", split="train")
    def mapr(r):
        instr = (r.get("system_prompt") or "")
        q = (r.get("question") or r.get("input") or "")
        resp = r.get("response") or r.get("output")
        if q: instr = (instr + "\n\n" + q).strip()
        out = fmt(instr, resp)
        return {"text": out} if out else None
    return ds.map(mapr, remove_columns=ds.column_names).filter(lambda r: r["text"] is not None)

def ds_norobots():
    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    def mapr(r):
        msgs = r.get("messages")
        if msgs:
            u = next((m["content"] for m in msgs if m.get("role")=="user"), None)
            a = next((m["content"] for m in msgs if m.get("role")=="assistant"), None)
            out = fmt(u,a)
        else:
            out = fmt(r.get("prompt"), r.get("response") or r.get("output"))
        return {"text": out} if out else None
    return ds.map(mapr, remove_columns=ds.column_names).filter(lambda r: r["text"] is not None)

def ds_codealpaca():
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    def mapr(r):
        instr = (r.get("instruction") or "")
        inp = (r.get("input") or "")
        if inp: instr = instr + "\n\n" + inp
        out = fmt(instr, r.get("output"))
        return {"text": out} if out else None
    return ds.map(mapr, remove_columns=ds.column_names).filter(lambda r: r["text"] is not None)

def build_mix():
    parts = [
        (ds_oasst1(), 1.5),
        (ds_dolly(), 1.0),
        (ds_openhermes(), 1.0),
        (ds_openorca(), 0.7),
        (ds_norobots(), 1.5),
        (ds_codealpaca(), 0.3),
    ]
    sizes = [len(d) for d,_ in parts]
    weights = [w*math.sqrt(n) for (_,w), n in zip(parts, sizes)]
    total = sum(weights); probs = [w/total for w in weights]
    target = 300_000
    chunks = []
    while sum(len(c) for c in chunks) < target:
        i = random.choices(range(len(parts)), probs)[0]
        chunks.append(parts[i][0].shuffle(seed=random.randint(0,1<<31)).select(range(min(2000, len(parts[i][0])))))
    ds = Dataset.from_dict({"text": sum([c["text"] for c in chunks], [])})
    return ds.train_test_split(test_size=0.01, seed=1234)

if __name__ == "__main__":
    mix = build_mix()
    tok = AutoTokenizer.from_pretrained("gpt2-xl", use_fast=True)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
    peft_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["c_attn","c_fc","c_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_cfg)

    args = TrainingArguments(
        output_dir="outputs/gpt2xl-sft",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        weight_decay=0.1,
        num_train_epochs=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=mix["train"],
        eval_dataset=mix["test"],
        args=args,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=True,
    )
    trainer.train()
    trainer.model.save_pretrained("outputs/gpt2xl-sft/adapter")
    tok.save_pretrained("outputs/gpt2xl-sft/tokenizer")
