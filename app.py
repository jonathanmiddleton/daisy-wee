
import os, time, json, uuid, secrets, sys
from typing import Dict, List, Optional, Tuple, Iterable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import torch

# Add repo root to path to import local modules
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.checkpoint import model_from_checkpoint
except Exception as e:
    raise RuntimeError("Unable to import tools.checkpoint.model_from_checkpoint. Run this server from the Daisy repo root or ensure it is on PYTHONPATH.") from e

try:
    from inference.generate import Generator
except Exception as e:
    raise RuntimeError("Unable to import inference.generate.Generator. Run this server from the Daisy repo root or ensure it is on PYTHONPATH.") from e

try:
    import tiktoken
except Exception as e:
    raise RuntimeError("tiktoken is required. pip install tiktoken") from e

security = HTTPBasic(auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MS
    MS = ModelState()
    yield

app = FastAPI(title="Daisy Responses-Compatible API", version="0.1.0", lifespan=lifespan)

# ---------- Auth ----------

def check_basic_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    user = os.getenv("BASIC_AUTH_USER")
    pwd = os.getenv("BASIC_AUTH_PASS")

    if user is None or pwd is None:
        return "User"

    if credentials is None or credentials.username is None or credentials.password is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not (secrets.compare_digest(credentials.username, user) and
            secrets.compare_digest(credentials.password, pwd)):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username

# ---------- Schemas

class InputText(BaseModel):
    type: str = Field("input_text", literal=True)
    text: str

class Message(BaseModel):
    role: str
    content: List[InputText]

class ResponsesRequest(BaseModel):
    model: Optional[str] = "daisy"
    input: List[Message]
    session_id: Optional[str] = None
    max_output_tokens: int = Field(256, alias="max_output_tokens")
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95
    repetition_penalty: Optional[float] = 1.15
    stream: Optional[bool] = False

# ---------- Session State

class Session:
    def __init__(self, sid: str):
        self.id = sid
        self.pairs: List[Tuple[str, str]] = []
        self.lock = None

# ---------- Model State

class ModelState:
    def __init__(self):
        self.device = self._select_device()
        ckpt = os.getenv("CHECKPOINT_PATH")
        if not ckpt:
            raise RuntimeError("CHECKPOINT_PATH env var is required")
        self.model, self.hparams = model_from_checkpoint(ckpt, device=self.device)
        self.model.eval()
        window = int(self.hparams.get("train_attention_window_len") or self.hparams.get("attention_window_len") or 2048)
        eos_id = int(self.hparams.get("eos_token_id", 50256))
        seed = int(os.getenv("SEED", "1337"))
        dtype = torch.bfloat16 if self.device.startswith("cuda") or self.device.startswith("mps") else torch.float32
        self.generator = Generator(
            model=self.model,
            window=window,
            seed=seed,
            device=self.device,
            dtype=dtype,
            eos_token_id=eos_id,
        )
        self.enc = tiktoken.get_encoding("gpt2")

    def _select_device(self) -> str:
        prefer = os.getenv("DEVICE", "").strip().lower()
        if prefer:
            return prefer
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

MS = None
SESSIONS: Dict[str, Session] = {}

# ---------- Helpers

def _get_or_create_session(session_id: Optional[str]) -> Session:
    if session_id and session_id in SESSIONS:
        return SESSIONS[session_id]
    sid = session_id or f"sess_{uuid.uuid4().hex}"
    s = Session(sid)
    SESSIONS[sid] = s
    return s

def _messages_to_pairs(messages: List[Message]) -> Tuple[Optional[str], Optional[str], List[Tuple[str,str]]]:
    system_prompt = None
    latest_user = None
    pairs: List[Tuple[str,str]] = []
    last_role = None
    last_user_text = None
    for m in messages:
        if not m.content or not m.content[0].text:
            continue
        text = m.content[0].text
        if m.role == "system":
            system_prompt = text
        elif m.role == "user":
            if last_role == "assistant" and last_user_text is not None:
                pairs.append((last_user_text, prev_assistant))  # fold
                last_user_text = None
            last_user_text = text
        elif m.role == "assistant":
            if last_user_text is not None:
                pairs.append((last_user_text, text))
                last_user_text = None
            else:
                pairs.append(("", text))
        prev_assistant = text if m.role == "assistant" else None
        last_role = m.role
    if last_user_text is not None:
        latest_user = last_user_text
    return system_prompt, latest_user, pairs

def _build_prompt(session: Session, user_text: str, system_text: Optional[str]) -> str:
    ack = "Acknowledged."
    out = []
    if system_text:
        out.append("### Instruction:\n" + system_text + "\n\n### Response:\n" + ack + "\n")
    for instr, resp in session.pairs:
        out.append("### Instruction:\n" + instr + "\n\n### Response:\n" + resp + "\n")
    out.append("### Instruction:\n" + user_text + "\n\n### Response:\n")
    return "\n".join(out)

def _encode(enc, s: str, device: str) -> torch.Tensor:
    ids = enc.encode(s)
    return torch.tensor(ids, device=device, dtype=torch.long)

def _decode(enc, token_id: int) -> str:
    return enc.decode([int(token_id)])

def _usage(out_ids: torch.Tensor, prompt_len: int) -> Dict[str,int]:
    total = int(out_ids.numel())
    completion = max(total - prompt_len, 0)
    return {"prompt_tokens": prompt_len, "completion_tokens": completion, "total_tokens": total}

# ---------- Endpoints

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/v1/sessions/reset")
def reset_session(body: dict, _: str = Depends(check_basic_auth)):
    sid = body.get("session_id")
    if not sid or sid not in SESSIONS:
        raise HTTPException(status_code=404, detail="session not found")
    s = SESSIONS[sid]
    s.pairs.clear()
    MS.generator.reset_history()
    return {"session_id": sid, "status": "reset"}

@app.post("/v1/responses")
def create_response(req: ResponsesRequest, _: str = Depends(check_basic_auth)):
    s = _get_or_create_session(req.session_id)
    system_prompt, latest_user, pairs = _messages_to_pairs(req.input)
    if pairs:
        s.pairs.extend(pairs)
    if not latest_user:
        raise HTTPException(status_code=400, detail="missing user message")
    prompt = _build_prompt(s, latest_user, system_prompt)
    prompt_ids = _encode(MS.enc, prompt, MS.device)
    if req.temperature is not None:
        MS.generator.set_temperature(req.temperature)
    if req.repetition_penalty is not None:
        MS.generator.set_repetition_penalty(req.repetition_penalty)
    if req.top_k is not None:
        MS.generator.top_k = req.top_k
    if req.top_p is not None:
        MS.generator.top_p = req.top_p

    def run_generation() -> Tuple[str, Dict[str,int], Dict[str,float]]:
        MS.generator.reset_history()
        with torch.inference_mode():
            it = MS.generator.generate(prompt_ids, req.max_output_tokens)
            tokens: List[int] = []
            try:
                while True:
                    t = next(it)
                    tokens.append(int(t))
            except StopIteration as e:
                out_ids, prefill_dur, step_dur = e.value
        usage = _usage(out_ids, prompt_len=prompt_ids.numel())
        completion_ids = out_ids[-usage["completion_tokens"]:] if usage["completion_tokens"] > 0 else torch.tensor([], device=out_ids.device, dtype=torch.long)
        text = MS.enc.decode([int(x) for x in completion_ids.tolist()])
        meta = {"prefill_ms": int(prefill_dur*1000), "gen_ms": int(step_dur*1000)}
        return text, usage, meta

    if req.stream:
        def event_stream() -> Iterable[bytes]:
            MS.generator.reset_history()
            with torch.inference_mode():
                it = MS.generator.generate(prompt_ids, req.max_output_tokens)
                usage = None
                meta = None
                delta_count = 0
                try:
                    while True:
                        t = next(it)
                        piece = _decode(MS.enc, int(t))
                        payload = {"type":"response.output_text.delta","delta":piece,"index":0}
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
                        delta_count += 1
                except StopIteration as e:
                    out_ids, prefill_dur, step_dur = e.value
                    usage = _usage(out_ids, prompt_len=prompt_ids.numel())
                    meta = {"prefill_ms": int(prefill_dur*1000), "gen_ms": int(step_dur*1000)}
                    comp_len = usage["completion_tokens"]
                    if comp_len > 0:
                        completion_ids = out_ids[-comp_len:]
                        final_text = MS.enc.decode([int(x) for x in completion_ids.tolist()])
                    else:
                        final_text = ""
                    s.pairs.append((latest_user, final_text))
                rid = f"resp_{uuid.uuid4().hex}"
                completed = {
                    "type":"response.completed",
                    "response": {
                        "id": rid,
                        "object": "response",
                        "created": int(time.time()),
                        "model": req.model,
                        "usage": usage,
                        "output": [{"type":"message","role":"assistant","content":[{"type":"output_text","text": final_text}]}],
                        "meta": meta,
                        "session_id": s.id
                    }
                }
                yield f"data: {json.dumps(completed, ensure_ascii=False)}\n\n".encode("utf-8")
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        text, usage, meta = run_generation()
        s.pairs.append((latest_user, text))
        rid = f"resp_{uuid.uuid4().hex}"
        body = {
            "id": rid,
            "object": "response",
            "created": int(time.time()),
            "model": req.model,
            "output": [{"type":"message","role":"assistant","content":[{"type":"output_text","text": text}]}],
            "usage": usage,
            "meta": meta,
            "session_id": s.id,
        }
        return JSONResponse(body)

# app.py (excerpt)

@app.get("/v1/completions/models")
def list_completion_models(_: str = Depends(check_basic_auth)):
    global MS
    if MS is None:
        raise HTTPException(status_code=503, detail="model not initialized")
    model_id = os.getenv("MODEL_ID", "daisy")
    ctx = int(
        (MS.hparams.get("train_attention_window_len")
         or MS.hparams.get("attention_window_len")
         or 2048)
    )
    payload = {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "context_window": ctx,
                "tokenizer": "gpt2",
                "modalities": ["text"],
                "endpoints": ["v1/completions", "v1/responses"],
                "capabilities": {
                    "streaming": True,
                    "system_prompt": False,
                    "tool_calls": False
                }
            }
        ]
    }
    return JSONResponse(payload)
