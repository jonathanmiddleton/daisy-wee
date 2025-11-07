
import os, sys, json, time
import requests

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")
USER = os.getenv("BASIC_AUTH_USER", "user")
PASS = os.getenv("BASIC_AUTH_PASS", "pass")

def non_stream_example():
    url = f"{BASE}/v1/responses"
    payload = {
        "model": "daisy",
        "input": [
            {"role":"system","content":[{"type":"input_text","text":"You are a helpful assistant."}]},
            {"role":"user","content":[{"type":"input_text","text":"List three prime numbers greater than 10."}]}
        ],
        "max_output_tokens": 64,
        "stream": False
    }
    r = requests.post(url, auth=(USER, PASS), json=payload, timeout=600)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))

def stream_example():
    url = f"{BASE}/v1/responses"
    payload = {
        "model": "daisy",
        "input": [
            {"role":"user","content":[{"type":"input_text","text":"Write a short haiku about the Hudson River."}]}
        ],
        "max_output_tokens": 64,
        "stream": True
    }
    with requests.post(url, auth=(USER, PASS), json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = json.loads(line[len("data: "):])
                t = data.get("type", "")
                if t.endswith(".delta"):
                    print(data["delta"], end="", flush=True)
                elif t == "response.completed":
                    print("\n\n[completed]")
                    print(json.dumps(data["response"], indent=2, ensure_ascii=False))

def list_models():
    url = f"{BASE}/v1/completions/models"
    r = requests.get(url, auth=(USER, PASS), timeout=60)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "nonstream"
    if which == "stream":
        stream_example()
    else:
        non_stream_example()
