# tests/test_generate_step_equals_forward.py
import os
import torch
import pytest

os.environ["DISABLE_O_ZERO_INIT"] = "1"  # ensure attention output proj isn't zeroed

from models.daisy.daisy_core import DaisyCore
from inference.generate import Generator

@pytest.mark.skip(reason="Disabled - requires changing DaisyCore.forward to return logits for testing")
@pytest.mark.parametrize("T,window", [(128, 256), (256, 512)])
def test_step_equals_forward_when_window_covers_all(T, window):
    torch.manual_seed(0)

    vocab_size = 512
    num_layers  = 16   # even, must be >=6 for ve
    num_heads   = 2
    model_dim   = 256
    max_seq_len = 2048

    model = DaisyCore(vocab_size, num_layers, num_heads, model_dim, max_seq_len, model_dim / num_heads).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16)
    with ctx:
        # single-batch prompt
        idx = torch.randint(low=0, high=vocab_size-1, size=(1, T), dtype=torch.long, device=device)

        # forward path (returns [T, padded_vocab]; take last step, trim to vocab_size)
        sliding_window_num_blocks = torch.tensor(1, dtype=torch.int32, device=device)
        logits_f = model(idx.squeeze(0), sliding_window_num_blocks)
        logits_f_last = logits_f[T-1, :vocab_size]

        # incremental path
        gen = Generator(model, window=window, device=device, dtype=torch.bfloat16,
                        temperature=0.0, top_k=None, top_p=None, repetition_penalty=1.0)
        logits_s = gen._prefill(idx[0])  # last-step logits

        assert torch.allclose(logits_s, logits_f_last, atol=1e-5, rtol=1e-5)
