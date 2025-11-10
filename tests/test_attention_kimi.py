import pytest
import torch
from einops import rearrange

# adjust this import if your file/module name is different
from models.daisy.attention_kimi import KimiLinearSelfAttention

try:
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
    FLA_AVAILABLE = True
except Exception:
    FLA_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()


def device():
    if CUDA_AVAILABLE:
        return torch.device("cuda")
    return torch.device("cpu")


def tol(dtype):
    if dtype == torch.float32:
        return dict(rtol=1e-4, atol=2e-4)
    if dtype == torch.bfloat16:
        return dict(rtol=3e-2, atol=3e-2)
    if dtype == torch.float16:
        return dict(rtol=3e-3, atol=3e-3)
    return dict(rtol=1e-4, atol=2e-4)


def rand_inputs(b, s, d, dev, dt):
    g = torch.Generator(device=dev.type)
    g.manual_seed(0)
    return torch.randn(b, s, d, device=dev, dtype=dt, generator=g)


def make_model(dim=256, heads=8, head_dim=32, expand_v=1.0, use_short_conv=True, mode="chunk"):
    return KimiLinearSelfAttention(
        dim=dim,
        num_heads=heads,
        max_seq_len=4096,
        head_dim=head_dim,
        expand_v=expand_v,
        mode=mode,
        use_short_conv=use_short_conv,
        allow_neg_eigval=False,
        layer_idx=0,
    )


@pytest.mark.skipif(not (CUDA_AVAILABLE and FLA_AVAILABLE), reason="requires CUDA + FLA")
@pytest.mark.parametrize("expand_v", [1.0, 2.0])
@pytest.mark.parametrize("use_short_conv", [True, False])
def test_chunk_vs_fused_equivalence_fp32(expand_v, use_short_conv):
    dev = device()
    dt = torch.float32
    B, T, H, Dh = 2, 48, 8, 32
    D = H * Dh
    x = rand_inputs(B, T, D, dev, dt)

    m = make_model(dim=D, heads=H, head_dim=Dh, expand_v=expand_v, use_short_conv=use_short_conv).to(dev)

    m.train()
    y_chunk = m.forward(x, ve=None, sa_lambdas=None, attn_mask=None)

    m.eval()
    y_fused = m.forward(x, ve=None, sa_lambdas=None, attn_mask=None)

    tt = tol(dt)
    assert torch.allclose(y_chunk, y_fused, **tt), (y_chunk - y_fused).abs().max().item()


@pytest.mark.skipif(not (CUDA_AVAILABLE and FLA_AVAILABLE), reason="requires CUDA + FLA")
@pytest.mark.parametrize("expand_v", [1.0, 2.0])
def test_streaming_step_equals_forward(expand_v):
    dev = device()
    dt = torch.float32
    B, T, H, Dh = 2, 40, 8, 32
    D = H * Dh
    x = rand_inputs(B, T, D, dev, dt)

    m = make_model(dim=D, heads=H, head_dim=Dh, expand_v=expand_v, use_short_conv=True).to(dev)
    m.eval()
    y_full = m.forward(x, ve=None, sa_lambdas=None, attn_mask=None)

    m.reset_history()
    ys = []
    for t in range(T):
        y_t, _, _ = m.step(x[:, t : t + 1], k_ctx=None, v_ctx=None, pos=t, ve=None, sa_lambdas=None, window=None)
        ys.append(y_t)
    y_stream = torch.cat(ys, dim=1)

    tt = tol(dt)
    assert torch.allclose(y_full, y_stream, **tt), (y_full - y_stream).abs().max().item()


@pytest.mark.skipif(not (CUDA_AVAILABLE and FLA_AVAILABLE), reason="requires CUDA + FLA")
def test_prefill_matches_forward_and_seen():
    dev = device()
    dt = torch.float32
    B, T, H, Dh = 2, 32, 8, 32
    D = H * Dh
    x = rand_inputs(B, T, D, dev, dt)

    m = make_model(dim=D, heads=H, head_dim=Dh).to(dev)
    m.eval()
    y_prefill, _, _ = m.prefill(x, ve=None, sa_lambdas=None, attn_mask=None)
    assert m._seen == T
    y_forward = m.forward(x, ve=None, sa_lambdas=None, attn_mask=None)

    tt = tol(dt)
    assert torch.allclose(y_prefill, y_forward, **tt), (y_prefill - y_forward).abs().max().item()

    m.reset_history()
    assert m._seen == 0


@pytest.mark.skipif(not (CUDA_AVAILABLE and FLA_AVAILABLE), reason="requires CUDA + FLA")
def test_ve_mixing_identity():
    dev = device()
    dt = torch.float32
    B, T, H, Dh = 2, 32, 8, 32
    D = H * Dh
    x = rand_inputs(B, T, D, dev, dt)

    m = make_model(dim=D, heads=H, head_dim=Dh).to(dev)
    m.eval()

    l0 = torch.tensor([1.0, 0.0], device=dev, dtype=dt)
    l1 = torch.tensor([0.0, 1.0], device=dev, dtype=dt)

    y_from_v = m.forward(x, ve=None, sa_lambdas=l0, attn_mask=None)
    y_from_ve = m.forward(x, ve=x, sa_lambdas=l1, attn_mask=None)

    tt = tol(dt)
    assert torch.allclose(y_from_v, y_from_ve, **tt), (y_from_v - y_from_ve).abs().max().item()


@pytest.mark.skipif(not (CUDA_AVAILABLE and FLA_AVAILABLE), reason="requires CUDA + FLA")
def test_parameter_shapes_and_broadcasting():
    dev = device()
    m = make_model().to(dev)
    assert list(m.A_log.shape) == [1, 1, m.num_heads, 1]
    assert m.dt_bias.shape[0] == m.key_dim


@pytest.mark.skipif(not (CUDA_AVAILABLE and FLA_AVAILABLE and torch.cuda.is_bf16_supported()), reason="requires CUDA bf16 + FLA")
def test_bfloat16_chunk_vs_fused():
    dev = device()
    dt = torch.bfloat16
    B, T, H, Dh = 2, 40, 8, 32
    D = H * Dh
    x = rand_inputs(B, T, D, dev, dt)

    m = make_model(dim=D, heads=H, head_dim=Dh).to(dev).to(dtype=dt)

    m.train()
    y_chunk = m.forward(x, ve=None, sa_lambdas=None, attn_mask=None)

    m.eval()
    y_fused = m.forward(x, ve=None, sa_lambdas=None, attn_mask=None)

    tt = tol(dt)
    assert torch.allclose(y_chunk, y_fused, **tt), (y_chunk - y_fused).abs().max().item()


@pytest.mark.skipif(not (CUDA_AVAILABLE and FLA_AVAILABLE), reason="requires CUDA + FLA")
def test_varlen_mask_equivalence_or_skip():
    dev = device()
    dt = torch.float32
    B, H, Dh = 2, 8, 32
    D = H * Dh
    T = 48
    lens = torch.tensor([T - 5, T], device=dev)
    x = rand_inputs(B, T, D, dev, dt)

    mask = torch.zeros(B, T, device=dev, dtype=torch.long)
    for b in range(B):
        mask[b, : lens[b].item()] = 1

    m = make_model(dim=D, heads=H, head_dim=Dh).to(dev)
    m.eval()

    try:
        y_masked = m.forward(x, ve=None, sa_lambdas=None, attn_mask=mask)
    except AssertionError as e:
        if "varlen requires flash-linear-attention install" in str(e):
            pytest.skip("varlen path not available in this build")
        raise

    y_expected = torch.zeros_like(y_masked)
    for b in range(B):
        m.reset_history()
        xb = x[b : b + 1, : lens[b]]
        yb = m.forward(xb, ve=None, sa_lambdas=None, attn_mask=None)
        y_expected[b, : lens[b]] = yb

    tt = tol(dt)
    assert torch.allclose(y_masked, y_expected, **tt), (y_masked - y_expected).abs().max().item()


@pytest.mark.skipif(not (CUDA_AVAILABLE and FLA_AVAILABLE), reason="requires CUDA + FLA")
def test_whitebox_against_direct_kernel_path():
    dev = device()
    dt = torch.float32
    B, T, H, Dh = 2, 32, 8, 32
    D = H * Dh
    x = rand_inputs(B, T, D, dev, dt)

    m = make_model(dim=D, heads=H, head_dim=Dh, expand_v=1.0, use_short_conv=True).to(dev)
    m.eval()

    if m.use_short_conv:
        q, _ = m.q_conv1d(x=m.q_proj(x), cache=None, output_final_state=False, cu_seqlens=None)
        k, _ = m.k_conv1d(x=m.k_proj(x), cache=None, output_final_state=False, cu_seqlens=None)
        v, _ = m.v_conv1d(x=m.v_proj(x), cache=None, output_final_state=False, cu_seqlens=None)
    else:
        q = torch.nn.functional.silu(m.q_proj(x))
        k = torch.nn.functional.silu(m.k_proj(x))
        v = torch.nn.functional.silu(m.v_proj(x))

    g = m.f_proj(x)
    g = fused_kda_gate(g, m.A_log, m.head_k_dim, g_bias=m.dt_bias)
    beta = m.b_proj(x).float().sigmoid()

    q = rearrange(q, "b t (h d) -> b t h d", d=m.head_k_dim)
    k = rearrange(k, "b t (h d) -> b t h d", d=m.head_k_dim)
    v = rearrange(v, "b t (h d) -> b t h d", d=int(m.head_v_dim))

    o_kernel, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
    )

    o_kernel = m.o_norm(o_kernel, rearrange(m.g_proj(x), "b t (h d) -> b t h d", d=int(m.head_v_dim)))
    o_kernel = rearrange(o_kernel, "b t h d -> b t (h d)")
    o_kernel = m.o_proj(o_kernel)

    y = m.forward(x, ve=None, sa_lambdas=None, attn_mask=None)

    tt = tol(dt)
    assert torch.allclose(o_kernel, y, **tt), (o_kernel - y).abs().max().item()

@pytest.mark.skipif(not (CUDA_AVAILABLE and FLA_AVAILABLE), reason="requires CUDA + FLA")
def test_ve_mixing_identity_headshaped():
    dev = device()
    dt = torch.float32
    B, T, H, Dh = 2, 16, 8, 32
    D = H * Dh
    x = rand_inputs(B, T, D, dev, dt)

    m = make_model(dim=D, heads=H, head_dim=Dh).to(dev).eval()
    # build head-shaped ve with last dim = head_v_dim
    with torch.no_grad():
        v_proj = m.v_proj(x)
        if m.use_short_conv:
            v_proj, _ = m.v_conv1d(x=v_proj, cache=None, output_final_state=False, cu_seqlens=None)
    ve_head = rearrange(v_proj, "b t (h d) -> b t h d", h=H)

    l0 = torch.tensor([1.0, 0.0], device=dev, dtype=dt)
    l1 = torch.tensor([0.0, 1.0], device=dev, dtype=dt)

    y_from_v  = m.forward(x, ve=None,     sa_lambdas=l0, attn_mask=None)
    y_from_ve = m.forward(x, ve=ve_head,  sa_lambdas=l1, attn_mask=None)

    tt = tol(dt)
    assert torch.allclose(y_from_v, y_from_ve, **tt)
