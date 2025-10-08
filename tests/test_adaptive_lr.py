import math
import pytest

from training.optim import AdaptiveLR


@pytest.mark.cpu
def test_lr_scale_base_main_phase_and_bounds():
    # Before cooldown: always 1.0
    assert AdaptiveLR._lr_scale_base(s=0.0, cooldown_frac=0.4) == pytest.approx(1.0)
    assert AdaptiveLR._lr_scale_base(s=0.59, cooldown_frac=0.4) == pytest.approx(1.0)
    # Clamp s below 0 and above 1
    assert AdaptiveLR._lr_scale_base(s=-1.0, cooldown_frac=0.4) == pytest.approx(1.0)
    # When cooldown fraction is non-positive, always 1.0 irrespective of s
    assert AdaptiveLR._lr_scale_base(s=0.75, cooldown_frac=0.0) == pytest.approx(1.0)
    assert AdaptiveLR._lr_scale_base(s=1.25, cooldown_frac=-0.5) == pytest.approx(1.0)


@pytest.mark.cpu
def test_lr_scale_base_full_cosine_cooldown():
    # Cooldown is the last 20% of training and uses full cosine (k=1.0)
    cf = 0.2
    k = 1.0
    # Start of cooldown: t=0 -> value 1.0
    assert AdaptiveLR._lr_scale_base(s=1 - cf, cooldown_frac=cf, cosine_frac=k) == pytest.approx(1.0)
    # Mid cooldown: t=0.5 -> 0.5 * (1 + cos(pi*0.5)) = 0.5
    mid_s = (1 - cf) + 0.5 * cf
    expected_mid = 0.5 * (1.0 + math.cos(math.pi * 0.5))
    assert AdaptiveLR._lr_scale_base(s=mid_s, cooldown_frac=cf, cosine_frac=k) == pytest.approx(expected_mid)
    # End of cooldown: t=1 -> 0.0
    assert AdaptiveLR._lr_scale_base(s=1.0, cooldown_frac=cf, cosine_frac=k) == pytest.approx(0.0)


@pytest.mark.cpu
def test_lr_scale_base_piecewise_cosine_then_linear():
    # Cooldown last 40%, with cosine over first half of cooldown (k=0.5), then linear to zero
    cf = 0.4
    k = 0.5
    start = 1 - cf

    # Start (t=0): value = 1.0
    assert AdaptiveLR._lr_scale_base(s=start, cooldown_frac=cf, cosine_frac=k) == pytest.approx(1.0)

    # At t=k=0.5: yk = 0.5
    s_at_k = start + k * cf
    yk = 0.5 * (1.0 + math.cos(math.pi * k))
    assert yk == pytest.approx(0.5)
    assert AdaptiveLR._lr_scale_base(s=s_at_k, cooldown_frac=cf, cosine_frac=k) == pytest.approx(yk)

    # In linear region, e.g., t=0.75 -> y = yk * (1 - (t-k)/(1-k))
    t = 0.75
    s_at_t = start + t * cf
    expected = yk * (1.0 - (t - k) / (1.0 - k))
    assert AdaptiveLR._lr_scale_base(s=s_at_t, cooldown_frac=cf, cosine_frac=k) == pytest.approx(expected)

    # End (t=1): value -> 0.0
    assert AdaptiveLR._lr_scale_base(s=1.0, cooldown_frac=cf, cosine_frac=k) == pytest.approx(0.0)


@pytest.mark.cpu
def test_maybe_adapt_reduce_m_two_triggers():
    # Set guards low so adaptation can trigger immediately
    lr = AdaptiveLR(H_eval=1, H_guard=1)
    # Sanity
    assert lr.m == pytest.approx(1.0)

    # Two or more triggers: use tau>tau_hi and dnr>dnr_hi and rho1<rho1_lo
    lr._maybe_adapt(tau=lr.tau_hi + 1e-3, rho1=lr.rho1_lo - 1e-3, dnr=lr.dnr_hi + 1e-3, delta_steps=1)

    assert lr.m == pytest.approx(max(1.0 * 0.70, lr.m_min))
    assert lr.since_change == 0
    assert lr.stable_accum == 0
    assert lr.eval_accum == 0


@pytest.mark.cpu
def test_maybe_adapt_reduce_m_one_trigger_with_tau_gate():
    lr = AdaptiveLR(H_eval=1, H_guard=1)
    # First, ensure m not already at min due to previous state; start fresh
    assert lr.m == pytest.approx(1.0)

    # Exactly one trigger (e.g., only tau), and tau>0.045 additional gate satisfied
    lr._maybe_adapt(tau=max(lr.tau_hi, 0.05), rho1=0.0, dnr=0.0, delta_steps=1)

    assert lr.m == pytest.approx(max(1.0 * 0.85, lr.m_min))
    assert lr.since_change == 0
    assert lr.stable_accum == 0
    assert lr.eval_accum == 0


@pytest.mark.cpu
def test_maybe_adapt_increase_m_under_stability():
    # Use small H_stable so increase happens after a couple of evals; no guard
    lr = AdaptiveLR(H_eval=1, H_guard=0, H_stable=2)
    # Start below 1.0 so increase is observable
    lr.m = 0.5

    # Stable conditions: tau < tau_lo, rho1 > rho1_hi, dnr < dnr_lo
    for _ in range(2):
        lr._maybe_adapt(tau=lr.tau_lo - 1e-6, rho1=lr.rho1_hi + 1e-6, dnr=lr.dnr_lo - 1e-6, delta_steps=1)

    # After accumulating stability for H_stable, m increases by 1% but capped at 1.0
    assert lr.m == pytest.approx(min(1.0, 0.5 * 1.01))
    assert lr.since_change == 0  # reset after increase
    assert lr.stable_accum == 0


@pytest.mark.cpu
def test_get_lr_combines_base_and_m_without_adapting_when_guard_blocks():
    # Default guard is high; ensures _maybe_adapt won't change m during this call
    lr = AdaptiveLR()
    # Manually set m to a known value
    lr.m = 0.8

    # Choose s within cooldown; with cosine_frac=1, base is cosine value at t=0.5 -> 0.5
    cooldown_frac = 0.5
    s = (1 - cooldown_frac) + 0.5 * cooldown_frac  # mid cooldown

    # Arbitrary step and loss; guards prevent adaptation
    out = lr.get_lr(step=10, loss=1.23, s=s, cooldown_frac=cooldown_frac)

    assert out == pytest.approx(0.5 * 0.8)

    # Also confirm subsequent call with same step does not error and still returns base*m (delta clamped to >=1)
    out2 = lr.get_lr(step=10, loss=1.5, s=s, cooldown_frac=cooldown_frac)
    assert out2 == pytest.approx(0.5 * 0.8)
