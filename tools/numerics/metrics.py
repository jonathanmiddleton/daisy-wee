
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import math
import torch
import torch.nn.functional as F


def stable_log_softmax(x: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(x.float(), dim=-1)


def stable_softmax(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x.float(), dim=-1)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float(); b = b.float()
    a_norm = a.norm(p=2, dim=-1)
    b_norm = b.norm(p=2, dim=-1)
    denom = (a_norm * b_norm).clamp_min(1e-12)
    return (a * b).sum(dim=-1) / denom


def kl_divergence(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    # Inputs are log-probs
    p = p_log.exp()
    return (p * (p_log - q_log)).sum(dim=-1)


def jensen_shannon(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    p = p_log.exp(); q = q_log.exp()
    m = 0.5 * (p + q)
    m_log = (m + 1e-12).log()
    return 0.5 * (kl_divergence(p_log, m_log) + kl_divergence(q_log, m_log))


def topk_overlap(logits_a: torch.Tensor, logits_b: torch.Tensor, ks: List[int]) -> List[int]:
    # returns counts of intersection sizes per k
    vals_a, idx_a = torch.sort(logits_a, dim=-1, descending=True)
    vals_b, idx_b = torch.sort(logits_b, dim=-1, descending=True)
    res: List[int] = []
    for k in ks:
        top_a = set(idx_a[..., :k].tolist())
        top_b = set(idx_b[..., :k].tolist())
        res.append(len(top_a.intersection(top_b)))
    return res


def top1_flip(logits_a: torch.Tensor, logits_b: torch.Tensor) -> bool:
    return int(torch.argmax(logits_a) != torch.argmax(logits_b)) == 1


def ref_margin(logits_ref: torch.Tensor) -> float:
    # margin of ref between top1 and top2
    vals, _ = torch.sort(logits_ref, descending=True)
    if vals.numel() < 2:
        return float("nan")
    return float(vals[0] - vals[1])


def delta_nll(logp_ref: torch.Tensor, logp_var: torch.Tensor, y: int) -> float:
    return float(logp_ref[y] - logp_var[y])


def edit_distance(a: List[int], b: List[int]) -> int:
    # Classic Levenshtein distance
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[m]


@dataclass
class CI:
    mean: float
    low: float
    high: float
    median: float


def bootstrap_ci(values: Iterable[float], n_boot: int = 500, conf: float = 0.95, seed: int = 123) -> CI:
    vals = torch.tensor([v for v in values if math.isfinite(v)], dtype=torch.float64)
    if vals.numel() == 0:
        return CI(mean=float("nan"), low=float("nan"), high=float("nan"), median=float("nan"))
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = []
    for _ in range(n_boot):
        idx = torch.randint(0, vals.numel(), (vals.numel(),), generator=g)
        means.append(vals[idx].mean().item())
    means.sort()
    lo_idx = int(((1 - conf) / 2) * n_boot)
    hi_idx = int((1 - (1 - conf) / 2) * n_boot) - 1
    return CI(mean=float(vals.mean().item()), low=float(means[lo_idx]), high=float(means[hi_idx]), median=float(vals.median().item()))


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    if trials <= 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denom = 1 + z**2 / trials
    center = (phat + z**2 / (2*trials)) / denom
    half = z * math.sqrt((phat*(1-phat) + z**2/(4*trials)) / trials) / denom
    return (center - half, center + half)
