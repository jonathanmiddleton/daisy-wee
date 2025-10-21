import torch
import pytest

from training.optim import build_optimizers_from_cfg, derive_named_param_groups
from models.daisy.daisy_core import DaisyCore


def make_small_daisy_core():
    # very small model for fast tests; num_layers must be even per DaisyCore assertions
    model = DaisyCore(
        vocab_size=256,
        num_layers=2,
        num_heads=2,
        model_dim=32,
        max_seq_len=64,
        head_dim=16,
        window_block_size=8,
        eos_token_id=0,
    )
    # Muon requires bf16 parameters; use bf16 across the board for simplicity
    return model.to(torch.bfloat16)


def build_cfg_per_group():
    # One optimizer instance per group (ensures an optimizer is not applied to multiple groups)
    return [
        {"type": "Muon", "params": [{"group": "hidden_matrix_params"}], "lr": 0.02, "weight_decay": 0.01},
        {"type": "AdamW", "params": [{"group": "embed_params"}], "lr": 3e-4},
        {"type": "AdamW", "params": [{"group": "scalar_params"}], "lr": 1e-3},
        {"type": "AdamW", "params": [{"group": "head_params"}], "lr": 3e-4},
    ]


@pytest.mark.cpu
@torch.no_grad()
def test_build_optimizers_groups_assignment_correct():
    model = make_small_daisy_core()

    cfg = build_cfg_per_group()
    optimizers = build_optimizers_from_cfg(cfg_list=cfg, model=model, rank=0, world_size=1)

    # Expect one optimizer per group
    assert len(optimizers) == 4
    # Each optimizer should only manage a single param group in this config
    for opt in optimizers:
        assert len(opt.param_groups) == 1

    # Verify optimizer type per group and parameter identity matching
    named_groups = derive_named_param_groups(model)

    # Build mapping from group name to optimizer instance
    group_to_opt = {opt.param_groups[0]["name"]: opt for opt in optimizers}

    # Ensure all required groups are present exactly once
    assert set(group_to_opt.keys()) == set(named_groups.keys())

    # Hidden matrices should be handled by Muon
    assert group_to_opt["hidden_matrix_params"].__class__.__name__ == "Muon"
    # Others by AdamW
    for g in ["embed_params", "scalar_params", "head_params"]:
        assert group_to_opt[g].__class__.__name__ == "AdamW"

    # Parameters in each optimizer group should exactly equal the derived canonical groups
    for gname, params in named_groups.items():
        opt_params = list(group_to_opt[gname].param_groups[0]["params"])  # type: ignore[index]
        assert set(map(id, opt_params)) == set(map(id, params)), f"Mismatch for group {gname}"


@pytest.mark.cpu
@torch.no_grad()
def test_no_parameter_in_multiple_optimizer_groups():
    model = make_small_daisy_core()

    cfg = build_cfg_per_group()
    optimizers = build_optimizers_from_cfg(cfg_list=cfg, model=model, rank=0, world_size=1)

    # Count how many times each parameter object appears across all optimizer groups
    counts: dict[int, int] = {}
    for opt in optimizers:
        for pg in opt.param_groups:
            for p in pg["params"]:
                counts[id(p)] = counts.get(id(p), 0) + 1

    # Every parameter must appear exactly once
    assert all(c == 1 for c in counts.values())

    # Sanity check: coverage equals model parameters
    all_params = set(map(id, model.parameters()))
    assert set(counts.keys()) == all_params


@pytest.mark.cpu
@torch.no_grad()
def test_group_does_not_span_multiple_optimizers():
    model = make_small_daisy_core()

    cfg = build_cfg_per_group()
    optimizers = build_optimizers_from_cfg(cfg_list=cfg, model=model, rank=0, world_size=1)

    # Track which optimizer index each group name appears in
    group_to_opt_indices: dict[str, set[int]] = {}
    for i, opt in enumerate(optimizers):
        for pg in opt.param_groups:
            gname = pg.get("name")
            group_to_opt_indices.setdefault(gname, set()).add(i)

    # Each named group should appear in exactly one optimizer (no spanning)
    assert all(len(ixs) == 1 for ixs in group_to_opt_indices.values())

    # Additionally confirm that each optimizer in this test is only applied to one group
    assert all(len(opt.param_groups) == 1 for opt in optimizers)



@pytest.mark.cpu
@torch.no_grad()
def test_frozen_groups_params_are_frozen_and_not_in_optimizers():
    model = make_small_daisy_core()

    cfg = build_cfg_per_group()
    frozen = ["embed_params", "head_params"]

    optimizers = build_optimizers_from_cfg(
        cfg_list=cfg, model=model, rank=0, world_size=1, frozen_groups=frozen
    )

    # Derive canonical groups to identify parameters by group
    named_groups = derive_named_param_groups(model)

    # 1) All parameters in frozen groups should have requires_grad == False
    for g in frozen:
        for p in named_groups[g]:
            assert p.requires_grad is False, f"Parameter in frozen group {g} should be frozen"

    # 2) No frozen parameters should be present in any optimizer's param groups
    frozen_param_ids = {id(p) for g in frozen for p in named_groups[g]}
    for opt in optimizers:
        for pg in opt.param_groups:
            for p in pg["params"]:
                assert id(p) not in frozen_param_ids, "Frozen parameter should not be assigned to any optimizer"

    # 3) The remaining non-frozen groups should still be present exactly once across optimizers
    non_frozen = set(named_groups.keys()) - set(frozen)
    seen_group_names = []
    for opt in optimizers:
        for pg in opt.param_groups:
            seen_group_names.append(pg.get("name"))
    assert set(seen_group_names) == non_frozen

    # 4) Sanity: returned optimizers should be fewer than original (since some groups were frozen)
    assert 0 < len(optimizers) < len(cfg)
