import dataclasses
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import List

import yaml

from model_specs import load_model_spec
from tools.helpers import _coerce_value


@dataclass
class Hyperparameters:
    # Required scenario-specific fields
    train_shards: str
    val_shards: str
    training_sequence_length: int
    val_seq_len: int
    target_tokens: int
    cooldown_frac: float
    attention_window_tokens: int  # new: max backward attention span actually trained (sliding window)
    window_block_size: int  # new: block granularity for sliding window/masks
    # Common fields with defaults
    vocab_size: int
    eos_token_id: int
    val_tokens: int  # how many tokens of validation data
    val_loss_every_tokens: int  # num tokens between validation passes (0 disables)
    snapshot_warmup_tokens: int  # tokens to skip before taking snapshots
    snapshot_per_n_tokens: int  # interval in tokens between snapshots
    save_checkpoint: bool
    num_layers: int
    num_heads: int
    model_dim: int
    head_dim: int
    optimizers: list[dict]
    # Force full attention windows (useful when resuming after smaller windows)
    full_windows: bool
    # Gradient accumulation
    grad_acc_steps: int
    # Model selection
    model_spec: str    # name of model spec under model_specs/, or a path to a spec file
    model_class: str  # fully-qualified class name, e.g., 'models.gpt2.gpt_core.GPT2Core'
    # Weights & Biases minimal logging config
    wandb_log: bool = False
    wandb_project: str = ""
    wandb_run_name: str = ""
    init_checkpoint: str | None = None


def load_hparams_from_yaml(config_path: str) -> Hyperparameters:
    """
    Load Hyperparameters from a YAML file. If a 'model_spec' key is present, also load and merge
    the named spec from model_specs/<name>.yml (or a provided file path). Training config values
    take precedence over spec values. Validates against the Hyperparameters dataclass.
    """
    cfg_dict = {}

    used_path = Path(config_path)
    try:
        with open(used_path, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found: {used_path}") from e

    # If a model_spec name/path is provided, load the spec and merge recognized fields
    model_spec_name = cfg_dict.get("model_spec")
    if model_spec_name:
        spec_dict = load_model_spec(str(model_spec_name))
        # Only merge keys that are valid Hyperparameters fields and not explicitly set in training cfg
        valid_names_for_merge = {f.name for f in dataclass_fields(Hyperparameters)}
        for k, v in (spec_dict.items() if isinstance(spec_dict, dict) else []):
            if k in valid_names_for_merge and (k not in cfg_dict or cfg_dict[k] is None):
                cfg_dict[k] = v

    # Validate keys after potential spec merge
    valid_names = {f.name for f in dataclass_fields(Hyperparameters)}
    unknown = set(cfg_dict) - valid_names
    if unknown:
        raise ValueError(f"Unknown hyperparameter(s) in {used_path}: {sorted(unknown)}")

    required = [f.name for f in dataclass_fields(Hyperparameters)
                if f.default is dataclasses.MISSING and getattr(f, "default_factory",
                                                               dataclasses.MISSING) is dataclasses.MISSING]
    missing = [name for name in required if name not in cfg_dict]
    if missing:
        raise ValueError(f"Missing required hyperparameter(s) in {used_path}: {missing}")

    args = Hyperparameters(**cfg_dict)

    # Additional validations per spec
    try:
        tsl = int(args.training_sequence_length)
        awt = int(args.attention_window_tokens)
        wbs = int(args.window_block_size)
    except Exception as e:
        raise ValueError(f"Invalid types for training_sequence_length/attention_window_tokens/window_block_size: {e}")
    if tsl % wbs != 0:
        raise ValueError(f"training_sequence_length ({tsl}) must be divisible by window_block_size ({wbs})")
    if awt % wbs != 0:
        raise ValueError(f"attention_window_tokens ({awt}) must be divisible by window_block_size ({wbs})")
    if tsl < awt:
        raise ValueError(f"training_sequence_length ({tsl}) must be >= attention_window_tokens ({awt})")

    return args


def apply_cli_overrides(args: Hyperparameters, override_args: List[str]) -> Hyperparameters:
    """
    Apply CLI overrides of the form --key=value or key=value to an existing Hyperparameters object.
    Unknown keys are ignored to allow passing through unrelated CLI args (e.g., torchrun's).
    Values are coerced to the target type using tools.helpers._coerce_value.
    """
    field_map = {f.name: f for f in dataclass_fields(Hyperparameters)}
    for s in override_args:
        if s.startswith("--"):
            s = s[2:]
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip().replace("-", "_")
        if k not in field_map:
            # Ignore unknown keys to allow torchrun args if any; alternatively, raise
            continue
        f = field_map[k]
        coerced = _coerce_value(v.strip(), f.type)
        setattr(args, k, coerced)
    return args


__all__ = [
    "Hyperparameters",
    "load_hparams_from_yaml",
    "apply_cli_overrides",
]
