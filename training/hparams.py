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
    # List of evaluation datasets with descriptive type and path glob
    val_shards: list[dict]
    training_sequence_length: int
    val_seq_len: int
    target_tokens: int
    cooldown_frac: float
    # Learning rate schedule selection
    learning_rate_schedule: str  # {'linear_decay','linear_warmup_cosine_decay','constant_with_cosine_decay'}
    train_attention_window_len: int  # training-time sliding attention window (<= model spec max)
    window_block_size: int  # block granularity for sliding window/masks (from spec)
    # Common fields with defaults
    vocab_size: int
    eos_token_id: int
    tot_val_tokens: int  # how many tokens of validation data
    val_loss_every_tokens: int  # num tokens between validation passes (0 disables)
    checkpoint_warmup_tokens: int  # tokens to skip before taking checkpoints
    checkpoint_per_n_tokens: int  # interval in tokens between checkpoints (0 = every update after warmup)
    save_checkpoint: bool
    num_layers: int
    num_heads: int
    model_dim: int
    head_dim: int
    optimizers: list[dict]
    # Force full attention windows (useful when resuming after smaller windows)
    full_windows: bool
    # Model context size (from ModelSpec); used to instantiate DaisyCore
    max_seq_len: int
    # Gradient accumulation
    grad_acc_steps: int
    # Model selection
    model_spec: str    # name of model spec under model_specs/, or a path to a spec file
    model_class: str  # fully-qualified class name, e.g., 'models.daisy.daisy_core.DaisyCore'
    # Torch compile/tuning flags
    torch_coordinate_descent_tuning: bool = False
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

    # Normalize/alias keys that use dots for namespacing
    if "torch.coordinate_descent_tuning" in cfg_dict and "torch_coordinate_descent_tuning" not in cfg_dict:
        cfg_dict["torch_coordinate_descent_tuning"] = cfg_dict.pop("torch.coordinate_descent_tuning")

    # Backward compatibility: alias old snapshot_* keys to checkpoint_* keys
    if "snapshot_warmup_tokens" in cfg_dict and "checkpoint_warmup_tokens" not in cfg_dict:
        cfg_dict["checkpoint_warmup_tokens"] = cfg_dict.pop("snapshot_warmup_tokens")
    if "snapshot_per_n_tokens" in cfg_dict and "checkpoint_per_n_tokens" not in cfg_dict:
        cfg_dict["checkpoint_per_n_tokens"] = cfg_dict.pop("snapshot_per_n_tokens")

    # If a model_spec name/path is provided, load the spec and merge recognized fields
    model_spec_name = cfg_dict.get("model_spec")
    spec_dict = None
    if model_spec_name:
        from dataclasses import asdict
        spec_dict = asdict(load_model_spec(str(model_spec_name))) # returns a dict
        # Merge ModelSpec fields into cfg. For window_block_size we ALWAYS take the ModelSpec value.
        valid_names_for_merge = {f.name for f in dataclass_fields(Hyperparameters)}
        for k, v in spec_dict.items():
            if k not in valid_names_for_merge:
                continue
            if k == "window_block_size":
                cfg_dict[k] = v
            elif k not in cfg_dict or cfg_dict[k] is None:
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

    # Normalize and validate new multi-eval schema for val_shards (no backward compatibility)
    vcfg = cfg_dict.get("val_shards")
    if not isinstance(vcfg, list) or len(vcfg) == 0:
        raise ValueError("val_shards must be a non-empty list of objects with 'path' and optional 'type' fields")
    norm_list: list[dict] = []
    for i, item in enumerate(vcfg, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"val_shards[{i}] must be a mapping with keys: path (str), type (str, optional)")
        path = item.get("path")
        vtype = item.get("type")
        if not isinstance(path, str) or not path:
            raise ValueError(f"val_shards[{i}].path must be a non-empty string")
        if vtype is not None and not isinstance(vtype, str):
            raise ValueError(f"val_shards[{i}].type must be a string when provided")
        if vtype is None:
            vtype = f"val{i}"
        norm_list.append({"type": vtype, "path": path})
    cfg_dict["val_shards"] = norm_list

    args = Hyperparameters(**cfg_dict)

    # Additional validations per spec
    try:
        tsl = int(args.training_sequence_length)
        tawt = int(args.train_attention_window_len)
        wbs = int(args.window_block_size)
        gas = int(args.grad_acc_steps)
        cdf = float(args.cooldown_frac)
        vlet = int(args.val_loss_every_tokens)
        spnt = int(args.checkpoint_per_n_tokens)
        swt = int(args.checkpoint_warmup_tokens)
    except Exception as e:
        raise ValueError(f"Invalid types in Hyperparameters: {e}")
    if tsl % wbs != 0:
        raise ValueError(f"training_sequence_length ({tsl}) must be divisible by window_block_size ({wbs})")
    if tawt % wbs != 0:
        raise ValueError(f"train_attention_window_len ({tawt}) must be divisible by window_block_size ({wbs})")
    if tsl < tawt:
        raise ValueError(f"training_sequence_length ({tsl}) must be >= train_attention_window_len ({tawt})")
    # Enforce training window <= model's supported max if spec is available
    if spec_dict is not None:
        spec_max = int(spec_dict["attention_window_len"])  # model's maximum window
        if tawt > spec_max:
            raise ValueError(
                f"train_attention_window_len ({tawt}) must be <= model_spec.attention_window_len ({spec_max})"
            )
    if gas < 1:
        raise ValueError(f"grad_acc_steps must be >= 1, got {gas}")
    if not (0.0 <= cdf <= 1.0):
        raise ValueError(f"cooldown_frac must be in [0,1], got {cdf}")
    if vlet < 0:
        raise ValueError(f"val_loss_every_tokens must be >= 0, got {vlet}")
    if spnt < 0:
        raise ValueError(f"checkpoint_per_n_tokens must be >= 0, got {spnt}")
    if swt < 0:
        raise ValueError(f"checkpoint_warmup_tokens must be >= 0, got {swt}")
    # Validate learning rate schedule option
    # Validate learning rate schedule option against optim dispatch table
    try:
        from training.optim import LEARNING_RATE_SCHEDULES
        valid_schedules = set(LEARNING_RATE_SCHEDULES.keys())
    except Exception:
        valid_schedules = {"linear_decay", "linear_warmup_cosine_decay", "constant_with_cosine_decay"}
    if args.learning_rate_schedule not in valid_schedules:
        raise ValueError(
            f"learning_rate_schedule must be one of {sorted(valid_schedules)}, got '{args.learning_rate_schedule}'"
        )

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
        k = k.strip().replace("-", "_").replace(".", "_")
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
