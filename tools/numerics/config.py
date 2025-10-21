
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class DecodingCfg:
    mode_open_loop: Dict[str, Any] = field(default_factory=lambda: {"enabled": True})
    mode_closed_loop: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "max_new_tokens": 256})


@dataclass
class SamplingCfg:
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class ControlsCfg:
    deterministic: bool = True
    # force_fp32_accum_attention: bool = True
    # force_fp32_accum_norms: bool = True
    # kv_cache_dtype: str = "fp16"
    # rope_cache_dtype: str = "fp32"


@dataclass
class DatasetCfg:
    path: Optional[str] = None  # prompts.jsonl; optional if using token shards
    min_prompts: int = 50
    max_prompts: int = 100
    length_buckets: List[int] = field(default_factory=lambda: [64, 512, 2048])
    # Optional: binary shard pattern for DistributedDataGenerator
    shard_pattern: Optional[str] = None


@dataclass
class MetricsCfg:
    topk: List[int] = field(default_factory=lambda: [1, 5, 10])
    margin_bins: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])


@dataclass
class OutputsCfg:
    root: str = "./artifacts_root"
    write_parquet: bool = True


@dataclass
class ReferenceCfg:
    device: str = "cpu"
    dtype: str = "fp32"
    compile: bool = False


@dataclass
class RunConfig:
    run_id: str = "gpt-numerics-v1"
    reference: ReferenceCfg = field(default_factory=ReferenceCfg)
    devices: List[str] = field(default_factory=lambda: ["cpu", "mps"])
    compile_modes: List[bool] = field(default_factory=lambda: [False, True])
    dtype_policies: List[str] = field(default_factory=lambda: ["fp32", "bf16", "fp16", "autocast_bf16"])
    decoding: DecodingCfg = field(default_factory=DecodingCfg)
    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    controls: ControlsCfg = field(default_factory=ControlsCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    metrics: MetricsCfg = field(default_factory=MetricsCfg)
    outputs: OutputsCfg = field(default_factory=OutputsCfg)

    # Model loading
    checkpoint: Optional[str] = None  # path to .pt
    device_override: Optional[str] = None

    @staticmethod
    def from_yaml(path: str) -> "RunConfig":
        d = _read_yaml(path)
        # Utility to map nested dicts to dataclasses
        def reify(cls, key: str):
            sub = d.get(key)
            if isinstance(sub, dict):
                return cls(**sub)
            return cls()

        return RunConfig(
            run_id=d.get("run_id", "gpt-numerics-v1"),
            reference=ReferenceCfg(**(d.get("reference") or {})),
            devices=list(d.get("devices") or ["cpu", "mps"]),
            compile_modes=list(d.get("compile_modes") or [False, True]),
            dtype_policies=list(d.get("dtype_policies") or ["fp32", "bf16", "fp16", "autocast_bf16"]),
            decoding=reify(DecodingCfg, "decoding"),
            sampling=reify(SamplingCfg, "sampling"),
            controls=reify(ControlsCfg, "controls"),
            dataset=reify(DatasetCfg, "dataset"),
            metrics=reify(MetricsCfg, "metrics"),
            outputs=reify(OutputsCfg, "outputs"),
            checkpoint=d.get("checkpoint"),
            device_override=d.get("device_override"),
        )

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
