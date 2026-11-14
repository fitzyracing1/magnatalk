"""Job spec: the small YAML/JSON document a project declares for each job."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .catalog import KNOWN_PRECISIONS

TASK_TYPES = ("inference", "fine_tune", "training", "signal_processing")

# Bytes per parameter for model weights at each precision.
BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "fp4": 0.5,
    "int8": 1.0,
    "int4": 0.5,
}

# Multiplier applied on top of weight memory to estimate total VRAM.
# These are deliberately simple, documented heuristics -- override them by
# setting `vram_gb` explicitly in the job spec.
#   inference:  weights + KV cache / activation headroom
#   fine_tune:  parameter-efficient fine-tuning (LoRA/QLoRA-style); full
#               fine-tuning with Adam needs far more -- set vram_gb yourself
#   training:   small-model training with optimizer state and activations
TASK_VRAM_FACTOR = {
    "inference": 1.2,
    "fine_tune": 2.5,
    "training": 4.0,
    "signal_processing": 1.0,
}

DEFAULT_SIGNAL_PROCESSING_VRAM_GB = 8.0


class JobSpecError(ValueError):
    """Raised when a job spec is malformed."""


@dataclass(frozen=True)
class JobSpec:
    task: str
    name: str = "unnamed-job"
    model_params_b: float | None = None
    vram_gb: float | None = None
    precision: str = "fp16"
    budget_per_hour_usd: float | None = None
    budget_per_job_usd: float | None = None
    max_runtime_hours: float | None = None
    preferred_providers: tuple[str, ...] = field(default_factory=tuple)
    excluded_providers: tuple[str, ...] = field(default_factory=tuple)

    def required_vram_gb(self) -> float:
        """Estimated VRAM the job needs, in GB."""
        if self.vram_gb is not None:
            return float(self.vram_gb)
        if self.model_params_b is not None:
            weights = self.model_params_b * BYTES_PER_PARAM[self.precision]
            return round(weights * TASK_VRAM_FACTOR[self.task], 1)
        if self.task == "signal_processing":
            return DEFAULT_SIGNAL_PROCESSING_VRAM_GB
        raise JobSpecError(
            f"job '{self.name}': set either 'vram_gb' or 'model_params_b' "
            "so the required memory can be determined"
        )

    def effective_budget_per_hour(self) -> float | None:
        """Hourly budget cap, deriving one from budget_per_job / max_runtime if needed."""
        caps = []
        if self.budget_per_hour_usd is not None:
            caps.append(self.budget_per_hour_usd)
        if self.budget_per_job_usd is not None and self.max_runtime_hours:
            caps.append(self.budget_per_job_usd / self.max_runtime_hours)
        return min(caps) if caps else None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "task": self.task,
            "model_params_b": self.model_params_b,
            "vram_gb": self.vram_gb,
            "precision": self.precision,
            "budget_per_hour_usd": self.budget_per_hour_usd,
            "budget_per_job_usd": self.budget_per_job_usd,
            "max_runtime_hours": self.max_runtime_hours,
            "preferred_providers": list(self.preferred_providers),
            "excluded_providers": list(self.excluded_providers),
        }


def _positive_or_none(d: dict, key: str, name: str) -> float | None:
    v = d.get(key)
    if v is None:
        return None
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
        raise JobSpecError(f"job '{name}': '{key}' must be a positive number")
    return float(v)


def _provider_list(d: dict, key: str, name: str) -> tuple[str, ...]:
    v = d.get(key, [])
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise JobSpecError(f"job '{name}': '{key}' must be a list of provider names")
    return tuple(p.lower() for p in v)


def job_from_dict(d: dict) -> JobSpec:
    if not isinstance(d, dict):
        raise JobSpecError("job spec must be a mapping")
    name = d.get("name", "unnamed-job")
    task = d.get("task")
    if task not in TASK_TYPES:
        raise JobSpecError(f"job '{name}': 'task' must be one of {TASK_TYPES}, got {task!r}")
    precision = d.get("precision", "fp16")
    if precision not in KNOWN_PRECISIONS:
        raise JobSpecError(
            f"job '{name}': unknown precision {precision!r}; known: {sorted(KNOWN_PRECISIONS)}"
        )
    spec = JobSpec(
        task=task,
        name=str(name),
        model_params_b=_positive_or_none(d, "model_params_b", name),
        vram_gb=_positive_or_none(d, "vram_gb", name),
        precision=precision,
        budget_per_hour_usd=_positive_or_none(d, "budget_per_hour_usd", name),
        budget_per_job_usd=_positive_or_none(d, "budget_per_job_usd", name),
        max_runtime_hours=_positive_or_none(d, "max_runtime_hours", name),
        preferred_providers=_provider_list(d, "preferred_providers", name),
        excluded_providers=_provider_list(d, "excluded_providers", name),
    )
    spec.required_vram_gb()  # fail fast if memory needs cannot be determined
    return spec


def load_job(path: str | Path) -> JobSpec:
    """Load a job spec from a YAML or JSON file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = json.loads(text) if p.suffix == ".json" else yaml.safe_load(text)
    return job_from_dict(data)
