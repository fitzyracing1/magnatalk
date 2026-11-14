"""The router: scores the chip catalog against a job spec.

Deterministic and explainable by construction:

1. Hard filters remove offers that cannot run the job (memory, precision,
   provider exclusions, budget). Every rejection is recorded with its reason.
2. If preferred providers are given and at least one survives filtering,
   the choice is restricted to them.
3. Remaining (chip, offer) candidates are scored 0..1 from three normalized
   components -- performance (a task-dependent blend of memory bandwidth and
   dense tensor throughput at the job's precision), cost (cheaper is better),
   and fit (prefer the smallest chip that fits comfortably instead of renting
   a huge chip for a tiny job).
4. Ties are broken lexicographically by chip id then provider, so the same
   inputs always produce the same answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import Chip, Offer
from .jobspec import JobSpec

# (perf_weight, cost_weight, fit_weight) per task type.
TASK_WEIGHTS = {
    "inference": (0.40, 0.40, 0.20),
    "fine_tune": (0.35, 0.45, 0.20),
    "training": (0.45, 0.40, 0.15),
    "signal_processing": (0.45, 0.40, 0.15),
}

# Within the perf component: (bandwidth_weight, compute_weight).
# Inference and signal processing are usually memory-bandwidth-bound;
# training and fine-tuning lean more on raw tensor throughput.
PERF_BLEND = {
    "inference": (0.75, 0.25),
    "fine_tune": (0.40, 0.60),
    "training": (0.30, 0.70),
    "signal_processing": (0.85, 0.15),
}

# A chip more than 2x bigger than the job needs starts losing fit score.
COMFORT_HEADROOM = 2.0


class NoFeasibleChipError(RuntimeError):
    def __init__(self, message: str, rejections: list[Rejection]):
        super().__init__(message)
        self.rejections = rejections


@dataclass(frozen=True)
class Rejection:
    chip_id: str
    provider: str
    reason: str


@dataclass(frozen=True)
class RankedOption:
    chip: Chip
    offer: Offer
    score: float
    note: str


@dataclass(frozen=True)
class Decision:
    job: JobSpec
    chip: Chip
    offer: Offer
    score: float
    reason: str
    required_vram_gb: float
    estimated_cost_per_hour_usd: float | None
    estimated_job_cost_usd: float | None
    runners_up: tuple[RankedOption, ...]
    rejections: tuple[Rejection, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "job": self.job.to_dict(),
            "chip": self.chip.id,
            "chip_name": self.chip.name,
            "provider": self.offer.provider,
            "score": round(self.score, 4),
            "reason": self.reason,
            "required_vram_gb": self.required_vram_gb,
            "estimated_cost_per_hour_usd": self.estimated_cost_per_hour_usd,
            "estimated_job_cost_usd": self.estimated_job_cost_usd,
            "price_date": self.offer.price_date,
            "price_source": self.offer.source,
            "runners_up": [
                {
                    "chip": r.chip.id,
                    "chip_name": r.chip.name,
                    "provider": r.offer.provider,
                    "price_per_hour_usd": r.offer.price_per_hour_usd,
                    "score": round(r.score, 4),
                    "note": r.note,
                }
                for r in self.runners_up
            ],
        }


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _fit_score(chip: Chip, required_gb: float) -> float:
    headroom = chip.memory_gb / required_gb
    if headroom <= COMFORT_HEADROOM:
        return 1.0
    return COMFORT_HEADROOM / headroom


def route(job: JobSpec, chips: list[Chip], top_n: int = 3) -> Decision:
    """Pick the best (chip, provider) offer for a job. Raises NoFeasibleChipError."""
    required_gb = job.required_vram_gb()
    budget = job.effective_budget_per_hour()

    rejections: list[Rejection] = []
    candidates: list[tuple[Chip, Offer]] = []

    for chip in chips:
        if not chip.offers:
            rejections.append(Rejection(chip.id, "-", "no rental offers in the catalog"))
            continue
        if job.precision not in chip.supported_precisions:
            for o in chip.offers:
                rejections.append(
                    Rejection(chip.id, o.provider, f"does not support {job.precision}")
                )
            continue
        if chip.memory_gb < required_gb:
            for o in chip.offers:
                rejections.append(
                    Rejection(
                        chip.id, o.provider,
                        f"{chip.memory_gb:g} GB memory < {required_gb:g} GB required",
                    )
                )
            continue
        for o in chip.offers:
            provider = o.provider.lower()
            if provider in job.excluded_providers:
                rejections.append(Rejection(chip.id, o.provider, "provider excluded by job spec"))
                continue
            if budget is not None:
                if o.price_per_hour_usd is None:
                    rejections.append(
                        Rejection(chip.id, o.provider,
                                  "price unknown, cannot verify against budget cap")
                    )
                    continue
                if o.price_per_hour_usd > budget:
                    rejections.append(
                        Rejection(
                            chip.id, o.provider,
                            f"${o.price_per_hour_usd:.2f}/hr exceeds "
                            f"${budget:.2f}/hr budget cap",
                        )
                    )
                    continue
            candidates.append((chip, o))

    preferred_note = ""
    if job.preferred_providers:
        preferred = [
            (c, o) for c, o in candidates if o.provider.lower() in job.preferred_providers
        ]
        if preferred:
            for c, o in candidates:
                if (c, o) not in preferred:
                    rejections.append(
                        Rejection(c.id, o.provider,
                                  "set aside: preferred providers have feasible offers")
                    )
            candidates = preferred
            preferred_note = (
                f" Restricted to preferred providers ({', '.join(job.preferred_providers)})."
            )
        else:
            preferred_note = (
                f" No feasible offer from preferred providers "
                f"({', '.join(job.preferred_providers)}); considered all providers."
            )

    if not candidates:
        lines = [f"  - {r.chip_id} @ {r.provider}: {r.reason}" for r in rejections]
        raise NoFeasibleChipError(
            f"no chip in the catalog can run job '{job.name}' "
            f"(needs {required_gb:g} GB at {job.precision}"
            + (f", <= ${budget:.2f}/hr" if budget is not None else "")
            + "). Rejections:\n" + "\n".join(lines),
            rejections,
        )

    # --- scoring ---
    bw = _minmax([c.memory_bandwidth_gbps for c, _ in candidates])
    throughputs = [c.throughput_tflops(job.precision) or 0.0 for c, _ in candidates]
    comp = _minmax(throughputs)
    prices = [o.price_per_hour_usd for _, o in candidates]
    known = [p for p in prices if p is not None]
    if known:
        lo, hi = min(known), max(known)
        cost = [
            0.0 if p is None else (1.0 if hi - lo < 1e-12 else (hi - p) / (hi - lo))
            for p in prices
        ]
    else:
        cost = [0.0 for _ in prices]

    w_perf, w_cost, w_fit = TASK_WEIGHTS[job.task]
    bw_w, comp_w = PERF_BLEND[job.task]

    scored: list[RankedOption] = []
    for i, (chip, offer) in enumerate(candidates):
        perf = bw_w * bw[i] + comp_w * comp[i]
        fit = _fit_score(chip, required_gb)
        score = w_perf * perf + w_cost * cost[i] + w_fit * fit
        price_txt = (
            f"${offer.price_per_hour_usd:.2f}/hr"
            if offer.price_per_hour_usd is not None
            else "price unknown"
        )
        note = (
            f"{price_txt}; {chip.memory_bandwidth_gbps:g} GB/s; "
            f"{chip.memory_gb:g} GB ({chip.memory_gb / required_gb:.1f}x headroom)"
        )
        scored.append(RankedOption(chip, offer, score, note))

    scored.sort(key=lambda r: (-r.score, r.chip.id, r.offer.provider))
    best, runners = scored[0], tuple(scored[1 : 1 + top_n])

    est_hourly = best.offer.price_per_hour_usd
    est_job = (
        round(est_hourly * job.max_runtime_hours, 2)
        if est_hourly is not None and job.max_runtime_hours
        else None
    )

    reason = _explain(job, best, scored, required_gb, budget, preferred_note)
    return Decision(
        job=job,
        chip=best.chip,
        offer=best.offer,
        score=best.score,
        reason=reason,
        required_vram_gb=required_gb,
        estimated_cost_per_hour_usd=est_hourly,
        estimated_job_cost_usd=est_job,
        runners_up=runners,
        rejections=tuple(rejections),
    )


def _explain(
    job: JobSpec,
    best: RankedOption,
    scored: list[RankedOption],
    required_gb: float,
    budget: float | None,
    preferred_note: str,
) -> str:
    chip, offer = best.chip, best.offer
    parts: list[str] = []

    if offer.price_per_hour_usd is not None:
        parts.append(
            f"Picked {chip.name} on {offer.provider} at "
            f"${offer.price_per_hour_usd:.2f}/hr (price dated {offer.price_date})."
        )
    else:
        parts.append(
            f"Picked {chip.name} on {offer.provider} (no published price in the catalog)."
        )

    headroom = chip.memory_gb / required_gb
    parts.append(
        f"The job needs ~{required_gb:g} GB at {job.precision}; this chip has "
        f"{chip.memory_gb:g} GB ({headroom:.1f}x headroom)."
    )

    if len(scored) == 1:
        parts.append("It is the only offer that passes the job's constraints.")
    else:
        task_hint = {
            "inference": "memory bandwidth, which dominates inference serving speed",
            "fine_tune": "tensor throughput and price, which dominate fine-tuning cost",
            "training": "tensor throughput, which dominates training speed",
            "signal_processing": "memory bandwidth, which dominates signal-processing throughput",
        }[job.task]
        parts.append(
            f"Among {len(scored)} feasible offers it has the best balance of {task_hint}, "
            f"hourly price, and right-sizing "
            f"(score {best.score:.2f} vs {scored[1].score:.2f} for the runner-up)."
        )

    if budget is not None:
        parts.append(f"It fits the ${budget:.2f}/hr budget cap.")
    if preferred_note:
        parts.append(preferred_note.strip())
    return " ".join(parts)
