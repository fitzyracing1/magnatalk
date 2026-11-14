from pathlib import Path

import pytest

from magnatalk.catalog import load_catalog
from magnatalk.jobspec import job_from_dict, load_job
from magnatalk.router import NoFeasibleChipError, route

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def chips():
    return load_catalog()


def test_8b_inference_pick_is_cheap_and_fits(chips):
    decision = route(load_job(EXAMPLES / "8b-inference.yaml"), chips)
    assert decision.chip.memory_gb >= decision.required_vram_gb
    assert decision.estimated_cost_per_hour_usd is not None
    assert decision.estimated_cost_per_hour_usd <= 1.50
    # A $1.50/hr budget should land on a small cost-efficient card.
    assert decision.chip.id in ("nvidia-rtx-4090", "nvidia-l40s", "nvidia-a10g")
    assert decision.reason
    assert decision.runners_up


def test_70b_inference_needs_huge_memory(chips):
    decision = route(load_job(EXAMPLES / "70b-inference.yaml"), chips)
    # 70B fp16 ~= 168 GB; only the B200 has both the memory and a rental offer
    # (MI300X would fit but has no priced offers from RunPod/Modal/Lambda).
    assert decision.chip.id == "nvidia-b200"
    assert decision.required_vram_gb == pytest.approx(168.0)
    assert "GB" in decision.reason


def test_13b_finetune_respects_job_budget(chips):
    decision = route(load_job(EXAMPLES / "13b-finetune.yaml"), chips)
    # $30 over 12h caps hourly spend at $2.50 -> A100 80GB territory.
    assert decision.estimated_cost_per_hour_usd <= 2.5
    assert decision.chip.memory_gb >= 65
    assert decision.chip.id == "nvidia-a100-80gb-sxm"
    assert decision.estimated_job_cost_usd == pytest.approx(
        decision.estimated_cost_per_hour_usd * 12, abs=0.01
    )


def test_signal_processing_prefers_bandwidth(chips):
    decision = route(load_job(EXAMPLES / "signal-processing.yaml"), chips)
    assert decision.estimated_cost_per_hour_usd <= 1.20
    assert decision.chip.memory_gb >= 16
    # Among budget options the RTX 4090 has by far the highest bandwidth.
    assert decision.chip.id == "nvidia-rtx-4090"


def test_routing_is_deterministic(chips):
    job = load_job(EXAMPLES / "8b-inference.yaml")
    a = route(job, chips)
    b = route(job, chips)
    assert a.to_dict() == b.to_dict()


def test_budget_cap_filters_expensive_offers(chips):
    job = job_from_dict({"task": "inference", "vram_gb": 70, "budget_per_hour_usd": 3.0})
    decision = route(job, chips)
    assert decision.estimated_cost_per_hour_usd <= 3.0
    for r in decision.runners_up:
        assert r.offer.price_per_hour_usd <= 3.0


def test_excluded_provider_never_picked(chips):
    job = job_from_dict({
        "task": "inference", "model_params_b": 8,
        "excluded_providers": ["runpod"],
    })
    decision = route(job, chips)
    assert decision.offer.provider != "runpod"
    assert all(r.offer.provider != "runpod" for r in decision.runners_up)


def test_preferred_provider_restricts_choice(chips):
    job = job_from_dict({
        "task": "inference", "model_params_b": 8,
        "preferred_providers": ["lambda"],
    })
    decision = route(job, chips)
    assert decision.offer.provider == "lambda"
    assert "preferred" in decision.reason


def test_preferred_provider_falls_back_when_infeasible(chips):
    # Lambda has no offer big enough for ~168 GB jobs except B200; exclude it
    # by preferring a provider with no feasible offer at this size + budget.
    job = job_from_dict({
        "task": "inference", "model_params_b": 70,
        "preferred_providers": ["nonexistent-cloud"],
    })
    decision = route(job, chips)
    assert decision.chip.id == "nvidia-b200"
    assert "No feasible offer from preferred providers" in decision.reason


def test_unsupported_precision_excludes_chip(chips):
    # A100 does not support fp8; an fp8 job must never land on it.
    job = job_from_dict({"task": "inference", "vram_gb": 60, "precision": "fp8"})
    decision = route(job, chips)
    assert decision.chip.id != "nvidia-a100-80gb-sxm"
    assert "fp8" in decision.chip.supported_precisions


def test_impossible_job_raises_with_rejections(chips):
    job = job_from_dict({"task": "training", "vram_gb": 4096})
    with pytest.raises(NoFeasibleChipError) as exc:
        route(job, chips)
    assert exc.value.rejections
    assert "4096" in str(exc.value)


def test_unknown_price_rejected_when_budget_set():
    chips = load_catalog()
    job = job_from_dict({"task": "inference", "vram_gb": 8, "budget_per_hour_usd": 100})
    decision = route(job, chips)
    assert decision.estimated_cost_per_hour_usd is not None
