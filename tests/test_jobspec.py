import json

import pytest

from magnatalk.jobspec import JobSpecError, job_from_dict, load_job


def test_vram_estimate_inference_fp16():
    job = job_from_dict({"task": "inference", "model_params_b": 8})
    # 8B params * 2 bytes * 1.2 inference factor = 19.2 GB
    assert job.required_vram_gb() == pytest.approx(19.2)


def test_vram_estimate_finetune():
    job = job_from_dict({"task": "fine_tune", "model_params_b": 13})
    # 13B * 2 bytes * 2.5 PEFT factor = 65 GB
    assert job.required_vram_gb() == pytest.approx(65.0)


def test_vram_estimate_respects_precision():
    job = job_from_dict({"task": "inference", "model_params_b": 70, "precision": "fp8"})
    assert job.required_vram_gb() == pytest.approx(70 * 1.0 * 1.2)


def test_explicit_vram_overrides_estimate():
    job = job_from_dict({"task": "inference", "model_params_b": 8, "vram_gb": 40})
    assert job.required_vram_gb() == 40


def test_signal_processing_default_vram():
    job = job_from_dict({"task": "signal_processing"})
    assert job.required_vram_gb() == 8.0


def test_missing_memory_info_raises():
    with pytest.raises(JobSpecError):
        job_from_dict({"task": "inference"})


def test_bad_task_raises():
    with pytest.raises(JobSpecError):
        job_from_dict({"task": "mining", "vram_gb": 8})


def test_bad_precision_raises():
    with pytest.raises(JobSpecError):
        job_from_dict({"task": "inference", "vram_gb": 8, "precision": "fp13"})


def test_negative_budget_raises():
    with pytest.raises(JobSpecError):
        job_from_dict({"task": "inference", "vram_gb": 8, "budget_per_hour_usd": -1})


def test_effective_budget_from_job_budget():
    job = job_from_dict({
        "task": "fine_tune", "model_params_b": 13,
        "budget_per_job_usd": 30, "max_runtime_hours": 12,
    })
    assert job.effective_budget_per_hour() == pytest.approx(2.5)


def test_effective_budget_takes_minimum():
    job = job_from_dict({
        "task": "fine_tune", "model_params_b": 13,
        "budget_per_hour_usd": 2.0,
        "budget_per_job_usd": 30, "max_runtime_hours": 12,
    })
    assert job.effective_budget_per_hour() == pytest.approx(2.0)


def test_load_yaml_and_json(tmp_path):
    y = tmp_path / "job.yaml"
    y.write_text("task: inference\nmodel_params_b: 8\n", encoding="utf-8")
    j = tmp_path / "job.json"
    j.write_text(json.dumps({"task": "inference", "model_params_b": 8}), encoding="utf-8")
    assert load_job(y).required_vram_gb() == load_job(j).required_vram_gb()


def test_provider_lists_lowercased():
    job = job_from_dict({
        "task": "inference", "vram_gb": 8,
        "preferred_providers": ["RunPod"], "excluded_providers": ["Modal"],
    })
    assert job.preferred_providers == ("runpod",)
    assert job.excluded_providers == ("modal",)
