"""Adapters are dry-run only: they render text and never touch the network.

These tests run with no API keys set; adapters must work regardless.
"""

import socket

import pytest

from magnatalk.catalog import load_catalog
from magnatalk.jobspec import job_from_dict
from magnatalk.providers import ADAPTERS, get_adapter
from magnatalk.router import route


@pytest.fixture()
def decisions():
    chips = load_catalog()
    out = {}
    out["runpod"] = route(job_from_dict({
        "name": "p-test", "task": "inference", "model_params_b": 8,
        "preferred_providers": ["runpod"],
    }), chips)
    out["modal"] = route(job_from_dict({
        "name": "p-test", "task": "inference", "model_params_b": 8,
        "preferred_providers": ["modal"],
    }), chips)
    out["lambda"] = route(job_from_dict({
        "name": "p-test", "task": "inference", "model_params_b": 8,
        "preferred_providers": ["lambda"],
    }), chips)
    return out


def test_adapters_exist_for_all_catalog_providers():
    providers = {o.provider for c in load_catalog() for o in c.offers}
    assert providers <= set(ADAPTERS)


def test_plans_render_without_api_keys(decisions, monkeypatch):
    for var in ("RUNPOD_API_KEY", "MODAL_TOKEN_ID", "LAMBDA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for name, decision in decisions.items():
        plan = get_adapter(name).plan_launch(decision)
        text = plan.render()
        assert "[DRY RUN]" in text
        assert "p-test" in plan.command or "p-test" in plan.summary
        assert plan.api_key_present is False
        assert "NOT set" in text


def test_api_key_presence_detected_without_exposure(decisions, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "sk-secret-value")
    plan = get_adapter("runpod").plan_launch(decisions["runpod"])
    assert plan.api_key_present is True
    assert "sk-secret-value" not in plan.render()
    assert "sk-secret-value" not in plan.command


def test_adapters_never_open_sockets(decisions, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("adapter attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for name, decision in decisions.items():
        get_adapter(name).plan_launch(decision).render()
