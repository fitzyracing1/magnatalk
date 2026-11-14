import importlib.util

import pytest

from magnatalk.mcp_server import list_chips_impl, pick_chip_impl


def test_pick_chip_impl_ok():
    result = pick_chip_impl({
        "name": "mcp-test", "task": "inference", "model_params_b": 8,
        "budget_per_hour_usd": 1.5,
    })
    assert result["ok"] is True
    assert result["chip"]
    assert result["provider"]
    assert result["reason"]
    assert isinstance(result["runners_up"], list)


def test_pick_chip_impl_infeasible():
    result = pick_chip_impl({"task": "training", "vram_gb": 4096})
    assert result["ok"] is False
    assert result["error"] == "no_feasible_chip"
    assert result["rejections"]


def test_list_chips_impl():
    chips = list_chips_impl()
    assert any(c["id"] == "nvidia-h100-sxm" for c in chips)
    h100 = next(c for c in chips if c["id"] == "nvidia-h100-sxm")
    assert set(h100["providers"]) == {"runpod", "modal", "lambda"}


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None,
                    reason="optional mcp dependency not installed")
def test_build_server_registers_tools():
    import anyio

    from magnatalk.mcp_server import build_server

    server = build_server()
    tools = anyio.run(server.list_tools)
    names = {t.name for t in tools}
    assert {"pick_chip", "list_chips"} <= names
