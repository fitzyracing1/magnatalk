"""magnatalk MCP server: expose chip routing as a tool for AI agents.

Run with ``magnatalk-mcp`` (stdio transport). Requires the optional ``mcp``
dependency: ``pip install magnatalk[mcp]``.

Example client config (Cursor / Claude Desktop style)::

    {"mcpServers": {"magnatalk": {"command": "magnatalk-mcp"}}}

The server is read-only with respect to money and hardware: it only routes
and explains. It never launches instances or calls provider APIs.
"""

from __future__ import annotations

import sys
from typing import Any

from .catalog import load_catalog
from .decision_log import append_decision
from .jobspec import job_from_dict
from .router import NoFeasibleChipError, route


def pick_chip_impl(job: dict[str, Any], log: bool = False) -> dict[str, Any]:
    """Shared implementation so it can be tested without the mcp package."""
    chips = load_catalog()
    spec = job_from_dict(job)
    try:
        decision = route(spec, chips)
    except NoFeasibleChipError as e:
        return {
            "ok": False,
            "error": "no_feasible_chip",
            "detail": str(e),
            "rejections": [
                {"chip": r.chip_id, "provider": r.provider, "reason": r.reason}
                for r in e.rejections
            ],
        }
    result = {"ok": True, **decision.to_dict()}
    if log:
        rec = append_decision(decision)
        result["decision_id"] = rec["id"]
    return result


def list_chips_impl() -> list[dict[str, Any]]:
    chips = load_catalog()
    return [
        {
            "id": c.id,
            "name": c.name,
            "memory_gb": c.memory_gb,
            "memory_bandwidth_gbps": c.memory_bandwidth_gbps,
            "fp16_tflops": c.fp16_tflops,
            "fp8_tflops": c.fp8_tflops,
            "providers": sorted({o.provider for o in c.offers}),
        }
        for c in chips
    ]


def build_server():
    try:
        # mcp>=2 renamed FastMCP to MCPServer; support both.
        from mcp.server.mcpserver import MCPServer as _Server
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as _Server
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "the MCP server needs the optional 'mcp' dependency: "
                "pip install 'magnatalk[mcp]'"
            ) from e

    server = _Server(
        "magnatalk",
        instructions=(
            "Pick AI chips on purpose. Call pick_chip with a job spec to get a "
            "deterministic, explained chip+provider choice with runners-up. "
            "This server never launches hardware or spends money."
        ),
    )

    @server.tool()
    def pick_chip(
        task: str,
        model_params_b: float | None = None,
        vram_gb: float | None = None,
        precision: str = "fp16",
        budget_per_hour_usd: float | None = None,
        budget_per_job_usd: float | None = None,
        max_runtime_hours: float | None = None,
        preferred_providers: list[str] | None = None,
        excluded_providers: list[str] | None = None,
        name: str = "mcp-job",
        log: bool = False,
    ) -> dict:
        """Choose the best AI chip + rental provider for a job.

        task: one of inference, fine_tune, training, signal_processing.
        Give either model_params_b (model size in billions of parameters)
        or vram_gb (explicit memory requirement). Returns the pick, a
        plain-English reason, runners-up, and estimated cost. Dry-run only:
        nothing is launched and nothing is billed.
        """
        job = {
            "name": name,
            "task": task,
            "model_params_b": model_params_b,
            "vram_gb": vram_gb,
            "precision": precision,
            "budget_per_hour_usd": budget_per_hour_usd,
            "budget_per_job_usd": budget_per_job_usd,
            "max_runtime_hours": max_runtime_hours,
            "preferred_providers": preferred_providers or [],
            "excluded_providers": excluded_providers or [],
        }
        return pick_chip_impl(job, log=log)

    @server.tool()
    def list_chips() -> list[dict]:
        """List the chip catalog: specs and which providers rent each chip."""
        return list_chips_impl()

    return server


def main() -> int:
    build_server().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
