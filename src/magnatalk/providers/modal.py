from __future__ import annotations

from ..router import Decision
from .base import LaunchPlan, ProviderAdapter

# Map catalog chip ids to Modal `gpu=` argument strings.
MODAL_GPU_ARGS = {
    "nvidia-h100-sxm": "H100",
    "nvidia-h200-sxm": "H200",
    "nvidia-b200": "B200",
    "nvidia-a100-80gb-sxm": "A100-80GB",
    "nvidia-l40s": "L40S",
    "nvidia-a10g": "A10G",
}


class ModalAdapter(ProviderAdapter):
    name = "modal"
    api_key_env = "MODAL_TOKEN_ID"

    def plan_launch(self, decision: Decision) -> LaunchPlan:
        gpu = MODAL_GPU_ARGS.get(decision.chip.id, decision.chip.name)
        command = (
            f"modal run job.py  # with @app.function(gpu=\"{gpu}\") "
            f"for job '{decision.job.name}'"
        )
        return LaunchPlan(
            provider=self.name,
            summary=f"Modal function on gpu=\"{gpu}\" for job '{decision.job.name}'",
            command=command,
            api_key_env=self.api_key_env,
            api_key_present=self._api_key_present(),
        )
