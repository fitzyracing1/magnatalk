from __future__ import annotations

from ..router import Decision
from .base import LaunchPlan, ProviderAdapter

# Map catalog chip ids to RunPod GPU type names shown on runpod.io.
RUNPOD_GPU_NAMES = {
    "nvidia-h100-sxm": "H100 SXM",
    "nvidia-h200-sxm": "H200",
    "nvidia-b200": "B200",
    "nvidia-a100-80gb-sxm": "A100 SXM",
    "nvidia-l40s": "L40S",
    "nvidia-rtx-4090": "RTX 4090",
}


class RunPodAdapter(ProviderAdapter):
    name = "runpod"
    api_key_env = "RUNPOD_API_KEY"

    def plan_launch(self, decision: Decision) -> LaunchPlan:
        gpu = RUNPOD_GPU_NAMES.get(decision.chip.id, decision.chip.name)
        command = (
            f'runpodctl create pod --gpuType "{gpu}" --gpuCount 1 '
            f'--name "{decision.job.name}" --imageName runpod/pytorch'
        )
        return LaunchPlan(
            provider=self.name,
            summary=f"1x {gpu} pod for job '{decision.job.name}'",
            command=command,
            api_key_env=self.api_key_env,
            api_key_present=self._api_key_present(),
        )
