from __future__ import annotations

from ..router import Decision
from .base import LaunchPlan, ProviderAdapter

# Map catalog chip ids to Lambda Cloud instance type names.
LAMBDA_INSTANCE_TYPES = {
    "nvidia-h100-sxm": "gpu_1x_h100_sxm5",
    "nvidia-b200": "gpu_1x_b200_sxm6",
    "nvidia-a100-80gb-sxm": "gpu_8x_a100_80gb_sxm4",
    "nvidia-a10g": "gpu_1x_a10",
}


class LambdaAdapter(ProviderAdapter):
    name = "lambda"
    api_key_env = "LAMBDA_API_KEY"

    def plan_launch(self, decision: Decision) -> LaunchPlan:
        instance = LAMBDA_INSTANCE_TYPES.get(decision.chip.id, decision.chip.name)
        command = (
            "curl -u $LAMBDA_API_KEY: https://cloud.lambda.ai/api/v1/instance-operations/launch "
            f"-d '{{\"instance_type_name\": \"{instance}\", \"name\": \"{decision.job.name}\"}}'"
        )
        return LaunchPlan(
            provider=self.name,
            summary=f"Lambda instance {instance} for job '{decision.job.name}'",
            command=command,
            api_key_env=self.api_key_env,
            api_key_present=self._api_key_present(),
        )
