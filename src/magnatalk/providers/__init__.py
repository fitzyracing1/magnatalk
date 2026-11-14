"""Provider adapters. DRY-RUN ONLY in v0.1.

Adapters never make network calls, never rent hardware, and never spend
money. They only render what *would* be launched. API keys are read from
environment variables purely to report whether they are set; they are never
required and never sent anywhere.
"""

from __future__ import annotations

from .base import LaunchPlan, ProviderAdapter
from .lambda_cloud import LambdaAdapter
from .modal import ModalAdapter
from .runpod import RunPodAdapter

ADAPTERS: dict[str, ProviderAdapter] = {
    "runpod": RunPodAdapter(),
    "modal": ModalAdapter(),
    "lambda": LambdaAdapter(),
}


def get_adapter(provider: str) -> ProviderAdapter | None:
    return ADAPTERS.get(provider.lower())


__all__ = ["ADAPTERS", "LaunchPlan", "ProviderAdapter", "get_adapter"]
