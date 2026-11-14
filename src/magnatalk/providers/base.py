from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..router import Decision


@dataclass(frozen=True)
class LaunchPlan:
    """A description of what WOULD be launched. Nothing is ever executed."""

    provider: str
    summary: str
    command: str
    api_key_env: str
    api_key_present: bool

    def render(self) -> str:
        key_state = "set" if self.api_key_present else "NOT set"
        return (
            f"[DRY RUN] {self.provider}: {self.summary}\n"
            f"  would run : {self.command}\n"
            f"  api key   : ${self.api_key_env} is {key_state} "
            f"(never required for a dry run; nothing is launched or billed)"
        )


class ProviderAdapter(ABC):
    """Base class for provider adapters. v0.1 adapters are dry-run only."""

    name: str
    api_key_env: str

    @abstractmethod
    def plan_launch(self, decision: Decision) -> LaunchPlan:
        """Return the launch plan for a routing decision. Never executes it."""

    def _api_key_present(self) -> bool:
        # Presence check only; the value is never read into the plan.
        return bool(os.environ.get(self.api_key_env))
