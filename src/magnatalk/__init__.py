"""magnatalk: pick AI chips on purpose.

Library quickstart::

    from magnatalk import load_catalog, load_job, route

    chips = load_catalog()
    job = load_job("job.yaml")
    decision = route(job, chips)
    print(decision.chip.name, decision.offer.provider, decision.reason)
"""

from .catalog import CatalogError, Chip, Offer, load_catalog, validate_catalog_dict
from .decision_log import annotate_decision, append_decision, read_decisions
from .jobspec import JobSpec, JobSpecError, job_from_dict, load_job
from .router import Decision, NoFeasibleChipError, RankedOption, route

__version__ = "0.1.0"

__all__ = [
    "CatalogError",
    "Chip",
    "Decision",
    "JobSpec",
    "JobSpecError",
    "NoFeasibleChipError",
    "Offer",
    "RankedOption",
    "__version__",
    "annotate_decision",
    "append_decision",
    "job_from_dict",
    "load_catalog",
    "load_job",
    "read_decisions",
    "route",
    "validate_catalog_dict",
]
