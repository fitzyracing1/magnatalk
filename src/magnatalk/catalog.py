"""Chip catalog loading and validation.

The catalog is a plain YAML file (``src/magnatalk/data/chips.yaml`` by default)
so that adding a new chip is just adding an entry -- no code changes needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

KNOWN_PRECISIONS = {"fp32", "fp16", "bf16", "fp8", "fp4", "int8", "int4"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class CatalogError(ValueError):
    """Raised when the catalog file is malformed."""


@dataclass(frozen=True)
class Offer:
    provider: str
    price_per_hour_usd: float | None
    price_date: str | None
    source: str | None
    notes: str | None = None


@dataclass(frozen=True)
class Chip:
    id: str
    name: str
    vendor: str
    memory_gb: float
    memory_bandwidth_gbps: float
    supported_precisions: tuple[str, ...]
    spec_source: str
    fp16_tflops: float | None = None
    bf16_tflops: float | None = None
    fp8_tflops: float | None = None
    notes: str | None = None
    offers: tuple[Offer, ...] = field(default_factory=tuple)

    def throughput_tflops(self, precision: str) -> float | None:
        """Dense tensor throughput used when running at ``precision``.

        int8/int4 jobs are assumed to be weight-only quantization executed on
        the FP8 (if available) or FP16 tensor pipeline.
        """
        if precision == "fp8" or precision == "fp4":
            return self.fp8_tflops
        if precision == "bf16":
            return self.bf16_tflops
        if precision in ("int8", "int4"):
            return self.fp8_tflops or self.fp16_tflops or self.bf16_tflops
        if precision == "fp16":
            return self.fp16_tflops
        if precision == "fp32":
            # Catalog does not track FP32; treat as unavailable for scoring.
            return None
        return None


def default_catalog_path() -> Path:
    return Path(str(resources.files("magnatalk").joinpath("data/chips.yaml")))


def _err(errors: list[str], where: str, msg: str) -> None:
    errors.append(f"{where}: {msg}")


def _require(d: dict, key: str, types: tuple[type, ...], where: str, errors: list[str]):
    if key not in d or d[key] is None:
        _err(errors, where, f"missing required field '{key}'")
        return None
    if not isinstance(d[key], types):
        _err(errors, where, f"field '{key}' must be {' or '.join(t.__name__ for t in types)}")
        return None
    return d[key]


def validate_catalog_dict(raw: dict) -> list[str]:
    """Validate the raw catalog dict. Returns a list of error strings (empty = valid)."""
    errors: list[str] = []
    if not isinstance(raw, dict):
        return ["catalog root must be a mapping"]
    if raw.get("schema_version") != 1:
        _err(errors, "catalog", "schema_version must be 1")
    chips = raw.get("chips")
    if not isinstance(chips, list) or not chips:
        return errors + ["catalog must contain a non-empty 'chips' list"]

    seen_ids: set[str] = set()
    for i, c in enumerate(chips):
        where = f"chips[{i}]"
        if not isinstance(c, dict):
            _err(errors, where, "chip entry must be a mapping")
            continue
        cid = _require(c, "id", (str,), where, errors)
        if cid:
            where = f"chip '{cid}'"
            if not ID_RE.match(cid):
                _err(errors, where, "id must be kebab-case (lowercase letters, digits, dashes)")
            if cid in seen_ids:
                _err(errors, where, "duplicate chip id")
            seen_ids.add(cid)
        _require(c, "name", (str,), where, errors)
        _require(c, "vendor", (str,), where, errors)
        for num_field in ("memory_gb", "memory_bandwidth_gbps"):
            v = _require(c, num_field, (int, float), where, errors)
            if isinstance(v, (int, float)) and v <= 0:
                _err(errors, where, f"'{num_field}' must be positive")
        for opt_field in ("fp16_tflops", "bf16_tflops", "fp8_tflops"):
            v = c.get(opt_field)
            if v is not None and (not isinstance(v, (int, float)) or v <= 0):
                _err(errors, where, f"'{opt_field}' must be a positive number or null")
        _require(c, "spec_source", (str,), where, errors)
        precs = _require(c, "supported_precisions", (list,), where, errors)
        if isinstance(precs, list):
            unknown = [p for p in precs if p not in KNOWN_PRECISIONS]
            if unknown:
                _err(errors, where, f"unknown precisions {unknown}; known: {sorted(KNOWN_PRECISIONS)}")
        offers = c.get("offers")
        if offers is None or not isinstance(offers, list):
            _err(errors, where, "'offers' must be a list (use [] when nobody rents this chip)")
            continue
        for j, o in enumerate(offers):
            owhere = f"{where} offers[{j}]"
            if not isinstance(o, dict):
                _err(errors, owhere, "offer must be a mapping")
                continue
            _require(o, "provider", (str,), owhere, errors)
            price = o.get("price_per_hour_usd")
            if price is not None:
                if not isinstance(price, (int, float)) or price <= 0:
                    _err(errors, owhere, "'price_per_hour_usd' must be a positive number or null")
                # Honesty rule: a real price must say when and where it came from.
                date = o.get("price_date")
                if not isinstance(date, str) or not DATE_RE.match(date):
                    _err(errors, owhere,
                         "a non-null price requires 'price_date' in YYYY-MM-DD format")
                src = o.get("source")
                if not isinstance(src, str) or not src.startswith(("http://", "https://")):
                    _err(errors, owhere,
                         "a non-null price requires 'source' (URL of the public pricing page)")
    return errors


def _parse_chip(c: dict) -> Chip:
    offers = tuple(
        Offer(
            provider=o["provider"],
            price_per_hour_usd=o.get("price_per_hour_usd"),
            price_date=o.get("price_date"),
            source=o.get("source"),
            notes=o.get("notes"),
        )
        for o in c.get("offers", [])
    )
    return Chip(
        id=c["id"],
        name=c["name"],
        vendor=c["vendor"],
        memory_gb=float(c["memory_gb"]),
        memory_bandwidth_gbps=float(c["memory_bandwidth_gbps"]),
        supported_precisions=tuple(c["supported_precisions"]),
        spec_source=c["spec_source"],
        fp16_tflops=c.get("fp16_tflops"),
        bf16_tflops=c.get("bf16_tflops"),
        fp8_tflops=c.get("fp8_tflops"),
        notes=c.get("notes"),
        offers=offers,
    )


def load_catalog(path: str | Path | None = None) -> list[Chip]:
    """Load and validate the catalog, returning chips in file order."""
    p = Path(path) if path else default_catalog_path()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    errors = validate_catalog_dict(raw)
    if errors:
        raise CatalogError(
            f"catalog {p} is invalid:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    return [_parse_chip(c) for c in raw["chips"]]
