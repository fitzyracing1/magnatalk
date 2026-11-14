import copy

import pytest
import yaml

from magnatalk.catalog import (
    CatalogError,
    default_catalog_path,
    load_catalog,
    validate_catalog_dict,
)

REQUIRED_IDS = {
    "nvidia-h100-sxm",
    "nvidia-h200-sxm",
    "nvidia-b200",
    "nvidia-a100-80gb-sxm",
    "nvidia-l40s",
    "nvidia-a10g",
    "nvidia-rtx-4090",
    "amd-mi300x",
    "google-tpu-v5e",
}


def raw_catalog() -> dict:
    return yaml.safe_load(default_catalog_path().read_text(encoding="utf-8"))


def test_bundled_catalog_is_valid():
    assert validate_catalog_dict(raw_catalog()) == []


def test_bundled_catalog_has_expected_chips():
    chips = load_catalog()
    assert REQUIRED_IDS <= {c.id for c in chips}


def test_every_price_has_date_and_source():
    for chip in load_catalog():
        for o in chip.offers:
            if o.price_per_hour_usd is not None:
                assert o.price_date, f"{chip.id}/{o.provider} price missing date"
                assert o.source and o.source.startswith("https://"), (
                    f"{chip.id}/{o.provider} price missing source URL"
                )


def test_unpriced_chips_have_no_invented_offers():
    chips = {c.id: c for c in load_catalog()}
    # MI300X and TPU v5e are not on RunPod/Modal/Lambda public pricing pages.
    assert chips["amd-mi300x"].offers == ()
    assert chips["google-tpu-v5e"].offers == ()


def test_validate_rejects_price_without_source():
    raw = copy.deepcopy(raw_catalog())
    offer = raw["chips"][0]["offers"][0]
    offer.pop("source")
    errors = validate_catalog_dict(raw)
    assert any("source" in e for e in errors)


def test_validate_rejects_price_without_date():
    raw = copy.deepcopy(raw_catalog())
    offer = raw["chips"][0]["offers"][0]
    offer["price_date"] = "last tuesday"
    errors = validate_catalog_dict(raw)
    assert any("price_date" in e for e in errors)


def test_validate_rejects_duplicate_ids():
    raw = copy.deepcopy(raw_catalog())
    raw["chips"].append(copy.deepcopy(raw["chips"][0]))
    errors = validate_catalog_dict(raw)
    assert any("duplicate" in e for e in errors)


def test_validate_rejects_missing_required_field():
    raw = copy.deepcopy(raw_catalog())
    raw["chips"][0].pop("memory_gb")
    errors = validate_catalog_dict(raw)
    assert any("memory_gb" in e for e in errors)


def test_validate_rejects_unknown_precision():
    raw = copy.deepcopy(raw_catalog())
    raw["chips"][0]["supported_precisions"] = ["fp16", "fp512"]
    errors = validate_catalog_dict(raw)
    assert any("fp512" in e for e in errors)


def test_load_raises_on_invalid_catalog(tmp_path):
    bad = tmp_path / "chips.yaml"
    bad.write_text("schema_version: 1\nchips:\n  - id: x\n", encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(bad)


def test_adding_a_new_chip_is_just_a_new_entry(tmp_path):
    raw = copy.deepcopy(raw_catalog())
    raw["chips"].append({
        "id": "future-chip-9000",
        "name": "Future Chip 9000",
        "vendor": "future",
        "memory_gb": 512,
        "memory_bandwidth_gbps": 20000,
        "fp16_tflops": 9000,
        "supported_precisions": ["fp16", "bf16", "fp8"],
        "spec_source": "https://example.com/future-chip-9000",
        "offers": [],
    })
    assert validate_catalog_dict(raw) == []
    p = tmp_path / "chips.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    chips = load_catalog(p)
    assert any(c.id == "future-chip-9000" for c in chips)
