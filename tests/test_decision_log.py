import pytest

from magnatalk.catalog import load_catalog
from magnatalk.decision_log import annotate_decision, append_decision, read_decisions
from magnatalk.jobspec import job_from_dict
from magnatalk.router import route


@pytest.fixture()
def decision():
    chips = load_catalog()
    job = job_from_dict({"name": "log-test", "task": "inference", "model_params_b": 8})
    return route(job, chips)


def test_append_and_read_roundtrip(tmp_path, decision):
    log = tmp_path / "decisions.jsonl"
    rec = append_decision(decision, log)
    records = read_decisions(log)
    assert len(records) == 1
    got = records[0]
    assert got["id"] == rec["id"]
    assert got["job"]["name"] == "log-test"
    assert got["chip"] == decision.chip.id
    assert got["provider"] == decision.offer.provider
    assert got["reason"]
    assert got["actual"] is None
    assert "timestamp" in got


def test_append_is_append_only(tmp_path, decision):
    log = tmp_path / "decisions.jsonl"
    append_decision(decision, log)
    append_decision(decision, log)
    assert len(read_decisions(log)) == 2


def test_annotate_sets_actual(tmp_path, decision):
    log = tmp_path / "decisions.jsonl"
    rec = append_decision(decision, log)
    annotate_decision(rec["id"], {"cost_usd": 1.23, "runtime_hours": 1.4}, log)
    records = read_decisions(log)
    assert records[0]["actual"] == {"cost_usd": 1.23, "runtime_hours": 1.4}


def test_annotate_unknown_id_raises(tmp_path, decision):
    log = tmp_path / "decisions.jsonl"
    append_decision(decision, log)
    with pytest.raises(KeyError):
        annotate_decision("nope", {"cost_usd": 1}, log)


def test_read_missing_file_is_empty(tmp_path):
    assert read_decisions(tmp_path / "nothing.jsonl") == []
