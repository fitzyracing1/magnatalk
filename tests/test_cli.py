import json
from pathlib import Path

from magnatalk.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_validate_ok(capsys):
    assert main(["validate"]) == 0
    assert "valid" in capsys.readouterr().out


def test_validate_bad_catalog(tmp_path, capsys):
    bad = tmp_path / "chips.yaml"
    bad.write_text("schema_version: 1\nchips:\n  - id: broken\n", encoding="utf-8")
    assert main(["validate", "--catalog", str(bad)]) == 1
    assert "INVALID" in capsys.readouterr().err


def test_list(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "NVIDIA H100 SXM" in out
    assert "runpod" in out


def test_list_json(capsys):
    assert main(["list", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert any(c["id"] == "amd-mi300x" for c in data)


def test_pick_text_output(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    rc = main(["pick", str(EXAMPLES / "8b-inference.yaml"), "--log-file", str(log)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PICK:" in out
    assert "WHY" in out
    assert "RUNNERS-UP" in out
    assert log.exists()


def test_pick_json_output(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    rc = main(["pick", str(EXAMPLES / "70b-inference.yaml"),
               "--json", "--log-file", str(log)])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["chip"] == "nvidia-b200"
    assert data["reason"]
    assert data["runners_up"] is not None
    assert data["id"]


def test_pick_no_log(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["pick", str(EXAMPLES / "8b-inference.yaml"), "--no-log"])
    assert rc == 0
    assert not list(tmp_path.glob("*.jsonl"))


def test_pick_infeasible_job(tmp_path, capsys):
    job = tmp_path / "job.yaml"
    job.write_text("task: training\nvram_gb: 4096\n", encoding="utf-8")
    rc = main(["pick", str(job), "--no-log"])
    assert rc == 2
    assert "no feasible chip" in capsys.readouterr().err


def test_pick_launch_plan_is_dry_run(tmp_path, capsys):
    rc = main(["pick", str(EXAMPLES / "8b-inference.yaml"),
               "--no-log", "--launch-plan"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[DRY RUN]" in out
    assert "nothing is launched or billed" in out


def test_log_and_annotate(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    main(["pick", str(EXAMPLES / "8b-inference.yaml"), "--json", "--log-file", str(log)])
    rec = json.loads(capsys.readouterr().out)

    assert main(["log", "--log-file", str(log)]) == 0
    assert rec["id"] in capsys.readouterr().out

    rc = main(["log", "--log-file", str(log), "--annotate", rec["id"],
               "--actual", '{"cost_usd": 0.95}'])
    assert rc == 0
    assert main(["log", "--log-file", str(log), "--json"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()
               if line.startswith("{")]
    assert records[-1]["actual"] == {"cost_usd": 0.95}


def test_log_empty(tmp_path, capsys):
    assert main(["log", "--log-file", str(tmp_path / "none.jsonl")]) == 0
    assert "no decisions" in capsys.readouterr().out


def test_missing_job_file(capsys):
    assert main(["pick", "does-not-exist.yaml", "--no-log"]) == 1
    assert "not found" in capsys.readouterr().err
