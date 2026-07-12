import json

from quantum_chaos.cli import main


def test_list_targets(capsys):
    rc = main(["list-targets"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "correlated_faults" in out
    assert "needle" in out


def test_list_optimizers(capsys):
    rc = main(["list-optimizers"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "vqs" in out and "qaoa" in out


def test_run_emits_json(capsys):
    rc = main([
        "run", "--target", "correlated_faults", "--factors", "6",
        "--optimizer", "vqs", "--seed", "0", "--budget", "2000",
        "--iterations", "30",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["optimizer"] == "vqs"
    assert "best_string" in payload
    assert payload["num_queries"] > 0


def test_run_writes_json_file(tmp_path, capsys):
    path = tmp_path / "res.json"
    rc = main([
        "run", "--target", "sum_threshold", "--factors", "6",
        "--optimizer", "random", "--seed", "1", "--budget", "500",
        "--json", str(path),
    ])
    assert rc == 0
    data = json.loads(path.read_text())
    assert data["found_known_worst"] is True


def test_benchmark_runs_multiple(capsys):
    rc = main([
        "benchmark", "--target", "sum_threshold", "--factors", "6",
        "--optimizers", "random,hillclimb", "--seed", "0", "--budget", "600",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "random" in out and "hillclimb" in out
