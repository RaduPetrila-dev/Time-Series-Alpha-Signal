import subprocess
import sys
import json
from pathlib import Path


def test_cli_runs(tmp_path: Path):
    """Ensure the CLI produces expected outputs when invoked via python -m"""
    outdir = tmp_path / "results"
    cmd = [sys.executable, "-m", "time_series_alpha_signal", "run", "--output", str(outdir), "--days", "200"]
    # run CLI
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # parse metrics printed to stdout
    metrics = json.loads(res.stdout)
    assert "sharpe" in metrics
    assert (outdir / "equity.png").exists()
    assert (outdir / "drawdown.png").exists()
    assert (outdir / "metrics.json").exists()
