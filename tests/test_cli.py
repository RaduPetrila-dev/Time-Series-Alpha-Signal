"""Tests for the CLI interface.

These are integration tests that invoke the CLI as a subprocess and
verify that expected output files are created and metrics are valid.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI and return the completed process."""
    cmd = [sys.executable, "-m", "time_series_alpha_signal", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


# ---------------------------------------------------------------------------
# Basic run
# ---------------------------------------------------------------------------


class TestCLIRun:
    """Tests for the ``run`` subcommand."""

    def test_run_produces_output_files(self, tmp_path: Path) -> None:
        outdir = tmp_path / "results"
        res = run_cli(
            "run",
            "--output", str(outdir),
            "--days", "200",
        )
        assert res.returncode == 0, f"CLI failed: {res.stderr}"

        metrics = json.loads(res.stdout)
        assert "sharpe" in metrics
        assert "cagr" in metrics
        assert "max_dd" in metrics

        expected_files = [
            "equity.png",
            "drawdown.png",
            "rolling_sharpe.png",
            "monthly_heatmap.png",
            "metrics.json",
            "daily_returns.csv",
            "weights.csv",
        ]
        for fname in expected_files:
            assert (outdir / fname).exists(), f"Missing output: {fname}"

    def test_run_metrics_json_matches_stdout(self, tmp_path: Path) -> None:
        outdir = tmp_path / "results"
        res = run_cli("run", "--output", str(outdir), "--days", "200")
        assert res.returncode == 0, f"CLI failed: {res.stderr}"

        stdout_metrics = json.loads(res.stdout)
        file_metrics = json.loads((outdir / "metrics.json").read_text())

        for key in ("sharpe", "cagr", "max_dd"):
            assert key in file_metrics
            assert abs(stdout_metrics[key] - file_metrics[key]) < 1e-6

    @pytest.mark.parametrize(
        "signal",
        ["momentum", "mean_reversion", "ewma_momentum", "ma_crossover"],
    )
    def test_run_with_different_signals(
        self, tmp_path: Path, signal: str
    ) -> None:
        outdir = tmp_path / f"results_{signal}"
        res = run_cli(
            "run",
            "--signal", signal,
            "--output", str(outdir),
            "--days", "200",
        )
        assert res.returncode == 0, f"{signal} failed: {res.stderr}"

        metrics = json.loads(res.stdout)
        assert "sharpe" in metrics


# ---------------------------------------------------------------------------
# Rebalance and cost model flags
# ---------------------------------------------------------------------------


class TestCLIFlags:
    """Tests for optional CLI flags."""

    def test_rebalance_weekly(self, tmp_path: Path) -> None:
        outdir = tmp_path / "results"
        res = run_cli(
            "run",
            "--output", str(outdir),
            "--days", "200",
            "--rebalance", "weekly",
        )
        assert res.returncode == 0, f"CLI failed: {res.stderr}"

    def test_impact_model_sqrt(self, tmp_path: Path) -> None:
        outdir = tmp_path / "results"
        res = run_cli(
            "run",
            "--output", str(outdir),
            "--days", "200",
            "--impact-model", "sqrt",
        )
        assert res.returncode == 0, f"CLI failed: {res.stderr}"

    def test_verbose_flag(self, tmp_path: Path) -> None:
        outdir = tmp_path / "results"
        res = run_cli(
            "run",
            "--output", str(outdir),
            "--days", "200",
            "-v",
        )
        assert res.returncode == 0, f"CLI failed: {res.stderr}"
        # Verbose mode should produce DEBUG output on stderr
        assert len(res.stderr) > 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestCLIErrors:
    """Tests for CLI error handling."""

    def test_invalid_signal_fails(self, tmp_path: Path) -> None:
        res = run_cli(
            "run",
            "--signal", "nonexistent_signal",
            "--output", str(tmp_path),
            "--days", "200",
        )
        assert res.returncode != 0

    def test_missing_csv_fails(self, tmp_path: Path) -> None:
        res = run_cli(
            "run",
            "--csv", str(tmp_path / "does_not_exist.csv"),
            "--output", str(tmp_path),
        )
        assert res.returncode != 0
