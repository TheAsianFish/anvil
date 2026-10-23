"""Rendering of the coverage simulation block and of the BCa, exact and Wilson additions to the
calibration block."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval import render

# Any: report.json is nested JSON read back by key.
Report = dict[str, Any]


def fixed_cell(labels: int, n: int, kappa: float, *, counts: bool = True) -> Report:
    lower = {"percentile": 180, "bca": 190}
    mean = {"percentile": 0.8, "bca": 0.79}
    if counts:
        lower |= {"exact": 196 if labels == 4 else 194, "wilson": 193}
        mean |= {"exact": 0.77, "wilson": 0.775}
    return {
        "kappa": kappa,
        "n": n,
        "labels": labels,
        "reps": 200,
        "lower_covered": lower,
        "interval_covered": {"percentile": 185, "bca": 192},
        "mean_lower": mean,
        "mean_bca_level": 0.07,
        "mean_estimate": 0.9,
        "bias": 0.0012,
        "bias_se": 0.0008,
        "sd_estimate": 0.0154,
        "mean_bootstrap_sd": 0.0157,
    }


def rule(found: int, met: int, coverage: float | None, up: float | None) -> Report:
    return {
        "found": found,
        "met": met,
        "mean_true_coverage": coverage,
        "mean_coverage_given_up": up,
        "mean_true_kappa": 0.92,
        "mean_shortfall_when_missed": 0.01,
    }


def search_cell(labels: int, n: int, target: float, *, counts: bool = True) -> Report:
    extra = {"exact": rule(20, 20, 0.7, 0.19), "wilson": rule(25, 24, 0.72, 0.17)}
    return {
        "target": target,
        "n": n,
        "labels": labels,
        "reps": 30,
        "oracle_coverage": 0.889,
        "point": rule(30, 15, 0.89, 0.0),
        "lower": rule(30, 27, 0.75, 0.14),
        "bca": rule(28, 26, None, 0.12),
        **(extra if counts else {}),
    }


def coverage_report(*, counts: bool = True) -> Report:
    return {
        "config": {
            "fixed_reps": 200,
            "search_reps": 30,
            "n_resamples": 1000,
            "search_population": {
                "beta": 0.8,
                "gamma": 4.0,
                "kappa_at_coverage": {"1.0": 0.84, "0.5": 0.99},
            },
        },
        "fixed": [
            fixed_cell(77, 250, 0.9, counts=counts),
            fixed_cell(4, 100, 0.85, counts=counts),
        ],
        "search": [
            search_cell(77, 100, 0.9, counts=counts),
            search_cell(4, 100, 0.95, counts=counts),
        ],
    }


META = {"command": "python -m eval.coverage", "git_sha": "abcdef1234567", "workers": 2}


def test_render_coverage_tables() -> None:
    text = render.render_coverage(coverage_report(), {**META, "wall_seconds": 34.0})
    assert "`abcdef1`" in text
    assert "2 workers" in text
    assert (
        "| Labels | Items | Kappa | Percentile bound covers | BCa bound covers"
        " | Exact bound covers | Wilson bound covers | Mean bound:" in text
    )
    assert "| Labels | Items | Target | Oracle coverage | Point: met" in text
    assert "Wilson: met | Point: true coverage" in text
    first, second = [ln for ln in text.splitlines() if ln.startswith("| 4 |") or "| 77 |" in ln][
        :2
    ]
    assert first.startswith(
        "| 4 | 100 | 0.85 | 90.0% | 95.0% | 98.0% | 96.5% | 0.800 / 0.790 / 0.770 / 0.775"
        " | 7.0% | 92.5% / 96.0%"
    )
    assert "+0.0012 ± 0.0008 | 0.0154 / 0.0157 |" in first
    assert second.startswith("| 77 | 250 | 0.9 |")
    assert (
        "| 4 | 100 | 0.95 | 88.9% | 15/30 (50.0%) | 27/30 (90.0%) | 26/28 (92.9%)"
        " | 20/20 (100.0%) | 24/25 (96.0%) |" in text
    )
    assert "| 89.0% | 75.0% | n/a | 70.0% | 72.0% |" in text
    assert "Percentile 90.0%, BCa 95.0%, Exact 97.5%, Wilson 96.5%." in text
    assert "- 4 labels: Percentile 90.0%, BCa 95.0%, Exact 98.0%, Wilson 96.5%" in text
    assert "- Worst cell: Percentile 90.0% (4 labels, 100 items, kappa 0.85);" in text
    assert "Exact 97.0% (77 labels, 250 items, kappa 0.9)" in text
    assert (
        "95% BCa lower bound met 52/56 (92.9%), coverage given up versus the oracle 12.0%" in text
    )
    assert "95% Wilson count bound met 48/50 (96.0%)" in text
    assert (
        "- 4 labels: Point found a threshold in 30/30, met 50.0%, true coverage 89.0% (89.0%"
        in text
    )
    assert (
        "Exact found a threshold in 20/30, met 100.0%, true coverage 70.0%"
        " (46.7% counting a search with no threshold as 0)" in text
    )
    assert "100.0% -> 0.840, 50.0% -> 0.990" in text


def test_render_coverage_pointwise_table() -> None:
    report = coverage_report()
    report["config"]["pointwise_rates"] = {"low": 0.8, "high": 0.999, "points": 2000}
    report["pointwise"] = [
        {
            "n": 30,
            "method": "wilson",
            "min": 0.92503,
            "rate_at_min": 0.9173,
            "mean": 0.97664,
            "share_below_nominal": 0.1865,
        }
    ]
    text = render.render_coverage(report, META)
    assert "2000 agreement rates p from 0.8 to 0.999" in text
    assert "| 30 | Wilson | 92.50% | 0.917 | 97.66% | 18.6% |" in text
    assert "Pointwise" not in render.render_coverage(coverage_report(), META)


def test_render_coverage_of_a_report_without_the_count_bounds() -> None:
    text = render.render_coverage(coverage_report(counts=False), META)
    assert "Exact" not in text
    assert "Wilson" not in text
    assert "| 4 | 100 | 0.85 | 90.0% | 95.0% | 0.800 / 0.790 | 7.0% |" in text


def test_blocks_for_coverage_missing_and_present(tmp_path: Path) -> None:
    assert "No coverage simulation yet" in render.blocks_for(tmp_path)["coverage"]
    run = tmp_path / "coverage-simulation"
    run.mkdir()
    (run / "report.json").write_text(json.dumps(coverage_report()), encoding="utf-8")
    (run / "meta.json").write_text(json.dumps(META), encoding="utf-8")
    assert "Fixed sample" in render.blocks_for(tmp_path)["coverage"]


def test_ledger_maps_coverage_run_and_skips_smoke(tmp_path: Path) -> None:
    meta = {**META, "teacher_model": "n/a", "wall_seconds": 34.0, "spent_usd": 0.0}
    for name in ("coverage-simulation", "coverage-simulation-smoke"):
        run = tmp_path / name
        run.mkdir()
        (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    text = render.render_ledger(tmp_path)
    assert "| `coverage-simulation` | eval-coverage | n/a |" in text
    assert "smoke" not in text
    assert "| 0m 34s | $0.00 |" in text


def calibration_report(*, bca: bool, by_rule: bool) -> Report:
    method = {"found": 1.0, "met": 0.9, "coverage": None, "kappa": None, "system_kappa": None}
    methods = {"point": method, "lower": method}
    if bca:
        methods["bca"] = method
    ece = {"median": 0.1}
    report: Report = {
        "source": {
            "dataset": "banking77",
            "teacher_model": "gold",
            "student_model": "tiny",
            "student_run": "run-1",
        },
        "n": 10,
        "kappa": 0.9,
        "full": {},
        "split_study": {
            "ece_before": ece,
            "ece_after": ece,
            "splits": 2,
            "fit_size": 5,
            "test_size": 5,
            "min_accepted": 30,
            "n_resamples": 100,
            "temperature": None,
            "targets": {"0.9": methods},
        },
        "curve": [],
    }
    if by_rule:
        cell = {"threshold": 0.8, "coverage": 0.7, "kappa": 0.91, "kappa_low": 0.88}
        report["by_rule"] = {
            "point": {"0.9": cell, "0.95": cell},
            "lower": {"0.9": cell, "0.95": None},
            "bca": {"0.9": {**cell, "threshold": None}, "0.95": cell},
        }
    return report


def render_calibration(report: Report) -> str:
    return render.render_calibration([("calibration-banking77-gold", report, META)])


def test_render_calibration_with_bca_and_by_rule() -> None:
    text = render_calibration(calibration_report(bca=True, by_rule=True))
    assert "95% BCa lower bound, calibrated" in text
    assert "95% percentile lower bound, calibrated" in text
    assert (
        "| Target kappa | Rule | Threshold | Coverage | Kappa on answered | Lower bound |" in text
    )
    assert "| 0.9 | point estimate | 0.800 | 70.0% | 0.910 | 0.880 |" in text
    assert "| 0.9 | 95% BCa lower bound | none | 70.0% | 0.910 | 0.880 |" in text


def test_render_calibration_without_bca_or_by_rule() -> None:
    text = render_calibration(calibration_report(bca=False, by_rule=False))
    assert "BCa" not in text
    assert "Threshold each rule serves" not in text
    assert "95% percentile lower bound, calibrated" in text
