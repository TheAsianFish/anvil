from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anvil.calibrate import choose_threshold, coverage_curve
from anvil.evaluate.metrics import cohen_kappa
from eval import coverage
from eval.coverage import (
    Cell,
    copy_rater,
    counts,
    fixed_cell,
    oracle_coverage,
    population_kappa,
    search_all,
    search_cell,
    simulate,
)


def test_copy_rater_has_the_copy_probability_as_its_population_kappa() -> None:
    rng = np.random.default_rng(0)
    for k, kappa in ((4, 0.9), (77, 0.85)):
        teacher = rng.integers(0, k, size=200_000, dtype=np.int64)
        student = copy_rater(rng, teacher, k, kappa)
        assert cohen_kappa(teacher.tolist(), student.tolist()) == pytest.approx(kappa, abs=0.005)


def test_search_population_kappa_matches_a_large_sample() -> None:
    rng = np.random.default_rng(1)
    n, k = 400_000, 10
    conf = rng.random(n)
    teacher = rng.integers(0, k, size=n, dtype=np.int64)
    student = copy_rater(rng, teacher, k, 1 - coverage.BETA * (1 - conf) ** coverage.GAMMA)
    for u in (1.0, 0.8, 0.5):
        keep = conf >= 1 - u
        measured = cohen_kappa(teacher[keep].tolist(), student[keep].tolist())
        assert measured == pytest.approx(population_kappa(u), abs=0.004)


def test_oracle_coverage_inverts_the_population_curve() -> None:
    for target in (0.9, 0.95):
        u = oracle_coverage(target)
        assert population_kappa(u) == pytest.approx(target)
    assert oracle_coverage(0.5) == 1.0  # everything already clears a low target


@pytest.mark.parametrize("seed", range(4))
def test_search_all_chooses_what_choose_threshold_chooses(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, k = 160, 5 if seed % 2 else 77
    conf = rng.random(n)
    teacher = rng.integers(0, k, size=n, dtype=np.int64)
    student = copy_rater(rng, teacher, k, 1 - coverage.BETA * (1 - conf) ** coverage.GAMMA)
    for target in (0.85, 0.9, 0.95):
        chosen = search_all(conf, teacher, student, k, target, seed=seed, n_resamples=200)
        for rule, point in chosen.items():
            expected = choose_threshold(
                conf, teacher, student, k, target, rule=rule, n_resamples=200, seed=seed
            )
            if expected is None:
                assert point is None
            else:
                assert point is not None
                assert (point.threshold, point.accepted) == (expected.threshold, expected.accepted)


def test_search_all_on_a_curve_nothing_clears() -> None:
    conf = np.linspace(0, 1, 50)
    teacher = np.zeros(50, dtype=np.int64)
    student = np.ones(50, dtype=np.int64)
    assert coverage_curve(conf, teacher, student, 2)[-1].kappa < 0.9
    assert set(search_all(conf, teacher, student, 2, 0.9, seed=0).values()) == {None}


def test_cells_report_their_counts() -> None:
    fixed = fixed_cell(Cell("fixed", 0, 0.9, 60, 4, 8, 0))
    covered = fixed["lower_covered"]
    assert isinstance(covered, dict)
    assert set(covered) == {"percentile", "bca", "exact", "wilson"}
    assert all(0 <= v <= 8 for v in covered.values())
    interval = fixed["interval_covered"]
    assert isinstance(interval, dict)
    assert set(interval) == {"percentile", "bca"}  # the count bounds are one-sided only
    assert isinstance(fixed["mean_bca_level"], float)
    search = search_cell(Cell("search", 0, 0.9, 80, 4, 3, 0))
    for rule in ("point", "lower", "bca", "exact", "wilson"):
        row = search[rule]
        assert isinstance(row, dict)
        assert 0 <= row["met"] <= row["found"] <= 3


def test_simulation_is_the_same_for_any_number_of_workers() -> None:
    one = simulate(2, 1, 5, workers=1)
    two = simulate(2, 1, 5, workers=2)
    assert one == two
    assert simulate(2, 1, 6, workers=1) != one
    config = one["config"]
    assert isinstance(config, dict)
    assert config["n_resamples"] == 1000  # what choose_threshold draws
    # 4 one-sided + 2 two-sided per fixed cell; 5 rules x (found, met) per search cell
    assert len(counts(one)) == 18 * 6 + 12 * 10


def test_main_writes_and_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Shrink the grid: the default one is the CI job's, about half a minute.
    monkeypatch.setattr(coverage, "SIZES", (40,))
    monkeypatch.setattr(coverage, "LABELS", (4,))
    monkeypatch.setattr(coverage, "N_RESAMPLES", 100)
    out = tmp_path / "coverage"
    argv = ["--fixed-reps", "3", "--search-reps", "1", "--workers", "1", "--out", str(out)]
    assert coverage.main(argv) == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["spent_usd"] == 0.0
    assert meta["command"].startswith("python -m eval.coverage --fixed-reps 3")
    assert len(report["fixed"]) == 3
    assert len(report["search"]) == 2
    assert coverage.main([*argv, "--check"]) == 0
    # A count off by more than one is a different simulation, not float noise.
    report["fixed"][0]["lower_covered"]["percentile"] += 2
    (out / "report.json").write_text(json.dumps(report), encoding="utf-8")
    assert coverage.main([*argv, "--check"]) == 1


def test_main_rejects_non_positive_reps() -> None:
    with pytest.raises(SystemExit):
        coverage.main(["--fixed-reps", "0"])
