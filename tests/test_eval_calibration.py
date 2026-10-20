"""The calibration harness on synthetic logits: report shape, the --logits path, rendering."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from eval import calibration
from eval.calibration import COVERAGE_GRID, METHODS, N_BINS, TARGETS, analyse, curve_at, main
from eval.render import render_calibration

N, K, SPLITS, RESAMPLES = 300, 5, 20, 200


def synthetic() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Overconfident logits: calibrated ones scaled by 3, so the fitted temperature is ~3."""
    rng = np.random.default_rng(0)
    calibrated = rng.normal(size=(N, K))
    calibrated[np.arange(N), rng.integers(0, K, N)] += 4.0
    probs = np.exp(calibrated - calibrated.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    teacher = np.array([rng.choice(K, p=p) for p in probs], dtype=np.int64)
    return 3 * calibrated, teacher


@cache
def report() -> dict[str, Any]:
    z, y = synthetic()
    return analyse(z, y, SPLITS, RESAMPLES, seed=0)


# Any: the report is nested JSON read back by key; there is no useful static type.
def section(r: dict[str, Any], key: str) -> dict[str, Any]:
    value = r[key]
    assert isinstance(value, dict)
    return value


def test_report_shape() -> None:
    r = report()
    assert r["n"] == N
    assert r["labels"] == K
    assert isinstance(r["kappa"], float)
    full = section(r, "full")
    assert set(full) == {"0.9", "0.95"}
    rel = section(r, "reliability")
    for side in ("before", "after"):
        bins = rel[side]
        assert isinstance(bins, list)
        assert len(bins) == N_BINS
        assert sum(b["n"] for b in bins) == N
    curve = r["curve"]
    assert isinstance(curve, list)
    assert curve
    for row in curve:
        assert row["level"] in COVERAGE_GRID
        assert row["coverage"] <= row["level"] + 1e-9
    study = section(r, "split_study")
    assert study["fit_size"] + study["test_size"] == N
    assert study["splits"] == SPLITS
    targets = section(study, "targets")
    assert set(targets) == {str(t) for t in TARGETS}
    for by_method in targets.values():
        assert set(by_method) == {name for name, _, _ in METHODS}


def test_temperature_is_fitted_near_three_and_calibration_helps() -> None:
    r = report()
    served = section(section(r, "full"), "0.9")
    assert 2.0 < served["temperature"] < 4.5
    study = section(r, "split_study")
    assert section(study, "ece_after")["median"] < section(study, "ece_before")["median"]
    assert served["ece_after"] < served["ece_before"]


def test_lower_rule_answers_no_more_than_point_rule() -> None:
    targets = section(section(report(), "split_study"), "targets")
    for by_method in targets.values():
        point, lower = by_method["point"], by_method["lower"]
        if "coverage" in point and "coverage" in lower:
            assert lower["coverage"]["median"] <= point["coverage"]["median"] + 1e-12


def test_analyse_is_deterministic() -> None:
    z, y = synthetic()
    assert analyse(z, y, 5, 50, seed=3) == analyse(z, y, 5, 50, seed=3)


def test_curve_ends_at_full_coverage() -> None:
    z, y = synthetic()
    rows = curve_at(z, y, 3.0)
    assert rows[0]["level"] == 1.0
    assert rows[0]["coverage"] == 1.0


def write_logits(directory: Path, source: dict[str, str]) -> Path:
    z, y = synthetic()
    directory.mkdir(parents=True)
    npz = directory / "logits.npz"
    np.savez_compressed(
        npz,
        logits=z.astype(np.float32),
        teacher=y,
        input_ids=np.array([f"i{k}" for k in range(N)]),
        labels=np.array([f"l{k}" for k in range(K)]),
        teacher_model=np.array("gold-x"),
    )
    (directory / "report.json").write_text(json.dumps({"source": source}), encoding="utf-8")
    return npz


def test_main_from_logits_writes_a_smoke_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {"dataset": "banking77", "teacher_kind": "gold", "teacher_model": "gold-x"}
    npz = write_logits(tmp_path / "in", source)
    results = tmp_path / "results"
    monkeypatch.setattr(calibration, "RESULTS_DIR", results)
    code = main(["banking77", "--logits", str(npz), "--splits", "5", "--resamples", "50"])
    assert code == 0
    out = results / "calibration-banking77-gold-x-smoke"
    written = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert written["source"] == source
    assert written["n"] == N
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["splits"] == 5
    assert meta["from_logits"] == str(npz)
    assert meta["spent_usd"] == 0.0


def test_main_without_previous_report_is_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    npz = write_logits(tmp_path / "in", {})
    (npz.parent / "report.json").unlink()
    monkeypatch.setattr(calibration, "RESULTS_DIR", tmp_path / "results")
    assert main(["banking77", "--logits", str(npz), "--splits", "2", "--resamples", "10"]) == 2


def test_main_logits_takes_one_dataset(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["banking77", "ag_news", "--logits", str(tmp_path / "x.npz")])
    assert exc.value.code == 2


def test_render_calibration_without_runs() -> None:
    assert "No calibration run yet" in render_calibration([])


def test_render_calibration_of_a_report() -> None:
    source = {
        "dataset": "banking77",
        "teacher_model": "gold-x",
        "student_model": "tiny",
        "student_run": "run-1",
        "teacher_kind": "gold",
    }
    text = render_calibration(
        [("calibration-banking77-gold-x", {"source": source, **report()}, {"command": "cmd"})]
    )
    assert "banking77" in text
    assert "| Dataset | Teacher | Student |" in text
    assert "95% lower bound, calibrated" in text
    assert "The teacher here is the dataset's gold labels" in text


def test_find_student_refuses_two_students_without_recorded_settings(tmp_path: Path) -> None:
    import json

    from anvil.errors import ConfigError
    from anvil.pipeline import Workspace
    from anvil.spec import TaskSpec
    from anvil.students.encoder import META_FILE
    from eval.student import find_student

    spec = TaskSpec.model_validate(
        {"name": "t", "kind": "classify", "instructions": "x", "labels": ["a", "b"]}
    )
    ws = Workspace(tmp_path / "ws")
    base = {
        "teacher_model": "gold",
        "prompt_hash": spec.prompt_hash,
        "student_model": spec.student.model,
        "splits": spec.splits.model_dump(),
        "train_labels": ["a", "b"],
    }

    def save(run: str, **extra: object) -> None:
        (ws.students / run).mkdir(parents=True)
        (ws.students / run / META_FILE).write_text(json.dumps({**base, **extra}), "utf-8")

    save("20260101-old")
    assert find_student(ws, spec, "gold", 2) == ws.students / "20260101-old"
    save("20260102-seed1")
    with pytest.raises(ConfigError, match="record no seed"):
        find_student(ws, spec, "gold", 2)
    other = spec.student.model_copy(update={"seed": 1}).model_dump()
    save("20260103-new-seed1", student_spec=other)
    save("20260104-new-seed0", student_spec=spec.student.model_dump())
    assert find_student(ws, spec, "gold", 2) == ws.students / "20260104-new-seed0"
